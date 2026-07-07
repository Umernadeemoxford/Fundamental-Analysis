"""Residual Income Model — intrinsic value for financial services companies.

A DCF doesn't work well for banks/insurers (their "debt" is their raw material,
not a financing choice, so free cash flow doesn't mean the same thing). The
residual income model instead starts from book value per share and asks: in
each future year, does the company earn MORE than shareholders' required
return on that book value (residual income), or less? Compounding that excess
(or shortfall) forward and discounting it back gives an intrinsic value that
works naturally for balance-sheet-driven businesses.
"""

import json

import numpy as np
import pandas as pd
import yfinance as yf

REQUIRED_FIELDS = [
    "net_income",
    "shareholders_equity",
    "shares_outstanding",
    "net_income_prior_year_1",
    "net_income_prior_year_2",
]

DEFAULT_PAYOUT_RATIO = 0.40
MAX_PAYOUT_RATIO = 0.90
MAX_EARNINGS_GROWTH_RATE = 0.20
EQUITY_RISK_PREMIUM = 0.05
TERMINAL_GROWTH_RATE = 0.03


def _get_sector(ticker):
    """Best-effort sector lookup; returns None (never raises) if it can't be found."""
    try:
        return yf.Ticker(ticker).info.get("sector")
    except Exception:
        return None


def _get_risk_free_rate():
    """Live 10-year Treasury yield (^TNX) as a decimal; same approach as dcf.py.
    Falls back to 4% if the fetch fails."""
    try:
        return yf.Ticker("^TNX").fast_info["lastPrice"] / 100
    except Exception:
        return 0.04


def _compute_beta_regression(ticker):
    """Beta via OLS regression of the stock's weekly returns against the S&P
    500's weekly returns over the trailing 2 years: beta = Cov(stock, market) /
    Var(market). Returns None (never raises) if there isn't enough overlapping
    price history to regress on.
    """
    try:
        stock_hist = yf.Ticker(ticker).history(period="2y", interval="1wk")
        market_hist = yf.Ticker("^GSPC").history(period="2y", interval="1wk")
    except Exception:
        return None

    if stock_hist.empty or market_hist.empty:
        return None

    stock_returns = stock_hist["Close"].pct_change().dropna()
    market_returns = market_hist["Close"].pct_change().dropna()
    stock_aligned, market_aligned = stock_returns.align(market_returns, join="inner")

    if len(stock_aligned) < 10:
        return None

    covariance_matrix = np.cov(stock_aligned.values, market_aligned.values)
    market_variance = covariance_matrix[1, 1]
    if market_variance == 0:
        return None

    return float(covariance_matrix[0, 1] / market_variance)


def _get_historical_payout_ratio(ticker):
    """Average dividend payout ratio (dividends paid / net income) over the
    most recent 3 fiscal years available from yfinance. The payout ratio is
    the share of earnings a company hands back to shareholders as dividends
    rather than reinvesting — the rest (the "plowback") compounds book value
    forward, which is what drives this model's forecast.

    Returns (payout_ratio, source):
    - "yfinance_cashflow": real dividends-paid/net-income data was available.
    - "no_dividends_paid": the company genuinely pays no dividends (e.g.
      Berkshire Hathaway deliberately retains all earnings — payout ratio of
      0 means all earnings are reinvested, i.e. plowback ratio = 1.0).
    - "fallback_default_data_error": the dividend fetch itself failed or
      returned malformed data, so 0.40 is used as a generic stand-in — this is
      NOT used just because dividends happen to be zero.
    """
    ratios = []
    try:
        cashflow = yf.Ticker(ticker).cashflow
        income_stmt = yf.Ticker(ticker).financials
        dividends_paid = cashflow.loc["Cash Dividends Paid"]
        net_income_series = income_stmt.loc["Net Income"]

        common_dates = sorted(set(dividends_paid.index) & set(net_income_series.index), reverse=True)[:3]
        for date in common_dates:
            dividend = dividends_paid.get(date)
            net_income = net_income_series.get(date)
            if dividend is None or net_income is None or net_income == 0:
                continue
            if dividend != dividend or net_income != net_income:  # NaN check
                continue
            ratios.append(abs(dividend) / net_income)
    except Exception:
        pass

    if ratios:
        return min(sum(ratios) / len(ratios), MAX_PAYOUT_RATIO), "yfinance_cashflow"

    # No usable ratio from the cashflow statement — check the raw dividend
    # history directly to tell "genuinely pays nothing" apart from "the data
    # just wasn't available", rather than assuming 0.40 either way.
    try:
        dividend_history = yf.Ticker(ticker).dividends
    except Exception:
        return DEFAULT_PAYOUT_RATIO, "fallback_default_data_error"

    if dividend_history.empty:
        return 0.0, "no_dividends_paid"

    cutoff = pd.Timestamp.now(tz=dividend_history.index.tz) - pd.DateOffset(years=3)
    recent_dividends_sum = dividend_history[dividend_history.index >= cutoff].sum()
    if recent_dividends_sum == 0:
        return 0.0, "no_dividends_paid"

    return DEFAULT_PAYOUT_RATIO, "fallback_default_data_error"


def run_residual_income(financials, market_price):
    """Estimate intrinsic value per share for a financial services company
    using a 5-year residual income model built from `financials` (the dict
    returned by data_fetch.fetch_financials_financial_services) and a live
    `market_price`.
    """
    ticker = financials.get("ticker")
    sector = _get_sector(ticker) if ticker else None
    if sector != "Financial Services":
        return {"error": "Residual Income Model only applicable for financial services companies"}

    missing = [field for field in REQUIRED_FIELDS if financials.get(field) is None]
    if missing:
        return {"error": f"insufficient data — missing: {missing}"}

    net_income = financials["net_income"]
    shareholders_equity = financials["shareholders_equity"]
    shares_outstanding = financials["shares_outstanding"]
    net_income_py1 = financials["net_income_prior_year_1"]
    net_income_py2 = financials["net_income_prior_year_2"]

    if not shares_outstanding:
        return {"error": "insufficient data — missing: ['shares_outstanding']"}

    # ============================================================
    # STEP 2: Historical dividend payout ratio (see docstring above).
    # ============================================================
    historical_payout_ratio, payout_ratio_source = _get_historical_payout_ratio(ticker)

    # ============================================================
    # STEP 3: Derived assumptions.
    # ============================================================
    # ROE: how much profit the company generates per dollar of book equity —
    # the engine that compounds book value forward each year below.
    roe = net_income / shareholders_equity

    # How fast has net income actually been growing the last 2 years, on
    # average? Capped at 20% so one unusually strong year can't run away.
    growth_last_year = (net_income - net_income_py1) / net_income_py1 if net_income_py1 else 0.0
    growth_year_before = (net_income_py1 - net_income_py2) / net_income_py2 if net_income_py2 else 0.0
    earnings_growth_rate = min((growth_last_year + growth_year_before) / 2, MAX_EARNINGS_GROWTH_RATE)

    # Cost of equity (CAPM, no fallback if beta can't be computed): the annual
    # return shareholders require for the risk of owning this company — the
    # bar residual income has to clear.
    risk_free_rate = _get_risk_free_rate()
    beta = _compute_beta_regression(ticker)
    if beta is None:
        return {"error": "insufficient data — missing: ['beta']"}
    cost_of_equity = risk_free_rate + beta * EQUITY_RISK_PREMIUM

    terminal_growth_rate = TERMINAL_GROWTH_RATE

    # Plowback ratio: the flip side of the payout ratio — the share of
    # earnings retained and reinvested rather than paid out as dividends.
    plowback_ratio = 1 - historical_payout_ratio

    # ============================================================
    # STEP 4: 5-year forecast.
    # Each year, book value per share grows by retained earnings (EPS minus
    # dividends), and "residual income" measures how much that year's EPS
    # exceeded (or fell short of) what shareholders required on the book
    # value they had going into the year.
    # ============================================================
    book_value_per_share = shareholders_equity / shares_outstanding
    starting_book_value_per_share = book_value_per_share

    forecasted_eps = []
    forecasted_book_values = []
    forecasted_dividends = []
    forecasted_residual_incomes = []

    prior_book_value_per_share = book_value_per_share
    for _year in range(1, 6):
        eps_year = prior_book_value_per_share * roe
        dividends_year = eps_year * historical_payout_ratio
        next_book_value_per_share = prior_book_value_per_share + eps_year - dividends_year
        residual_income_year = eps_year - (cost_of_equity * prior_book_value_per_share)

        forecasted_eps.append(eps_year)
        forecasted_book_values.append(next_book_value_per_share)
        forecasted_dividends.append(dividends_year)
        forecasted_residual_incomes.append(residual_income_year)

        prior_book_value_per_share = next_book_value_per_share

    # ============================================================
    # STEP 5: Terminal value — the lump-sum value of every dollar of residual
    # income generated after year 5, forever, assuming it settles into steady
    # terminal growth.
    # ============================================================
    terminal_residual_value = (
        forecasted_residual_incomes[-1] * (1 + terminal_growth_rate) / (cost_of_equity - terminal_growth_rate)
    )

    # ============================================================
    # STEP 6: Discount everything back to today and add starting book value.
    # Intrinsic value = what the company is worth today (book value) plus the
    # present value of every dollar of "extra" profit it's expected to
    # generate above what shareholders require.
    # ============================================================
    present_value_residual_incomes = [
        ri / (1 + cost_of_equity) ** year for year, ri in enumerate(forecasted_residual_incomes, start=1)
    ]
    present_value_terminal = terminal_residual_value / (1 + cost_of_equity) ** 5

    intrinsic_value_per_share = (
        sum(present_value_residual_incomes) + present_value_terminal + starting_book_value_per_share
    )

    # ============================================================
    # STEP 7: Compare the model's estimate to what the market is charging.
    # ============================================================
    if market_price < intrinsic_value_per_share * 0.85:
        verdict = "UNDERVALUED"
    elif market_price > intrinsic_value_per_share * 1.15:
        verdict = "OVERVALUED"
    else:
        verdict = "FAIRLY VALUED"

    margin_of_safety = (
        (intrinsic_value_per_share - market_price) / intrinsic_value_per_share * 100
        if intrinsic_value_per_share
        else None
    )

    return {
        "intrinsic_value_per_share": round(intrinsic_value_per_share, 2),
        "market_price": round(market_price, 2),
        "verdict": verdict,
        "margin_of_safety": round(margin_of_safety, 2) if margin_of_safety is not None else None,
        "cost_of_equity": round(cost_of_equity, 4),
        "beta": round(beta, 4),
        "risk_free_rate": round(risk_free_rate, 4),
        "terminal_growth_rate": terminal_growth_rate,
        "roe": round(roe, 4),
        "earnings_growth_rate": round(earnings_growth_rate, 4),
        "historical_payout_ratio": round(historical_payout_ratio, 4),
        "payout_ratio_source": payout_ratio_source,
        "plowback_ratio": round(plowback_ratio, 4),
        "book_value_per_share": round(starting_book_value_per_share, 2),
        "forecasted_eps": [round(v, 2) for v in forecasted_eps],
        "forecasted_book_values": [round(v, 2) for v in forecasted_book_values],
        "forecasted_dividends": [round(v, 2) for v in forecasted_dividends],
        "forecasted_residual_incomes": [round(v, 2) for v in forecasted_residual_incomes],
        "terminal_residual_value": round(terminal_residual_value, 2),
        "shares_outstanding": shares_outstanding,
    }


if __name__ == "__main__":
    from calculator_financial_services import calculate_financial_ratios
    from data_fetch import fetch_financials_financial_services

    # --- 1. JPM: full pipeline, bank ratios + residual income. ------------
    print("=" * 60)
    print("JPM - full pipeline")
    print("=" * 60)
    jpm_financials = fetch_financials_financial_services("JPM")
    jpm_price = yf.Ticker("JPM").fast_info["lastPrice"]
    jpm_ratios = calculate_financial_ratios(jpm_financials, jpm_price)
    jpm_result = run_residual_income(jpm_financials, jpm_price)

    print("Ratios:")
    print(json.dumps(jpm_ratios, indent=2))
    print()
    print("Residual income model:")
    print(json.dumps(jpm_result, indent=2))
    if "error" not in jpm_result:
        print()
        print(f"Intrinsic value per share : ${jpm_result['intrinsic_value_per_share']:.2f}")
        print(f"Current market price      : ${jpm_result['market_price']:.2f}")
        print(f"Verdict                   : {jpm_result['verdict']}")
        print(f"Margin of safety          : {jpm_result['margin_of_safety']:.2f}%")

    # --- 2. BRK-B: insurance ratios. ---------------------------------------
    print()
    print("=" * 60)
    print("BRK-B - insurance ratios + residual income")
    print("=" * 60)
    brk_financials = fetch_financials_financial_services("BRK-B")
    brk_price = yf.Ticker("BRK-B").fast_info["lastPrice"]
    brk_ratios = calculate_financial_ratios(brk_financials, brk_price)
    brk_result = run_residual_income(brk_financials, brk_price)

    print("Ratios:")
    print(json.dumps(brk_ratios, indent=2))
    print()
    print("Residual income model:")
    print(json.dumps(brk_result, indent=2))

    # --- 3. AAPL: confirm the non-financial-services block triggers. ------
    print()
    print("=" * 60)
    print("AAPL - should be blocked (not Financial Services)")
    print("=" * 60)
    from data_fetch import fetch_financials

    aapl_financials = fetch_financials("AAPL")
    aapl_price = yf.Ticker("AAPL").fast_info["lastPrice"]
    aapl_result = run_residual_income(aapl_financials, aapl_price)
    print(json.dumps(aapl_result, indent=2))
