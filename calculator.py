"""Compute financial ratios from the dictionary returned by data_fetch.fetch_financials."""

import json


def _safe_div(numerator, denominator, multiplier=1):
    """Divide numerator by denominator, returning None if either is missing or the
    denominator is zero (avoids ZeroDivisionError and propagates missing data)."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round((numerator / denominator) * multiplier, 2)


def _safe_combine(func, *values):
    """Apply func(*values) unless any value is None, in which case return None."""
    if any(value is None for value in values):
        return None
    return round(func(*values), 2)


def calculate_ratios(financials, market_price):
    """Compute profitability, liquidity, valuation, cash flow, and growth ratios.

    `financials` is the dict returned by data_fetch.fetch_financials. Any ratio whose
    required inputs are None, or whose denominator is zero, is returned as None.
    """
    f = financials

    revenue = f.get("revenue")
    net_income = f.get("net_income")
    gross_profit = f.get("gross_profit")
    operating_income = f.get("operating_income")
    total_assets = f.get("total_assets")
    shareholders_equity = f.get("shareholders_equity")
    current_assets = f.get("current_assets")
    current_liabilities = f.get("current_liabilities")
    long_term_debt = f.get("long_term_debt")
    interest_expense = f.get("interest_expense")
    eps = f.get("eps")
    shares_outstanding = f.get("shares_outstanding")
    operating_cash_flow = f.get("operating_cash_flow")
    capex = f.get("capex")
    revenue_py1 = f.get("revenue_prior_year_1")
    revenue_py2 = f.get("revenue_prior_year_2")
    net_income_py1 = f.get("net_income_prior_year_1")

    # market_cap: total market value of equity; the base for pb, ps, ev, and fcf_yield.
    market_cap = _safe_combine(lambda p, s: p * s, market_price, shares_outstanding)

    # free_cash_flow: cash left after maintaining/growing the asset base; funds
    # dividends, buybacks, and debt paydown.
    free_cash_flow = _safe_combine(lambda ocf, cx: ocf - cx, operating_cash_flow, capex)

    # ev (enterprise value): theoretical takeover cost, net of debt and working capital.
    ev = _safe_combine(
        lambda mc, ltd, ca, cl: mc + ltd - (ca - cl),
        market_cap,
        long_term_debt,
        current_assets,
        current_liabilities,
    )

    return {
        # gross_margin: % of revenue left after cost of goods sold; pricing power / cost control.
        "gross_margin": _safe_div(gross_profit, revenue, 100),
        # operating_margin: % of revenue left after operating expenses; core operating efficiency.
        "operating_margin": _safe_div(operating_income, revenue, 100),
        # net_margin: % of revenue that becomes bottom-line profit; overall profitability.
        "net_margin": _safe_div(net_income, revenue, 100),
        # roe: return generated on shareholders' invested capital.
        "roe": _safe_div(net_income, shareholders_equity, 100),
        # roa: return generated on the company's total asset base, regardless of financing.
        "roa": _safe_div(net_income, total_assets, 100),
        # current_ratio: ability to cover short-term liabilities with short-term assets.
        "current_ratio": _safe_div(current_assets, current_liabilities),
        # debt_to_equity: reliance on long-term debt versus shareholder capital; leverage risk.
        "debt_to_equity": _safe_div(long_term_debt, shareholders_equity),
        # interest_coverage: how many times over operating income can pay interest expense.
        "interest_coverage": _safe_div(operating_income, interest_expense),
        # pe_ratio: price paid per dollar of earnings; how expensive the stock is relative to profit.
        "pe_ratio": _safe_div(market_price, eps),
        # pb_ratio: price paid per dollar of book equity; premium over net asset value.
        "pb_ratio": _safe_div(market_cap, shareholders_equity),
        # ps_ratio: price paid per dollar of revenue; useful when earnings are unstable or negative.
        "ps_ratio": _safe_div(market_cap, revenue),
        # market_cap: total equity value the market currently assigns to the company.
        "market_cap": market_cap,
        # ev: enterprise value, the full acquisition cost including debt and cash/working capital.
        "ev": ev,
        # ev_to_ebitda: valuation multiple normalized for capital structure (uses operating income as EBITDA proxy).
        "ev_to_ebitda": _safe_div(ev, operating_income),
        # free_cash_flow: actual cash generated after capital investment; harder to manipulate than earnings.
        "free_cash_flow": free_cash_flow,
        # fcf_yield: free cash flow relative to market cap; cash return an investor is buying at current price.
        "fcf_yield": _safe_div(free_cash_flow, market_cap, 100),
        # revenue_growth_1y: year-over-year top-line growth, most recent year vs. prior year.
        "revenue_growth_1y": _safe_div(
            _safe_combine(lambda r, r1: r - r1, revenue, revenue_py1), revenue_py1, 100
        ),
        # revenue_growth_2y: top-line growth the year before that, for trend context.
        "revenue_growth_2y": _safe_div(
            _safe_combine(lambda r1, r2: r1 - r2, revenue_py1, revenue_py2), revenue_py2, 100
        ),
        # net_income_growth_1y: year-over-year bottom-line growth; profit momentum.
        "net_income_growth_1y": _safe_div(
            _safe_combine(lambda ni, ni1: ni - ni1, net_income, net_income_py1),
            net_income_py1,
            100,
        ),
    }


if __name__ == "__main__":
    from data_fetch import fetch_financials

    try:
        import yfinance as yf

        live_price = yf.Ticker("AAPL").fast_info["lastPrice"]
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch live AAPL price from yfinance: {exc}") from exc

    financials = fetch_financials("AAPL")
    ratios = calculate_ratios(financials, live_price)

    print(f"AAPL market price used: {live_price}")
    print(json.dumps(ratios, indent=2))
