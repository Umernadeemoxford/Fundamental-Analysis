"""Compute bank and insurance financial ratios from the dictionary returned by
data_fetch.fetch_financials_financial_services.
"""

import json

import yfinance as yf


def _safe_div(numerator, denominator, multiplier=1):
    """Divide, returning None if either operand is missing or the denominator
    is zero (avoids ZeroDivisionError and propagates missing data)."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round((numerator / denominator) * multiplier, 2)


def _safe_sum(*values):
    if any(value is None for value in values):
        return None
    return sum(values)


def _safe_sub(*values):
    if any(value is None for value in values):
        return None
    result = values[0]
    for value in values[1:]:
        result -= value
    return result


def _safe_mul(*values):
    if any(value is None for value in values):
        return None
    result = 1
    for value in values:
        result *= value
    return result


def _safe_add(a, b):
    if a is None or b is None:
        return None
    return round(a + b, 2)


def _get_dividend_yield(ticker):
    """Live dividend yield from yfinance. Note: current yfinance versions
    already return this as a percentage (e.g. 1.79 meaning 1.79%), not a
    decimal fraction — so it's rounded and returned as-is, not multiplied by
    100 (multiplying again would misreport a ~1.8% yield as ~180%)."""
    if not ticker:
        return None
    try:
        value = yf.Ticker(ticker).info.get("dividendYield")
    except Exception:
        return None
    return round(value, 2) if isinstance(value, (int, float)) else None


def calculate_financial_ratios(financials, market_price):
    """Compute bank and/or insurance financial ratios from `financials` (the
    dict returned by data_fetch.fetch_financials_financial_services) and a
    live `market_price`. Any ratio whose inputs are None or whose denominator
    is zero is returned as None — this is expected for ratios that don't apply
    to the company's sub_type (e.g. loss_ratio for a bank).
    """
    f = financials

    net_interest_income = f.get("net_interest_income")
    net_interest_income_py1 = f.get("net_interest_income_prior_year_1")
    total_loans = f.get("total_loans")
    total_deposits = f.get("total_deposits")
    noninterest_expense = f.get("noninterest_expense")
    noninterest_income = f.get("noninterest_income")
    provision_for_loan_losses = f.get("provision_for_loan_losses")
    nonperforming_loans_proxy = f.get("nonperforming_loans_proxy")
    allowance_for_loan_losses = f.get("allowance_for_loan_losses")
    tier1_capital = f.get("tier1_capital")
    risk_weighted_assets = f.get("risk_weighted_assets")
    net_income = f.get("net_income")
    net_income_py1 = f.get("net_income_prior_year_1")
    net_income_py2 = f.get("net_income_prior_year_2")
    total_assets = f.get("total_assets")
    shareholders_equity = f.get("shareholders_equity")
    shares_outstanding = f.get("shares_outstanding")
    eps = f.get("eps")
    premiums_earned = f.get("premiums_earned")
    premiums_earned_py1 = f.get("premiums_earned_prior_year_1")
    premiums_earned_py2 = f.get("premiums_earned_prior_year_2")
    claims_incurred = f.get("claims_incurred")
    investment_income = f.get("investment_income")
    underwriting_expenses = f.get("underwriting_expenses")
    insurance_reserves = f.get("insurance_reserves")

    market_cap = _safe_mul(market_price, shares_outstanding)
    total_revenue_proxy = _safe_sum(net_interest_income, noninterest_income)

    # roe: return on shareholders' equity — the primary profitability metric
    # for both banks and insurers.
    roe = _safe_div(net_income, shareholders_equity, 100)

    # roa: return on total assets — how efficiently the balance sheet
    # generates profit, independent of how it's financed.
    roa = _safe_div(net_income, total_assets, 100)

    # nim: net interest margin — the core spread a bank earns between what it
    # pays depositors and what it charges borrowers.
    # True NIM requires average earning assets — using total assets as proxy.
    nim = _safe_div(net_interest_income, total_assets, 100)

    # efficiency_ratio: cost to generate $1 of revenue — lower is better; a
    # classic bank-management-quality gauge.
    efficiency_ratio = _safe_div(noninterest_expense, total_revenue_proxy, 100)

    # cost_to_income: an alternative efficiency lens, expenses relative to the
    # bottom line rather than to revenue.
    cost_to_income = _safe_div(noninterest_expense, net_income, 100)

    # npl_ratio: non-performing loans as a % of total loans — the headline
    # credit-risk indicator; rising NPLs signal deteriorating loan quality.
    npl_ratio = _safe_div(nonperforming_loans_proxy, total_loans, 100)

    # coverage_ratio: how well the loan-loss allowance is provisioned against
    # the bad loans already on the books.
    coverage_ratio = _safe_div(allowance_for_loan_losses, nonperforming_loans_proxy, 100)

    # provision_to_loans: new provisions set aside this year as a % of the
    # loan book — a rising trend signals management expects credit to worsen.
    provision_to_loans = _safe_div(provision_for_loan_losses, total_loans, 100)

    # loan_loss_rate: new provisions relative to the whole balance sheet, a
    # broader view of how much credit risk is weighing on the bank.
    loan_loss_rate = _safe_div(provision_for_loan_losses, total_assets, 100)

    # tier1_capital_ratio: core regulatory capital cushion against
    # risk-weighted assets — regulators require a minimum of roughly 6%.
    tier1_capital_ratio = _safe_div(tier1_capital, risk_weighted_assets, 100)

    # equity_to_assets: a simple, regulator-agnostic leverage measure — how
    # much of the balance sheet is funded by shareholders vs. borrowed money.
    equity_to_assets = _safe_div(shareholders_equity, total_assets, 100)

    # ldr: loan-to-deposit ratio — above 100% means the bank is lending more
    # than it holds in deposits, implying reliance on wholesale/other funding.
    ldr = _safe_div(total_loans, total_deposits, 100)

    # casa_ratio: the share of deposits sitting in low-cost checking/savings
    # accounts vs. higher-cost term deposits — cannot be computed from EDGAR's
    # XBRL data, which doesn't break deposits down by type.
    casa_ratio = None
    casa_ratio_note = "CASA requires deposit breakdown not available in EDGAR XBRL"

    # pe_ratio: price paid per dollar of trailing earnings.
    pe_ratio = _safe_div(market_price, eps)

    # pb_ratio: price paid per dollar of book equity — the primary valuation
    # metric for banks, since their assets/liabilities are mostly marked at
    # or near fair value already.
    pb_ratio = _safe_div(market_cap, shareholders_equity)

    # ps_ratio: price relative to total revenue (net interest income +
    # noninterest income), a bank-appropriate stand-in for "sales".
    ps_ratio = _safe_div(market_cap, total_revenue_proxy)

    # dividend_yield: cash return to shareholders via dividends at the
    # current price (see _get_dividend_yield for the scaling caveat).
    dividend_yield = _get_dividend_yield(f.get("ticker"))

    # net_income_growth_1y / _2y: year-over-year profit momentum.
    net_income_growth_1y = _safe_div(_safe_sub(net_income, net_income_py1), net_income_py1, 100)
    net_income_growth_2y = _safe_div(_safe_sub(net_income_py1, net_income_py2), net_income_py2, 100)

    # nii_growth_1y: growth in the core lending spread business, separate
    # from noninterest income swings.
    nii_growth_1y = _safe_div(
        _safe_sub(net_interest_income, net_interest_income_py1), net_interest_income_py1, 100
    )

    # loss_ratio: claims paid out as a % of premiums collected — below ~60%
    # is generally considered good underwriting.
    loss_ratio = _safe_div(claims_incurred, premiums_earned, 100)

    # expense_ratio: operating costs to run the insurance business as a % of
    # premiums.
    expense_ratio = _safe_div(underwriting_expenses, premiums_earned, 100)

    # combined_ratio: loss_ratio + expense_ratio — below 100% means the
    # insurer made an underwriting profit before investment income; above
    # 100% means it lost money on underwriting alone.
    combined_ratio = _safe_add(loss_ratio, expense_ratio)

    # investment_yield: return earned by investing the float (reserves).
    investment_yield = _safe_div(investment_income, insurance_reserves, 100)

    # float_proxy: float is policyholder money the insurer holds and invests
    # before it has to pay claims — Buffett's key insurance metric, since it's
    # effectively interest-free (or even negative-cost) leverage.
    float_proxy = insurance_reserves

    # float_to_equity: how much leverage the float itself adds on top of
    # shareholders' own capital.
    float_to_equity = _safe_div(insurance_reserves, shareholders_equity, 100)

    # underwriting_profit_margin: profit purely from underwriting (premiums
    # minus claims minus expenses), before investment income.
    underwriting_profit_margin = _safe_div(
        _safe_sub(premiums_earned, claims_incurred, underwriting_expenses), premiums_earned, 100
    )

    # premium_growth_1y / _2y: top-line momentum for an insurer.
    premium_growth_1y = _safe_div(_safe_sub(premiums_earned, premiums_earned_py1), premiums_earned_py1, 100)
    premium_growth_2y = _safe_div(
        _safe_sub(premiums_earned_py1, premiums_earned_py2), premiums_earned_py2, 100
    )

    # p_to_float: Buffett-style insurance valuation — price paid per dollar of
    # float the company controls, rather than per dollar of book value.
    p_to_float = _safe_div(market_cap, insurance_reserves)

    return {
        "sub_type": f.get("sub_type"),
        # Bank ratios
        "roe": roe,
        "roa": roa,
        "nim": nim,
        "efficiency_ratio": efficiency_ratio,
        "cost_to_income": cost_to_income,
        "npl_ratio": npl_ratio,
        "coverage_ratio": coverage_ratio,
        "provision_to_loans": provision_to_loans,
        "loan_loss_rate": loan_loss_rate,
        "tier1_capital_ratio": tier1_capital_ratio,
        "equity_to_assets": equity_to_assets,
        "ldr": ldr,
        "casa_ratio": casa_ratio,
        "casa_ratio_note": casa_ratio_note,
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
        "ps_ratio": ps_ratio,
        "dividend_yield": dividend_yield,
        "net_income_growth_1y": net_income_growth_1y,
        "net_income_growth_2y": net_income_growth_2y,
        "nii_growth_1y": nii_growth_1y,
        # Insurance ratios
        "loss_ratio": loss_ratio,
        "expense_ratio": expense_ratio,
        "combined_ratio": combined_ratio,
        "investment_yield": investment_yield,
        "float_proxy": float_proxy,
        "float_to_equity": float_to_equity,
        "underwriting_profit_margin": underwriting_profit_margin,
        "premium_growth_1y": premium_growth_1y,
        "premium_growth_2y": premium_growth_2y,
        "p_to_float": p_to_float,
    }


if __name__ == "__main__":
    from data_fetch import fetch_financials_financial_services

    for ticker in ["JPM", "BRK-B"]:
        financials = fetch_financials_financial_services(ticker)
        price = yf.Ticker(ticker).fast_info["lastPrice"]
        ratios = calculate_financial_ratios(financials, price)
        print(f"=== {ticker} ({ratios['sub_type']}) - price ${price:.2f} ===")
        print(json.dumps(ratios, indent=2))
        print()
