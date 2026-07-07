"""Fetch company financial statement data from the SEC XBRL frames API."""

import json

import requests
import yfinance as yf

SEC_HEADERS = {"User-Agent": "Umer Nadeem umer.nadeem.mba25@said.oxford.edu"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def _entries_for_tags(us_gaap, tags, unit_keys):
    """Return the deduped, date-descending list of annual 10-K entries across all of
    `tags` (companies sometimes switch XBRL tags for the same concept mid-history,
    e.g. Revenues -> RevenueFromContractWithCustomerExcludingAssessedTax, so entries
    from every listed tag are merged into one continuous series)."""
    by_end = {}
    for tag in tags:
        fact = us_gaap.get(tag)
        if not fact:
            continue
        units = fact.get("units", {})
        for unit_key in unit_keys:
            entries = units.get(unit_key)
            if not entries:
                continue
            for entry in entries:
                if entry.get("form") != "10-K" or not entry.get("end"):
                    continue
                # A value for the same fiscal year end can appear in more than one
                # filing (e.g. as a prior-year comparative); keep the latest-filed one.
                end = entry["end"]
                if end not in by_end or entry.get("filed", "") > by_end[end].get("filed", ""):
                    by_end[end] = entry
    return sorted(by_end.values(), key=lambda e: e["end"], reverse=True)


def _latest_value(us_gaap, tags, unit_keys=("USD",)):
    """Most recent annual 10-K value for the given tag(s), or None if unavailable."""
    entries = _entries_for_tags(us_gaap, tags, unit_keys)
    return entries[0]["val"] if entries else None


def _three_year_values(us_gaap, tags, unit_keys=("USD",)):
    """The three most recent annual 10-K values for the given tag(s), padded with
    None if fewer than three years are available."""
    entries = _entries_for_tags(us_gaap, tags, unit_keys)
    values = [entry["val"] for entry in entries[:3]]
    while len(values) < 3:
        values.append(None)
    return values


def _latest_end_date(us_gaap, tags, unit_keys=("USD",)):
    entries = _entries_for_tags(us_gaap, tags, unit_keys)
    return entries[0]["end"] if entries else None


def _value_at_index(us_gaap, tags, index, unit_keys=("USD",)):
    """Annual 10-K value at the given position in the date-descending series
    (index 0 = most recent, 1 = prior year, ...), or None if unavailable."""
    entries = _entries_for_tags(us_gaap, tags, unit_keys)
    return entries[index]["val"] if len(entries) > index else None


def fetch_financials(ticker):
    """Fetch key annual (10-K) financial statement figures for `ticker` from SEC EDGAR.

    Raises ValueError if the ticker cannot be resolved to a CIK, and RuntimeError if
    either SEC request fails. Individual missing fields are returned as None rather
    than raising.
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("ticker must be a non-empty string")

    try:
        response = requests.get(TICKERS_URL, headers=SEC_HEADERS, timeout=15)
        response.raise_for_status()
        ticker_map = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Failed to fetch SEC ticker list: {exc}") from exc

    ticker_upper = ticker.upper()
    cik = None
    for entry in ticker_map.values():
        if str(entry.get("ticker", "")).upper() == ticker_upper:
            cik = entry.get("cik_str")
            break

    if cik is None:
        raise ValueError(f"Ticker '{ticker}' not found in SEC company tickers list")

    padded_cik = str(cik).zfill(10)

    try:
        response = requests.get(
            COMPANY_FACTS_URL.format(cik=padded_cik), headers=SEC_HEADERS, timeout=15
        )
        response.raise_for_status()
        company_facts = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Failed to fetch company facts for CIK {padded_cik}: {exc}") from exc

    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})

    # Revenue: top line for growth rate, gross/net margin, and price-to-sales.
    revenue, revenue_py1, revenue_py2 = _three_year_values(
        us_gaap, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]
    )

    # Net income: feeds net margin, ROE, ROA, and EPS/P-E based ratios.
    net_income, net_income_py1, net_income_py2 = _three_year_values(us_gaap, ["NetIncomeLoss"])

    # Gross profit: feeds gross margin (gross profit / revenue).
    gross_profit, gross_profit_py1, gross_profit_py2 = _three_year_values(us_gaap, ["GrossProfit"])

    # Operating income: feeds operating margin (operating income / revenue).
    operating_income = _latest_value(us_gaap, ["OperatingIncomeLoss"])

    # Total assets: feeds ROA and asset turnover ratios.
    total_assets = _latest_value(us_gaap, ["Assets"])

    # Total liabilities: feeds liabilities-to-assets and solvency ratios.
    total_liabilities = _latest_value(us_gaap, ["Liabilities"])

    # Shareholders' equity: feeds ROE, debt-to-equity, and book value per share.
    shareholders_equity = _latest_value(
        us_gaap,
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    )

    # Current assets: feeds the current ratio, quick ratio, and working capital.
    current_assets = _latest_value(us_gaap, ["AssetsCurrent"])

    # Current liabilities: feeds the current ratio, quick ratio, and working capital.
    current_liabilities = _latest_value(us_gaap, ["LiabilitiesCurrent"])

    # Long-term debt: feeds debt-to-equity and long-term leverage ratios.
    long_term_debt = _latest_value(us_gaap, ["LongTermDebt"])

    # Operating cash flow: feeds cash flow margin, free cash flow, and cash conversion.
    operating_cash_flow, operating_cash_flow_py1, operating_cash_flow_py2 = _three_year_values(
        us_gaap, ["NetCashProvidedByUsedInOperatingActivities"]
    )

    # Capex: feeds free cash flow (FCF = operating cash flow - capex).
    capex = _latest_value(us_gaap, ["PaymentsToAcquirePropertyPlantAndEquipment"])

    # Shares outstanding: feeds EPS, market cap, and book value per share.
    shares_outstanding = _latest_value(us_gaap, ["CommonStockSharesOutstanding"], unit_keys=("shares",))

    # EPS: feeds the price-to-earnings (P/E) ratio.
    eps = _latest_value(us_gaap, ["EarningsPerShareBasic"], unit_keys=("USD/shares",))

    # Interest expense: feeds the interest coverage ratio (EBIT / interest expense).
    interest_expense = _latest_value(us_gaap, ["InterestExpense"])

    # Depreciation & amortisation: non-cash charge added back in FCFF calculation.
    depreciation_amortisation = _latest_value(us_gaap, ["DepreciationDepletionAndAmortization"])

    # Prior-year current assets: needed to compute change in net working capital for FCFF.
    current_assets_prior_year = _value_at_index(us_gaap, ["AssetsCurrent"], 1)

    # Prior-year current liabilities: needed to compute change in net working capital for FCFF.
    current_liabilities_prior_year = _value_at_index(us_gaap, ["LiabilitiesCurrent"], 1)

    fiscal_year_end = _latest_end_date(
        us_gaap, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]
    ) or _latest_end_date(us_gaap, ["Assets"])

    result = {
        "ticker": ticker_upper,
        "cik": padded_cik,
        "fiscal_year_end": fiscal_year_end,
        "revenue": revenue,
        "revenue_prior_year_1": revenue_py1,
        "revenue_prior_year_2": revenue_py2,
        "net_income": net_income,
        "net_income_prior_year_1": net_income_py1,
        "net_income_prior_year_2": net_income_py2,
        "gross_profit": gross_profit,
        "gross_profit_prior_year_1": gross_profit_py1,
        "gross_profit_prior_year_2": gross_profit_py2,
        "operating_income": operating_income,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "shareholders_equity": shareholders_equity,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "long_term_debt": long_term_debt,
        "operating_cash_flow": operating_cash_flow,
        "operating_cash_flow_prior_year_1": operating_cash_flow_py1,
        "operating_cash_flow_prior_year_2": operating_cash_flow_py2,
        "capex": capex,
        "shares_outstanding": shares_outstanding,
        "eps": eps,
        "interest_expense": interest_expense,
        "depreciation_amortisation": depreciation_amortisation,
        "current_assets_prior_year": current_assets_prior_year,
        "current_liabilities_prior_year": current_liabilities_prior_year,
    }

    # EDGAR's CommonStockSharesOutstanding tag is missing/unreliable for some
    # companies (dual-class structures, foreign filers, certain financials) -
    # fall back across yfinance and computed sources rather than leaving it None.
    from data_utils import get_shares_outstanding

    shares_outstanding, shares_outstanding_source = get_shares_outstanding(ticker_upper, result)
    result["shares_outstanding"] = shares_outstanding
    result["shares_outstanding_source"] = shares_outstanding_source

    return result


def _get_financial_services_sub_type(ticker):
    """Bank vs. insurance vs. other financial-services sub-type, inferred from
    yfinance's industry string (e.g. "Banks - Diversified", "Insurance -
    Diversified"). Never raises; defaults to "financial_services_other".
    """
    try:
        industry = (yf.Ticker(ticker).info.get("industry") or "").lower()
    except Exception:
        industry = ""
    if "bank" in industry:
        return "bank"
    if "insurance" in industry:
        return "insurance"
    return "financial_services_other"


def fetch_financials_financial_services(ticker):
    """Fetch bank/insurance-specific annual (10-K) figures for `ticker` from SEC
    EDGAR — same CIK lookup, error handling, and never-crash-on-missing-field
    pattern as fetch_financials, but pulls industry-specific XBRL tags (net
    interest income, loan quality, regulatory capital, insurance float, etc.)
    instead of the generic income-statement/balance-sheet fields. Bank and
    insurance tags are both attempted regardless of `sub_type` — tags that don't
    apply to a given company simply come back None.

    Raises ValueError if the ticker cannot be resolved to a CIK, and RuntimeError
    if either SEC request fails. Individual missing fields are returned as None.
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("ticker must be a non-empty string")

    try:
        response = requests.get(TICKERS_URL, headers=SEC_HEADERS, timeout=15)
        response.raise_for_status()
        ticker_map = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Failed to fetch SEC ticker list: {exc}") from exc

    ticker_upper = ticker.upper()
    cik = None
    for entry in ticker_map.values():
        if str(entry.get("ticker", "")).upper() == ticker_upper:
            cik = entry.get("cik_str")
            break

    if cik is None:
        raise ValueError(f"Ticker '{ticker}' not found in SEC company tickers list")

    padded_cik = str(cik).zfill(10)

    try:
        response = requests.get(
            COMPANY_FACTS_URL.format(cik=padded_cik), headers=SEC_HEADERS, timeout=15
        )
        response.raise_for_status()
        company_facts = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Failed to fetch company facts for CIK {padded_cik}: {exc}") from exc

    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})

    # --- Bank tags -----------------------------------------------------

    # Net interest income: core spread income for banks; feeds NIM.
    net_interest_income, net_interest_income_py1, net_interest_income_py2 = _three_year_values(
        us_gaap, ["InterestAndFeeIncomeLoansAndLeases", "InterestIncomeExpenseAfterProvisionForLoss"]
    )

    # Total loans: feeds NPL ratio, loan-to-deposit ratio, provision-to-loans.
    total_loans = _latest_value(us_gaap, ["LoansAndLeasesReceivableNetReported", "LoansAndLeasesReceivableGross"])

    # Total deposits: feeds the loan-to-deposit liquidity ratio.
    total_deposits = _latest_value(us_gaap, ["Deposits"])

    # Noninterest expense: feeds the efficiency ratio and cost-to-income.
    noninterest_expense = _latest_value(us_gaap, ["NoninterestExpense"])

    # Noninterest income: feeds the efficiency ratio and the price-to-sales proxy.
    noninterest_income = _latest_value(us_gaap, ["NoninterestIncome"])

    # Provision for loan losses: feeds provision-to-loans and loan-loss-rate.
    provision_for_loan_losses = _latest_value(
        us_gaap, ["ProvisionForLoanAndLeaseLosses", "ProvisionForLoanLeaseAndOtherLosses"]
    )

    # Nonperforming loans proxy: feeds the NPL ratio and coverage ratio (credit risk).
    nonperforming_loans_proxy = _latest_value(
        us_gaap, ["ImpairedFinancingReceivableRecordedInvestment", "FinancingReceivableNonaccrual"]
    )

    # Allowance for loan losses: feeds the coverage ratio (loss-absorbing cushion).
    allowance_for_loan_losses = _latest_value(us_gaap, ["AllowanceForLoanAndLeaseLosses"])

    # Tier 1 capital: feeds the Tier 1 capital ratio (regulatory capital adequacy).
    tier1_capital = _latest_value(us_gaap, ["TierOneRiskBasedCapital"])

    # Risk-weighted assets: feeds the Tier 1 capital ratio's denominator.
    risk_weighted_assets = _latest_value(us_gaap, ["RiskWeightedAssets"])

    # Net income: feeds ROE, ROA, cost-to-income, EPS/P-E, and growth rates.
    net_income, net_income_py1, net_income_py2 = _three_year_values(us_gaap, ["NetIncomeLoss"])

    # Total assets: feeds ROA, the NIM proxy, equity-to-assets, and loan-loss-rate.
    total_assets = _latest_value(us_gaap, ["Assets"])

    # Shareholders' equity: feeds ROE, P/B, equity-to-assets, float-to-equity.
    shareholders_equity = _latest_value(
        us_gaap,
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    )

    # Shares outstanding: feeds EPS, P/B, P/S, and market cap.
    shares_outstanding = _latest_value(us_gaap, ["CommonStockSharesOutstanding"], unit_keys=("shares",))

    # EPS: feeds the P/E ratio.
    eps = _latest_value(us_gaap, ["EarningsPerShareBasic"], unit_keys=("USD/shares",))

    # --- Insurance tags --------------------------------------------------

    # Premiums earned: top line for insurers; feeds loss/expense/combined ratios.
    premiums_earned, premiums_earned_py1, premiums_earned_py2 = _three_year_values(
        us_gaap, ["PremiumsEarnedNet", "PremiumsWrittenNet"]
    )

    # Claims incurred: feeds the loss ratio (claims paid as % of premiums).
    claims_incurred = _latest_value(
        us_gaap, ["PolicyholderBenefitsAndClaimsIncurredNet", "LiabilityForClaimsAndClaimsAdjustmentExpense"]
    )

    # Investment income: feeds the investment yield earned on float.
    investment_income = _latest_value(us_gaap, ["NetInvestmentIncome"])

    # Underwriting expenses: feeds the expense ratio.
    underwriting_expenses = _latest_value(us_gaap, ["InsuranceCommissionsAndFees", "OtherExpenses"])

    # Insurance reserves ("float"): feeds float-to-equity and price-to-float.
    insurance_reserves = _latest_value(us_gaap, ["LiabilityForFuturePolicyBenefits", "InsuranceLiability"])

    fiscal_year_end = _latest_end_date(us_gaap, ["NetIncomeLoss"]) or _latest_end_date(us_gaap, ["Assets"])

    result = {
        "ticker": ticker_upper,
        "cik": padded_cik,
        "fiscal_year_end": fiscal_year_end,
        "sub_type": _get_financial_services_sub_type(ticker),
        "net_interest_income": net_interest_income,
        "net_interest_income_prior_year_1": net_interest_income_py1,
        "net_interest_income_prior_year_2": net_interest_income_py2,
        "total_loans": total_loans,
        "total_deposits": total_deposits,
        "noninterest_expense": noninterest_expense,
        "noninterest_income": noninterest_income,
        "provision_for_loan_losses": provision_for_loan_losses,
        "nonperforming_loans_proxy": nonperforming_loans_proxy,
        "allowance_for_loan_losses": allowance_for_loan_losses,
        "tier1_capital": tier1_capital,
        "risk_weighted_assets": risk_weighted_assets,
        "net_income": net_income,
        "net_income_prior_year_1": net_income_py1,
        "net_income_prior_year_2": net_income_py2,
        "total_assets": total_assets,
        "shareholders_equity": shareholders_equity,
        "shares_outstanding": shares_outstanding,
        "eps": eps,
        "premiums_earned": premiums_earned,
        "premiums_earned_prior_year_1": premiums_earned_py1,
        "premiums_earned_prior_year_2": premiums_earned_py2,
        "claims_incurred": claims_incurred,
        "investment_income": investment_income,
        "underwriting_expenses": underwriting_expenses,
        "insurance_reserves": insurance_reserves,
    }

    # Same fallback treatment as fetch_financials: EDGAR's share-count tag is
    # unreliable for dual-class/foreign/certain financial-services filers.
    from data_utils import get_shares_outstanding

    shares_outstanding, shares_outstanding_source = get_shares_outstanding(ticker_upper, result)
    result["shares_outstanding"] = shares_outstanding
    result["shares_outstanding_source"] = shares_outstanding_source

    return result


if __name__ == "__main__":
    data = fetch_financials("AAPL")
    print(json.dumps(data, indent=2))

    print(
        "Revenue trend:",
        [data["revenue_prior_year_2"], data["revenue_prior_year_1"], data["revenue"]],
    )
    print(
        "Net income trend:",
        [data["net_income_prior_year_2"], data["net_income_prior_year_1"], data["net_income"]],
    )
    print(
        "Gross profit trend:",
        [data["gross_profit_prior_year_2"], data["gross_profit_prior_year_1"], data["gross_profit"]],
    )
    print(
        "Operating cash flow trend:",
        [
            data["operating_cash_flow_prior_year_2"],
            data["operating_cash_flow_prior_year_1"],
            data["operating_cash_flow"],
        ],
    )
    print("Depreciation & amortisation:", data["depreciation_amortisation"])
    print("Current assets (prior year):", data["current_assets_prior_year"])
    print("Current liabilities (prior year):", data["current_liabilities_prior_year"])
