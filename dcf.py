"""Discounted Cash Flow (DCF) intrinsic value model — non-financial companies only.

A DCF estimates what a company is "really worth" today by projecting the cash it
will generate in the future and converting those future dollars into today's
dollars (since a dollar next year is worth less than a dollar today). It is not
meaningful for banks/insurers because their "cash flow" and balance sheet work
completely differently (their debt is their raw material, not a financing choice),
so those companies are blocked below and should use a residual income model instead.
"""

import io
import json
import math

import pandas as pd
import requests
import yfinance as yf

DAMODARAN_ERP_URL = "https://pages.stern.nyu.edu/~adamodar/pc/implprem/ERPbymonth.xlsx"
WORLD_BANK_GDP_URL = (
    "https://api.worldbank.org/v2/country/US/indicator/NY.GDP.MKTP.KD.ZG?format=json&mrv=5"
)

# Fields the model absolutely needs. If any of these is missing we can't build a
# credible forecast, so we bail out early with a clear message instead of guessing.
REQUIRED_FIELDS = [
    "revenue",
    "revenue_prior_year_1",
    "revenue_prior_year_2",
    "operating_income",
    "capex",
    "shares_outstanding",
    "long_term_debt",
    "current_assets",
    "current_liabilities",
]


def _get_sector(ticker):
    """Best-effort sector lookup; returns None (never raises) if it can't be found."""
    try:
        return yf.Ticker(ticker).info.get("sector")
    except Exception:
        return None


def _safe_growth_rate(new_value, old_value):
    """(new - old) / old, but 0.0 instead of a crash if the old value is zero."""
    if not old_value:
        return 0.0
    return (new_value - old_value) / old_value


def _get_forward_growth_rate(ticker, revenue_current):
    """Analyst consensus forward revenue growth, derived from yfinance's forward
    revenue estimates (`revenue_estimate`, rows "0y" = next annual estimate and
    "+1y" = the year after that). Returns (forward_growth_rate,
    revenue_estimate_year1, revenue_estimate_year2) or (None, None, None) if
    analyst estimates aren't available — never raises.
    """
    try:
        revenue_estimate = yf.Ticker(ticker).revenue_estimate
        revenue_next_year = revenue_estimate.loc["0y", "avg"]
        revenue_year_after = revenue_estimate.loc["+1y", "avg"]
    except Exception:
        return None, None, None

    values = (revenue_next_year, revenue_year_after, revenue_current)
    if any(v is None for v in values) or any(isinstance(v, float) and math.isnan(v) for v in values):
        return None, None, None
    if not revenue_current or not revenue_next_year:
        return None, None, None

    forward_growth_year1 = (revenue_next_year - revenue_current) / revenue_current
    forward_growth_year2 = (revenue_year_after - revenue_next_year) / revenue_next_year
    forward_growth_rate = (forward_growth_year1 + forward_growth_year2) / 2

    return forward_growth_rate, float(revenue_next_year), float(revenue_year_after)


def _get_risk_free_rate():
    """Live 10-year Treasury yield (^TNX) as a decimal; falls back to 4% if the
    fetch fails, since a stale-but-reasonable rate beats crashing the whole DCF."""
    try:
        return yf.Ticker("^TNX").fast_info["lastPrice"] / 100
    except Exception:
        return 0.04


def _get_beta(ticker):
    """Best-effort beta lookup; returns None (never raises) if it can't be found."""
    try:
        return yf.Ticker(ticker).info.get("beta")
    except Exception:
        return None


def _get_equity_risk_premium():
    """Live market-implied equity risk premium from Damodaran's monthly ERP
    spreadsheet (most recent month's value from the "Last 12 months data" sheet).
    Falls back to 0.055 if the fetch, parse, or format ever fails — this reflects
    current market pricing, so it replaces a hardcoded sector ERP table entirely.
    """
    try:
        response = requests.get(DAMODARAN_ERP_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        sheet = pd.read_excel(io.BytesIO(response.content), sheet_name="Last 12 months data")
        erp_columns = [col for col in sheet.columns if "erp" in str(col).lower()]
        if not erp_columns:
            raise ValueError("no ERP column found in Damodaran spreadsheet")
        latest_erp = sheet[erp_columns[0]].dropna().iloc[-1]
        return round(float(latest_erp), 4), "damodaran_implied"
    except Exception:
        return 0.055, "fallback_hardcoded"


def _get_terminal_growth_rate():
    """Long-run nominal GDP growth from the World Bank's real US GDP growth
    forecast (last 5 years, averaged) plus 2% for inflation, capped to a sane
    2%-4% range. Falls back to 0.025 if the fetch/parse fails.
    """
    try:
        response = requests.get(WORLD_BANK_GDP_URL, timeout=15)
        response.raise_for_status()
        payload = response.json()
        entries = payload[1]
        real_growth_values = [entry["value"] for entry in entries if entry.get("value") is not None]
        if not real_growth_values:
            raise ValueError("World Bank API returned no GDP growth values")
        avg_real_growth = (sum(real_growth_values) / len(real_growth_values)) / 100
        nominal_growth = avg_real_growth + 0.02
        nominal_growth = max(0.02, min(nominal_growth, 0.04))
        return nominal_growth, "worldbank_api"
    except Exception:
        return 0.025, "fallback_hardcoded"


def _get_size_premium(market_cap):
    """Duff & Phelps-style size premium: smaller companies carry extra risk that
    a single market-wide beta/ERP doesn't fully capture."""
    if market_cap > 200_000_000_000:
        return 0.0, "Mega Cap"
    if market_cap > 10_000_000_000:
        return 0.005, "Large Cap"
    if market_cap > 2_000_000_000:
        return 0.015, "Mid Cap"
    return 0.030, "Small Cap"


def _get_market_assumptions(ticker, financials, market_price):
    """All market-derived DCF assumptions in one place: equity risk premium
    (Damodaran), size premium, cost of equity/debt, capital structure weights,
    WACC, and terminal growth rate (World Bank GDP forecast). Every input is
    fetched live rather than hardcoded, with a documented fallback if a source
    is unreachable.
    """
    shares_outstanding = financials.get("shares_outstanding")
    long_term_debt = financials.get("long_term_debt")
    interest_expense = financials.get("interest_expense")

    # --- Step 1: Equity risk premium (Damodaran implied ERP) ---
    equity_risk_premium, erp_source = _get_equity_risk_premium()

    # --- Step 2: Size premium based on market cap ---
    market_cap = market_price * shares_outstanding
    size_premium, size_category = _get_size_premium(market_cap)

    # --- Step 3: Cost of equity (CAPM + size premium) ---
    risk_free_rate = _get_risk_free_rate()
    beta = _get_beta(ticker)
    # Beta missing - flag it by falling back to the market-average beta of 1.0
    # (the `beta` field returned below stays None so callers can see this happened).
    effective_beta = beta if beta is not None else 1.0
    # Size premium accounts for the additional risk of smaller companies that
    # CAPM alone understates — standard in Duff & Phelps methodology.
    cost_of_equity = risk_free_rate + (effective_beta * equity_risk_premium) + size_premium

    # --- Step 4: Terminal growth rate (World Bank GDP forecast) ---
    terminal_growth_rate, gdp_source = _get_terminal_growth_rate()

    # --- Cost of debt: what the company actually pays on its debt, after tax ---
    # Interest is tax-deductible, so the true cost to the company is lower than
    # the stated interest rate — that's what the (1 - tax_rate) adjustment does.
    if not interest_expense or not long_term_debt:
        cost_of_debt_pretax = 0.05
    else:
        cost_of_debt_pretax = interest_expense / long_term_debt
    tax_rate = 0.21  # standard US federal corporate tax rate
    cost_of_debt = cost_of_debt_pretax * (1 - tax_rate)

    # --- Capital structure weights: how much of the company is funded by ---
    # --- shareholders' equity vs. by debt, at current market value. ---
    if not long_term_debt:
        weight_equity, weight_debt = 1.0, 0.0
    else:
        total_capital = market_cap + long_term_debt
        weight_equity = market_cap / total_capital
        weight_debt = long_term_debt / total_capital

    # Blend the two costs by how much of the company each one actually funds.
    wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt)
    # Keep the result within a sane real-world range regardless of noisy inputs.
    wacc = max(0.07, min(wacc, 0.15))

    return {
        "wacc": wacc,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt": cost_of_debt,
        "weight_equity": weight_equity,
        "weight_debt": weight_debt,
        "beta": beta,
        "risk_free_rate": risk_free_rate,
        "equity_risk_premium": equity_risk_premium,
        "erp_source": erp_source,
        "size_premium": size_premium,
        "size_category": size_category,
        "terminal_growth_rate": terminal_growth_rate,
        "gdp_source": gdp_source,
    }


def run_dcf(financials, market_price):
    """Estimate intrinsic value per share for a non-financial company using a
    5-year discounted cash flow model built entirely from `financials` (the dict
    returned by data_fetch.fetch_financials) and a live `market_price`.
    """

    # --- Guardrail: DCF math doesn't work for banks/insurers -------------------
    # A bank's deposits and loans ARE its business, not "debt" in the normal sense,
    # so free cash flow and enterprise value don't mean the same thing for them.
    ticker = financials.get("ticker")
    sector = _get_sector(ticker) if ticker else None
    if sector == "Financial Services":
        return {
            "error": "DCF not applicable for financial services companies — use residual income model instead"
        }

    # --- Bail out cleanly if we don't have the raw numbers we need -------------
    missing = [field for field in REQUIRED_FIELDS if financials.get(field) is None]
    if missing:
        return {"error": f"insufficient data for DCF — missing: {missing}"}

    revenue = financials["revenue"]
    revenue_py1 = financials["revenue_prior_year_1"]
    revenue_py2 = financials["revenue_prior_year_2"]
    operating_income = financials["operating_income"]
    capex = financials["capex"]
    shares_outstanding = financials["shares_outstanding"]
    long_term_debt = financials["long_term_debt"]
    current_assets = financials["current_assets"]
    current_liabilities = financials["current_liabilities"]

    if not shares_outstanding:
        return {"error": "insufficient data for DCF — missing: ['shares_outstanding']"}

    # ============================================================
    # STEP 1: NOPAT — Net Operating Profit After Tax.
    # NOPAT = operating profit after tax, stripping out the effect of debt —
    # represents cash generated purely from business operations.
    # ============================================================
    tax_rate = 0.21
    nopat = operating_income * (1 - tax_rate)

    # ============================================================
    # STEP 2: D&A — Depreciation & Amortisation.
    # D&A is non-cash — it reduces accounting profit but no cash actually left
    # the business, so we add it back.
    # Note: fetch_financials doesn't currently pull a D&A figure, so this will
    # fall back to 0 (flagged via da_missing) until that field is added upstream.
    # ============================================================
    da = financials.get("depreciation_amortisation")
    da_missing = da is None
    da = da if da is not None else 0

    # ============================================================
    # STEP 3: CAPEX — cash spent on physical assets to maintain and grow the
    # business (already extracted above as `capex`).
    # ============================================================

    # ============================================================
    # STEP 4: Change in Net Working Capital.
    # If working capital increases, the business is tying up more cash in
    # operations — this reduces free cash flow. If it decreases, cash is
    # being released.
    # Note: fetch_financials only gives the latest year's current
    # assets/liabilities, not the prior year's, so there's nothing to compare
    # against yet — delta_nwc falls back to 0 (flagged via nwc_prior_missing)
    # until prior-year balance sheet data is added upstream.
    # ============================================================
    current_assets_prior = financials.get("current_assets_prior_year")
    current_liabilities_prior = financials.get("current_liabilities_prior_year")
    nwc_prior_missing = current_assets_prior is None or current_liabilities_prior is None
    if nwc_prior_missing:
        delta_nwc = 0
    else:
        net_working_capital = current_assets - current_liabilities
        net_working_capital_prior = current_assets_prior - current_liabilities_prior
        delta_nwc = net_working_capital - net_working_capital_prior

    # ============================================================
    # STEP 5: FCFF — Free Cash Flow to the Firm.
    # FCFF is the cash available to all capital providers (both debt and
    # equity holders) after operating expenses and investment.
    # ============================================================
    fcff = nopat + da - capex - delta_nwc

    # ============================================================
    # STEP 6: FCF margin — what share of every revenue dollar becomes FCFF.
    # ============================================================
    fcf_margin = fcff / revenue if revenue else 0.0

    # ============================================================
    # Turn history (and, preferably, analyst forecasts) into forward-looking
    # growth assumptions.
    # ============================================================
    # Fallback: how fast has revenue actually been growing, on average, the
    # last 2 years? Purely backward-looking, so it's only used when analysts
    # covering the stock haven't published forward revenue estimates.
    growth_last_year = _safe_growth_rate(revenue, revenue_py1)
    growth_year_before = _safe_growth_rate(revenue_py1, revenue_py2)
    historical_growth_rate = (growth_last_year + growth_year_before) / 2

    # Preferred: analyst consensus forward revenue growth — reflects what
    # analysts who actually cover the company expect next, not just an
    # extrapolation of the past.
    forward_growth_rate, revenue_estimate_year1, revenue_estimate_year2 = _get_forward_growth_rate(
        ticker, revenue
    )

    if forward_growth_rate is not None:
        growth_source = "analyst_consensus"
        revenue_growth_rate = forward_growth_rate
    else:
        growth_source = "historical_average"
        revenue_growth_rate = historical_growth_rate

    # Cap it at 35% so one unusually strong estimate/year can't project an
    # unrealistic future, regardless of which source it came from.
    revenue_growth_rate = min(revenue_growth_rate, 0.35)

    # Market-derived assumptions, all fetched live rather than hardcoded:
    # WACC ("Weighted Average Cost of Capital", the discount rate used to convert
    # future cash into today's dollars) via CAPM + size premium, and "terminal
    # growth rate" (the pace the company can plausibly grow at FOREVER once it
    # matures) from the World Bank's long-run US GDP growth forecast.
    market_assumptions = _get_market_assumptions(ticker, financials, market_price)
    wacc = market_assumptions["wacc"]
    terminal_growth_rate = market_assumptions["terminal_growth_rate"]

    # ============================================================
    # STEP 3: Project revenue and FCFF for the next 5 years.
    # Years 1-2 use analyst consensus (or its historical fallback) directly,
    # since that reflects real forward expectations over the horizon analysts
    # actually forecast. Years 3-5 taper toward terminal growth — each year
    # the growth rate above the long-run terminal rate shrinks by 15% — as
    # uncertainty increases beyond the analyst horizon.
    # Each FCFF component is projected on its own terms rather than just
    # scaling a single blended margin: NOPAT scales with revenue (assuming a
    # stable operating margin), D&A and capex are held flat since they're
    # tied to existing physical assets rather than this year's sales, and
    # the change in working capital scales with the dollar change in revenue.
    # ============================================================
    nopat_margin = nopat / revenue if revenue else 0.0
    nwc_ratio = delta_nwc / revenue if revenue else 0.0

    projected_revenues = []
    projected_fcfs = []
    prior_year_revenue = revenue
    for year in range(1, 6):
        if year <= 2:
            year_growth_rate = revenue_growth_rate
        else:
            decay_factor = 0.85 ** (year - 2)
            year_growth_rate = terminal_growth_rate + (revenue_growth_rate - terminal_growth_rate) * decay_factor
        this_year_revenue = prior_year_revenue * (1 + year_growth_rate)

        projected_nopat = this_year_revenue * nopat_margin
        projected_da = da  # held constant — non-cash, relatively stable
        projected_capex = capex  # held constant as before
        revenue_change = this_year_revenue - prior_year_revenue
        projected_delta_nwc = nwc_ratio * revenue_change
        projected_fcff = projected_nopat + projected_da - projected_capex - projected_delta_nwc

        projected_revenues.append(this_year_revenue)
        projected_fcfs.append(projected_fcff)
        prior_year_revenue = this_year_revenue

    # ============================================================
    # STEP 4: Terminal value — the lump-sum value of every dollar of cash flow
    # the company will generate AFTER year 5, forever, assuming it settles into
    # steady terminal growth. This is usually the single biggest piece of a DCF.
    # ============================================================
    terminal_value = projected_fcfs[-1] * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)

    # ============================================================
    # STEP 5: Discount everything back to today's dollars and turn that into a
    # per-share price.
    # ============================================================
    # Money in the future is worth less than money today, so each future FCF
    # (and the terminal value) is shrunk by the discount rate for every year it's away.
    present_value_fcfs = [fcf / (1 + wacc) ** year for year, fcf in enumerate(projected_fcfs, start=1)]
    present_value_terminal = terminal_value / (1 + wacc) ** 5

    # Enterprise value = what the whole business (debt + equity) is worth today.
    enterprise_value = sum(present_value_fcfs) + present_value_terminal

    # Equity value = what's left for shareholders after paying off long-term debt,
    # adjusted for net working capital (short-term assets minus short-term liabilities).
    equity_value = enterprise_value - long_term_debt + (current_assets - current_liabilities)

    # Spread that equity value across every share to get a price per share.
    intrinsic_value_per_share = equity_value / shares_outstanding

    # ============================================================
    # STEP 6: Compare the model's estimate to what the market is actually charging.
    # ============================================================
    if intrinsic_value_per_share:
        margin_of_safety = (intrinsic_value_per_share - market_price) / intrinsic_value_per_share * 100
    else:
        margin_of_safety = None

    if market_price < intrinsic_value_per_share * 0.85:
        verdict = "UNDERVALUED"
    elif market_price > intrinsic_value_per_share * 1.15:
        verdict = "OVERVALUED"
    else:
        verdict = "FAIRLY VALUED"

    # ============================================================
    # STEP 7: Hand back everything, not just the final number, so the reasoning
    # behind the estimate stays visible.
    # ============================================================
    return {
        "intrinsic_value_per_share": round(intrinsic_value_per_share, 2),
        "market_price": round(market_price, 2),
        "verdict": verdict,
        "margin_of_safety": round(margin_of_safety, 2) if margin_of_safety is not None else None,
        "wacc": round(wacc, 4),
        "cost_of_equity": round(market_assumptions["cost_of_equity"], 4),
        "cost_of_debt": round(market_assumptions["cost_of_debt"], 4),
        "weight_equity": round(market_assumptions["weight_equity"], 4),
        "weight_debt": round(market_assumptions["weight_debt"], 4),
        "beta": market_assumptions["beta"],
        "risk_free_rate": round(market_assumptions["risk_free_rate"], 4),
        "equity_risk_premium": market_assumptions["equity_risk_premium"],
        "erp_source": market_assumptions["erp_source"],
        "size_premium": market_assumptions["size_premium"],
        "size_category": market_assumptions["size_category"],
        "terminal_growth_rate": round(terminal_growth_rate, 4),
        "gdp_source": market_assumptions["gdp_source"],
        "revenue_growth_rate": round(revenue_growth_rate, 4),
        "growth_source": growth_source,
        "forward_growth_rate": round(forward_growth_rate, 4) if forward_growth_rate is not None else None,
        "historical_growth_rate": round(historical_growth_rate, 4),
        "revenue_estimate_year1": round(revenue_estimate_year1, 2) if revenue_estimate_year1 is not None else None,
        "revenue_estimate_year2": round(revenue_estimate_year2, 2) if revenue_estimate_year2 is not None else None,
        "nopat": round(nopat, 2),
        "da": round(da, 2),
        "da_missing": da_missing,
        "delta_nwc": round(delta_nwc, 2),
        "nwc_prior_missing": nwc_prior_missing,
        "fcff": round(fcff, 2),
        "fcf_margin": round(fcf_margin, 4),
        "projected_revenues": [round(v, 2) for v in projected_revenues],
        "projected_fcfs": [round(v, 2) for v in projected_fcfs],
        "terminal_value": round(terminal_value, 2),
        "enterprise_value": round(enterprise_value, 2),
        "equity_value": round(equity_value, 2),
        "shares_outstanding": shares_outstanding,
    }


if __name__ == "__main__":
    from data_fetch import fetch_financials

    # --- AAPL: a normal, non-financial company - the DCF should run fully. -----
    aapl_financials = fetch_financials("AAPL")
    aapl_price = yf.Ticker("AAPL").fast_info["lastPrice"]
    aapl_result = run_dcf(aapl_financials, aapl_price)

    print(json.dumps(aapl_result, indent=2))

    if "error" not in aapl_result:
        print()
        print("AAPL market assumptions (live vs. fallback)")
        print("-" * 40)
        print(f"Equity risk premium       : {aapl_result['equity_risk_premium']:.4f}  ({aapl_result['erp_source']})")
        print(f"Size premium              : {aapl_result['size_premium']:.4f}  ({aapl_result['size_category']})")
        print(f"Terminal growth rate      : {aapl_result['terminal_growth_rate']:.4f}  ({aapl_result['gdp_source']})")
        print()
        print("AAPL revenue growth assumption")
        print("-" * 40)
        print(f"Growth source used        : {aapl_result['growth_source']}")
        print(f"Forward growth rate       : {aapl_result['forward_growth_rate']}")
        print(f"Historical growth rate    : {aapl_result['historical_growth_rate']}")
        print(f"Revenue growth rate used  : {aapl_result['revenue_growth_rate']}")
        print(f"Revenue estimate year 1   : {aapl_result['revenue_estimate_year1']}")
        print(f"Revenue estimate year 2   : {aapl_result['revenue_estimate_year2']}")
        print()
        print("AAPL FCFF breakdown")
        print("-" * 40)
        print(f"NOPAT                     : ${aapl_result['nopat']:,.0f}")
        print(f"D&A (missing: {aapl_result['da_missing']})       : ${aapl_result['da']:,.0f}")
        print(f"CAPEX                     : ${aapl_financials['capex']:,.0f}")
        print(f"Delta NWC (missing: {aapl_result['nwc_prior_missing']}) : ${aapl_result['delta_nwc']:,.0f}")
        print(f"FCFF                      : ${aapl_result['fcff']:,.0f}")
        print(f"FCF margin                : {aapl_result['fcf_margin']:.4f}")
        print()
        print("AAPL DCF summary")
        print("-" * 40)
        print(f"Intrinsic value per share : ${aapl_result['intrinsic_value_per_share']:.2f}")
        print(f"Current market price      : ${aapl_result['market_price']:.2f}")
        print(f"Verdict                   : {aapl_result['verdict']}")
        print(f"Margin of safety          : {aapl_result['margin_of_safety']:.2f}%")
        print()
        print("WACC breakdown (CAPM)")
        print("-" * 40)
        print(f"Risk-free rate (^TNX)     : {aapl_result['risk_free_rate']:.4f}")
        print(f"Beta                      : {aapl_result['beta']}")
        print(f"Equity risk premium       : {aapl_result['equity_risk_premium']:.4f}")
        print(f"Size premium              : {aapl_result['size_premium']:.4f}")
        print(f"Cost of equity            : {aapl_result['cost_of_equity']:.4f}")
        print(f"Cost of debt (after-tax)  : {aapl_result['cost_of_debt']:.4f}")
        print(f"Weight equity / debt      : {aapl_result['weight_equity']:.4f} / {aapl_result['weight_debt']:.4f}")
        print(f"WACC                      : {aapl_result['wacc']:.4f}")
        print()
        print(f"{'Year':<6}{'Projected Revenue':>20}{'Projected FCF':>18}")
        for year, (rev, fcf) in enumerate(
            zip(aapl_result["projected_revenues"], aapl_result["projected_fcfs"]), start=1
        ):
            print(f"{year:<6}{rev:>20,.0f}{fcf:>18,.0f}")

    # --- JPM: a bank - should be blocked before any DCF math runs. -------------
    print()
    print("JPM (Financial Services) check")
    print("-" * 40)
    jpm_financials = fetch_financials("JPM")
    jpm_price = yf.Ticker("JPM").fast_info["lastPrice"]
    jpm_result = run_dcf(jpm_financials, jpm_price)
    print(json.dumps(jpm_result, indent=2))
