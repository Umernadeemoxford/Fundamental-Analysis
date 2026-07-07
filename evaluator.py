"""Rubric-based quality evaluator for the equity analysis pipeline.

Every analysis this project produces is a chain of assumptions: SEC tags that
might be missing or mistagged, a valuation model built on projected growth
rates, a peer set picked from a static sector map, an LLM summarizing a 10-K.
Before any of that reaches a user, this module scores it on 5 dimensions
(1-5 each) — four of them pure Python sanity checks, one an LLM-as-judge call
— so a bad or thin analysis gets flagged instead of presented with the same
confidence as a solid one. This mirrors how professional evaluation rubrics
work: quantitative checks catch mechanical errors, a judge model catches
qualitative issues a script can't (genericness, hallucination risk).
"""

import json
import os
import re
from datetime import datetime

import anthropic
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = "claude-sonnet-4-6"
MINIMUM_ACCEPTABLE_SCORE = 3.5

# ============================================================
# DIMENSION 1 — Data Completeness
# Measures: how much of the raw financial-statement data we needed was
# actually available from SEC EDGAR. A ratio computed from mostly-missing
# inputs (or silently defaulted to 0) is not trustworthy even if the
# arithmetic is correct.
#
# The "critical fields" a company must report depend on what kind of company
# it is: a bank's 10-K has no "revenue" or "gross_profit" line (its whole
# income statement is structured around net interest income instead), so
# checking it against the non-financial field list would always report it as
# mostly-missing data regardless of how complete its actual filing is.
# ============================================================
CRITICAL_FIELDS_NON_FINANCIAL = [
    "revenue",
    "net_income",
    "gross_profit",
    "operating_income",
    "total_assets",
    "total_liabilities",
    "shareholders_equity",
    "current_assets",
    "current_liabilities",
    "long_term_debt",
    "operating_cash_flow",
    "capex",
    "shares_outstanding",
    "eps",
    "depreciation_amortisation",
    "interest_expense",
]

CRITICAL_FIELDS_FINANCIAL_SERVICES = [
    "net_income",
    "shareholders_equity",
    "shares_outstanding",
    "eps",
    "total_assets",
    "net_interest_income",
    "total_deposits",
    "total_loans",
    "noninterest_expense",
    "noninterest_income",
    "provision_for_loan_losses",
    "tier1_capital",
]


def _evaluate_data_completeness(financials, company_type):
    critical_fields = (
        CRITICAL_FIELDS_FINANCIAL_SERVICES if company_type == "financial_services" else CRITICAL_FIELDS_NON_FINANCIAL
    )
    missing_fields = [field for field in critical_fields if financials.get(field) is None]
    fields_checked = len(critical_fields)
    fields_populated = fields_checked - len(missing_fields)
    populated_pct = fields_populated / fields_checked

    if populated_pct >= 1.0:
        score = 5
    elif populated_pct >= 0.90:
        score = 4
    elif populated_pct >= 0.75:
        score = 3
    elif populated_pct >= 0.60:
        score = 2
    else:
        score = 1

    return {
        "score": score,
        "missing_fields": missing_fields,
        "fields_checked": fields_checked,
        "fields_populated": fields_populated,
    }


# ============================================================
# DIMENSION 2 — Data Consistency
# Measures: do the numbers we pulled actually hang together internally?
# These are the same sanity checks a human analyst runs before trusting a
# spreadsheet — EPS times share count should roughly equal net income, a
# balance sheet should balance, gross profit can't exceed revenue. A failure
# here usually means a tag mismatch or a stale/inconsistent filing, not a
# real business anomaly.
# A check only counts toward the score if its required inputs are actually
# available — a bank not having "gross_profit" isn't a consistency problem,
# it's a different business model, so that check is simply skipped rather
# than counted as a failure.
# ============================================================


def _evaluate_data_consistency(financials, verdict_result, company_type):
    if company_type == "financial_services":
        return _evaluate_data_consistency_financial_services(financials, verdict_result)
    return _evaluate_data_consistency_non_financial(financials, verdict_result)


def _evaluate_data_consistency_financial_services(financials, verdict_result):
    """Bank/insurer-specific consistency checks — a bank's balance sheet and
    income statement don't have gross_profit/revenue/current_assets at all,
    so the non-financial checks above would be meaningless here."""
    checks = []  # (name, applicable, passed, fail_reason)

    eps = financials.get("eps")
    shares_outstanding = financials.get("shares_outstanding")
    net_income = financials.get("net_income")
    total_assets = financials.get("total_assets")
    total_loans = financials.get("total_loans")
    tier1_capital = financials.get("tier1_capital")
    net_interest_income = financials.get("net_interest_income")
    noninterest_income = financials.get("noninterest_income")

    # Check 1: EPS x shares outstanding should approximate net income (a
    # looser 20% band than the non-financial check, since bank EPS figures
    # sometimes reflect preferred-dividend adjustments common income doesn't).
    if eps is not None and shares_outstanding is not None and net_income:
        passed = abs((eps * shares_outstanding) - net_income) / abs(net_income) < 0.20
        checks.append((
            "EPS reconciliation", True, passed,
            "EPS x shares outstanding does not reconcile with net income within 20%.",
        ))
    else:
        checks.append(("EPS reconciliation", False, None, None))

    # Check 2: total assets must exceed the loan book - loans are only one
    # part of a bank's balance sheet, so this should always hold.
    if total_assets is not None and total_loans is not None:
        passed = total_assets > total_loans
        checks.append((
            "Total assets vs. total loans", True, passed,
            "Total assets are not greater than total loans, which is inconsistent for a bank balance sheet.",
        ))
    else:
        checks.append(("Total assets vs. total loans", False, None, None))

    # Check 3: Tier 1 capital (a slice of equity) must be smaller than total assets.
    if tier1_capital is not None and total_assets is not None:
        passed = tier1_capital < total_assets
        checks.append((
            "Tier 1 capital vs. total assets", True, passed,
            "Tier 1 capital is not smaller than total assets, which shouldn't be possible.",
        ))
    else:
        checks.append(("Tier 1 capital vs. total assets", False, None, None))

    # Check 4: net income should be less than total revenue (net interest
    # income + noninterest income) - profit can't exceed the top line.
    if net_income is not None and net_interest_income is not None and noninterest_income is not None:
        passed = net_income < (net_interest_income + noninterest_income)
        checks.append((
            "Net income vs. total revenue", True, passed,
            "Net income exceeds net interest income + noninterest income, which is impossible.",
        ))
    else:
        checks.append(("Net income vs. total revenue", False, None, None))

    # Check 5: the valuation model's intrinsic value output should be a sane number.
    intrinsic_value_per_share = verdict_result.get("intrinsic_value_per_share")
    market_price = verdict_result.get("market_price")
    if intrinsic_value_per_share is not None and market_price:
        passed = 0 < intrinsic_value_per_share < (market_price * 20)
        checks.append((
            "Intrinsic value sanity", True, passed,
            "Intrinsic value per share is non-positive or more than 20x the market price - "
            "the valuation model's output looks implausible.",
        ))
    else:
        checks.append(("Intrinsic value sanity", False, None, None))

    applicable = [c for c in checks if c[1]]
    passed_checks = [c[0] for c in applicable if c[2]]
    failed_checks = [{"check": c[0], "reason": c[3]} for c in applicable if not c[2]]

    score = round((len(passed_checks) / len(applicable)) * 5, 1) if applicable else 1.0

    return {"score": score, "passed_checks": passed_checks, "failed_checks": failed_checks}


def _evaluate_data_consistency_non_financial(financials, verdict_result):
    checks = []  # (name, applicable, passed, fail_reason)

    eps = financials.get("eps")
    shares_outstanding = financials.get("shares_outstanding")
    net_income = financials.get("net_income")
    gross_profit = financials.get("gross_profit")
    revenue = financials.get("revenue")
    total_liabilities = financials.get("total_liabilities")
    shareholders_equity = financials.get("shareholders_equity")
    total_assets = financials.get("total_assets")
    operating_income = financials.get("operating_income")
    revenue_py1 = financials.get("revenue_prior_year_1")
    revenue_py2 = financials.get("revenue_prior_year_2")

    # Check 1: EPS x shares outstanding should approximate net income.
    if eps is not None and shares_outstanding is not None and net_income:
        passed = abs((eps * shares_outstanding) - net_income) / abs(net_income) < 0.15
        checks.append((
            "EPS reconciliation", True, passed,
            "EPS x shares outstanding does not reconcile with net income within 15% - "
            "eps or shares_outstanding may be from a different period or share class.",
        ))
    else:
        checks.append(("EPS reconciliation", False, None, None))

    # Check 2: gross profit cannot exceed revenue.
    if gross_profit is not None and revenue is not None:
        passed = gross_profit <= revenue
        checks.append((
            "Gross profit reconciliation", True, passed,
            "Gross profit exceeds revenue, which is impossible - one of these figures is likely mistagged.",
        ))
    else:
        checks.append(("Gross profit reconciliation", False, None, None))

    # Check 3: the balance sheet should balance (assets = liabilities + equity).
    if total_liabilities is not None and shareholders_equity is not None and total_assets:
        passed = abs((total_liabilities + shareholders_equity) - total_assets) / abs(total_assets) < 0.10
        checks.append((
            "Balance sheet equation", True, passed,
            "Total liabilities + shareholders' equity does not reconcile with total assets within 10%.",
        ))
    else:
        checks.append(("Balance sheet equation", False, None, None))

    # Check 4: operating income shouldn't exceed gross profit (opex is subtracted from it).
    if operating_income is not None and gross_profit is not None:
        passed = operating_income <= gross_profit
        checks.append((
            "Operating income vs. gross profit", True, passed,
            "Operating income exceeds gross profit, which shouldn't happen since operating "
            "expenses are subtracted from gross profit to get there.",
        ))
    else:
        checks.append(("Operating income vs. gross profit", False, None, None))

    # Check 5: no implausible single-year revenue swing across the 3-year window.
    if revenue is not None and revenue_py1 and revenue_py2:
        growth_latest = (revenue - revenue_py1) / abs(revenue_py1)
        growth_prior = (revenue_py1 - revenue_py2) / abs(revenue_py2)
        anomaly = any(g > 2.0 or g < -0.8 for g in (growth_latest, growth_prior))
        checks.append((
            "Revenue trend anomaly", True, not anomaly,
            f"Revenue moved {growth_latest * 100:.0f}% and {growth_prior * 100:.0f}% year-over-year "
            "across the 3-year window - a swing this large (>200% up or >80% down) usually signals "
            "a data or tagging issue rather than genuine business performance.",
        ))
    else:
        checks.append(("Revenue trend anomaly", False, None, None))

    # Check 6: the valuation model's intrinsic value output should be a sane number.
    intrinsic_value_per_share = verdict_result.get("intrinsic_value_per_share")
    market_price = verdict_result.get("market_price")
    if intrinsic_value_per_share is not None and market_price:
        passed = 0 < intrinsic_value_per_share < (market_price * 20)
        checks.append((
            "Intrinsic value sanity", True, passed,
            "Intrinsic value per share is non-positive or more than 20x the market price - "
            "the valuation model's output looks implausible.",
        ))
    else:
        checks.append(("Intrinsic value sanity", False, None, None))

    # Check 7: free cash flow shouldn't exceed total revenue in magnitude.
    fcff = (verdict_result.get("valuation_result") or {}).get("fcff")
    if fcff is not None and revenue:
        passed = abs(fcff) < revenue
        checks.append((
            "FCF sanity", True, passed,
            "Free cash flow to the firm exceeds total revenue in magnitude, which is not realistic.",
        ))
    else:
        checks.append(("FCF sanity", False, None, None))

    applicable = [c for c in checks if c[1]]
    passed_checks = [c[0] for c in applicable if c[2]]
    failed_checks = [{"check": c[0], "reason": c[3]} for c in applicable if not c[2]]

    score = round((len(passed_checks) / len(applicable)) * 5, 1) if applicable else 1.0

    return {"score": score, "passed_checks": passed_checks, "failed_checks": failed_checks}


# ============================================================
# DIMENSION 3 — Valuation Model Reliability
# Measures: are the valuation model's own assumptions and outputs within
# ranges a real analyst would accept, or has some input (a bad growth
# estimate, a WACC that collapsed toward zero) pushed the model into
# territory where its output shouldn't be trusted at face value?
# ============================================================


def _evaluate_valuation_reliability(verdict_result, company_type):
    if company_type == "financial_services":
        return _evaluate_valuation_reliability_financial_services(verdict_result)
    return _evaluate_valuation_reliability_non_financial(verdict_result)


def _evaluate_valuation_reliability_financial_services(verdict_result):
    """Residual-income-specific reliability checks — this model has no
    revenue_growth_rate/wacc/fcf_margin/terminal_value/enterprise_value at
    all (those are DCF-specific), so the non-financial checks below would
    always come back "not applicable" and unfairly tank the score."""
    valuation_result = verdict_result.get("valuation_result") or {}
    passed_checks = []
    failed_checks = []

    # Check 1: ROE (the engine compounding book value forward) should be plausible.
    roe = valuation_result.get("roe")
    if roe is not None:
        if 0.05 < roe < 0.40:
            passed_checks.append("ROE reasonableness")
        else:
            failed_checks.append({
                "check": "ROE reasonableness",
                "reason": f"ROE of {roe * 100:.1f}% is outside the plausible 5%-40% range.",
            })

    # Check 2: cost of equity (the discount rate here) should sit in a realistic band.
    cost_of_equity = valuation_result.get("cost_of_equity")
    if cost_of_equity is not None:
        if 0.06 < cost_of_equity < 0.18:
            passed_checks.append("Cost of equity reasonableness")
        else:
            failed_checks.append({
                "check": "Cost of equity reasonableness",
                "reason": f"Cost of equity of {cost_of_equity * 100:.2f}% is outside the plausible 6%-18% range.",
            })

    # Check 3: payout ratio should be a valid share of earnings (0-90%).
    historical_payout_ratio = valuation_result.get("historical_payout_ratio")
    if historical_payout_ratio is not None:
        if 0.0 <= historical_payout_ratio <= 0.90:
            passed_checks.append("Payout ratio range")
        else:
            failed_checks.append({
                "check": "Payout ratio range",
                "reason": f"Historical payout ratio of {historical_payout_ratio * 100:.1f}% is outside the plausible 0%-90% range.",
            })

    # Check 4: intrinsic value per share should be a positive number.
    intrinsic_value_per_share = verdict_result.get("intrinsic_value_per_share")
    if intrinsic_value_per_share is not None:
        if intrinsic_value_per_share > 0:
            passed_checks.append("Intrinsic value positivity")
        else:
            failed_checks.append({
                "check": "Intrinsic value positivity",
                "reason": f"Intrinsic value per share of {intrinsic_value_per_share} is not positive.",
            })

    # Check 5: book value per share should be a positive number.
    book_value_per_share = valuation_result.get("book_value_per_share")
    if book_value_per_share is not None:
        if book_value_per_share > 0:
            passed_checks.append("Book value positivity")
        else:
            failed_checks.append({
                "check": "Book value positivity",
                "reason": f"Book value per share of {book_value_per_share} is not positive.",
            })

    checks_passed = len(passed_checks)
    if checks_passed >= 5:
        score = 5
    elif checks_passed == 4:
        score = 4
    elif checks_passed == 3:
        score = 3
    elif checks_passed == 2:
        score = 2
    else:
        score = 1

    return {"score": score, "passed_checks": passed_checks, "failed_checks": failed_checks}


def _evaluate_valuation_reliability_non_financial(verdict_result):
    valuation_result = verdict_result.get("valuation_result") or {}
    passed_checks = []
    failed_checks = []

    # Check 1: the assumed growth rate driving the forecast should be plausible.
    growth_rate = valuation_result.get("revenue_growth_rate")
    if growth_rate is not None:
        if -0.30 <= growth_rate <= 0.50:
            passed_checks.append("Growth rate reasonableness")
        else:
            failed_checks.append({
                "check": "Growth rate reasonableness",
                "reason": f"Assumed growth rate of {growth_rate * 100:.1f}% is outside the plausible -30% to +50% range.",
            })

    # Check 2: WACC (the discount rate) should sit in a realistic band.
    wacc = valuation_result.get("wacc")
    if wacc is not None:
        if 0.06 <= wacc <= 0.18:
            passed_checks.append("WACC reasonableness")
        else:
            failed_checks.append({
                "check": "WACC reasonableness",
                "reason": f"WACC of {wacc * 100:.2f}% is outside the plausible 6%-18% range.",
            })

    # Check 3: margin of safety shouldn't be an extreme outlier.
    margin_of_safety = verdict_result.get("margin_of_safety")
    if margin_of_safety is not None:
        if -200 <= margin_of_safety <= 200:
            passed_checks.append("Margin of safety range")
        else:
            failed_checks.append({
                "check": "Margin of safety range",
                "reason": f"Margin of safety of {margin_of_safety:.1f}% is an extreme outlier (beyond +/-200%).",
            })

    # Check 4: FCF margin should be a plausible share of revenue.
    fcf_margin = valuation_result.get("fcf_margin")
    if fcf_margin is not None:
        if -0.50 <= fcf_margin <= 0.60:
            passed_checks.append("FCF margin reasonableness")
        else:
            failed_checks.append({
                "check": "FCF margin reasonableness",
                "reason": f"FCF margin of {fcf_margin * 100:.1f}% is outside the plausible -50% to +60% range.",
            })

    # Check 5: terminal value shouldn't dominate the whole valuation - if it
    # does, the model is really just a bet on the terminal-year assumptions.
    terminal_value = valuation_result.get("terminal_value")
    enterprise_value = valuation_result.get("enterprise_value")
    if terminal_value is not None and enterprise_value:
        proportion = terminal_value / enterprise_value
        if abs(proportion) < 0.85:
            passed_checks.append("Terminal value proportion")
        else:
            failed_checks.append({
                "check": "Terminal value proportion",
                "reason": f"Terminal value is {proportion * 100:.0f}% of enterprise value - the model "
                          "is overly sensitive to terminal-year assumptions rather than the explicit forecast.",
            })

    checks_passed = len(passed_checks)
    if checks_passed >= 5:
        score = 5
    elif checks_passed == 4:
        score = 4
    elif checks_passed == 3:
        score = 3
    elif checks_passed == 2:
        score = 2
    else:
        score = 1

    return {"score": score, "passed_checks": passed_checks, "failed_checks": failed_checks}


# ============================================================
# DIMENSION 4 — Comps Quality
# Measures: is the peer benchmark actually meaningful, or is it built on too
# few peers, thin data, mismatched sectors, or missing multiples? A comps
# table with one peer and half its cells empty shouldn't carry the same
# weight in a verdict as a clean 5-peer set with full data.
# ============================================================


def _get_ticker_sector(ticker):
    """Best-effort live sector lookup; never raises."""
    try:
        return yf.Ticker(ticker).info.get("sector")
    except Exception:
        return None


def _evaluate_comps_quality(comps_result):
    passed_checks = []
    failed_checks = []

    peers = comps_result.get("peers") or []
    target_sector = comps_result.get("sector")

    # Check 1: at least 3 peers found - fewer makes the benchmark statistically weak.
    if len(peers) >= 3:
        passed_checks.append("Peer count")
    else:
        failed_checks.append({
            "check": "Peer count",
            "reason": f"Only {len(peers)} peer(s) found - fewer than 3 peers makes the comps benchmark weak.",
        })

    # Check 2: at least 70% of all peer ratio values should be populated.
    all_values = [value for peer in peers for value in (peer.get("ratios") or {}).values()]
    populated_pct = (sum(1 for v in all_values if v is not None) / len(all_values)) if all_values else 0.0
    if populated_pct >= 0.70:
        passed_checks.append("Peer data completeness")
    else:
        failed_checks.append({
            "check": "Peer data completeness",
            "reason": f"Only {populated_pct * 100:.0f}% of peer ratio values are populated - comps may be based on thin data.",
        })

    # Check 3: every peer should actually be in the target's sector (re-verified
    # live rather than trusted purely on how comps.py picked them, since a
    # hardcoded sector map can drift out of date).
    peer_sectors = [_get_ticker_sector(peer["ticker"]) for peer in peers]
    if peers and all(sector == target_sector for sector in peer_sectors):
        passed_checks.append("Sector match")
    else:
        mismatched = [peer["ticker"] for peer, sector in zip(peers, peer_sectors) if sector != target_sector]
        failed_checks.append({
            "check": "Sector match",
            "reason": (
                f"Peer(s) {mismatched} are not currently classified in the target's sector ({target_sector})."
                if mismatched
                else "No peers available to verify sector match."
            ),
        })

    # Check 4: at least 3 of the 4 core valuation multiples need both a
    # target value and a peer median to actually benchmark anything.
    target_ratios = comps_result.get("target_ratios") or {}
    peer_medians = comps_result.get("peer_medians") or {}
    core_multiples = ["pe_ratio", "pb_ratio", "ev_to_ebitda", "ps_ratio"]
    complete_multiples = sum(
        1 for key in core_multiples if target_ratios.get(key) is not None and peer_medians.get(key) is not None
    )
    if complete_multiples >= 3:
        passed_checks.append("Core multiple coverage")
    else:
        failed_checks.append({
            "check": "Core multiple coverage",
            "reason": f"Only {complete_multiples} of 4 core valuation multiples have both target and peer data.",
        })

    score = round((len(passed_checks) / 4) * 5, 1)
    return {"score": score, "passed_checks": passed_checks, "failed_checks": failed_checks}


# ============================================================
# DIMENSION 5 — AI Output Quality (LLM-as-judge)
# Measures: is the Claude-generated management summary actually grounded in
# real, specific facts about this company, or is it generic boilerplate that
# could apply to any filer? A script can't judge prose quality or spot subtle
# hallucination risk, so this dimension hands that off to a second Claude
# call acting purely as an evaluator, not a generator.
# ============================================================

AI_JUDGE_SYSTEM_PROMPT = (
    "You are a quality evaluator for financial analysis outputs. Score the following "
    "management summary on a scale of 1-5 using this rubric. Return ONLY a JSON object "
    "with no other text."
)


def _get_claude_client():
    # Locally, python-dotenv loads ANTHROPIC_API_KEY from .env into the
    # environment. On Streamlit Cloud there's no .env file (it's gitignored
    # and never uploaded) - secrets are configured via the app's Secrets
    # manager and exposed through st.secrets instead, so fall back to that.
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            import streamlit as st

            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — add it to .env locally or to Streamlit Cloud's Secrets")
    return anthropic.Anthropic(api_key=api_key)


def _strip_code_fences(text):
    """Claude occasionally wraps JSON in ```json ... ``` despite being told
    not to - strip that off before parsing rather than trusting prompt compliance."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _evaluate_ai_output_quality(ticker, verdict_result):
    management_available = verdict_result.get("management_available")
    management_summary = verdict_result.get("management_summary")

    if not management_available or not management_summary:
        return {
            "score": 2,
            "groundedness": None,
            "completeness": None,
            "accuracy_risk": None,
            "justification": "Management extraction was unavailable for this ticker",
        }

    management_result = verdict_result.get("management_result") or {}
    extracted_guidance = management_result.get("extracted_guidance") or {}
    sections_found = management_result.get("sections_found") or []
    confidence_score = extracted_guidance.get("confidence_score")
    key_growth_drivers = extracted_guidance.get("key_growth_drivers") or []
    key_risks = extracted_guidance.get("key_risks") or []

    user_prompt = f"""Evaluate this management summary extracted from a 10-K filing:

TICKER: {ticker}
MANAGEMENT SUMMARY: {management_summary}
SECTIONS FOUND: {sections_found}
CONFIDENCE SCORE: {confidence_score}
KEY GROWTH DRIVERS: {key_growth_drivers}
KEY RISKS: {key_risks}

Score on these criteria and return as JSON with this exact shape:
{{
  "score": <number 1-5, overall quality>,
  "groundedness": <number 1-5 - is the summary grounded in specific company details or generic boilerplate? 1=entirely generic, 5=highly specific with real numbers and named initiatives>,
  "completeness": <number 1-5 - does it cover growth drivers, risks, and management tone? 1=missing most, 5=covers all>,
  "accuracy_risk": <number 1-5 - are there any statements that appear hallucinated or inconsistent with the ticker? 1=high hallucination risk, 5=all claims plausible>,
  "justification": "<one sentence explaining the score>"
}}

Return ONLY the JSON object, no markdown, no code fences."""

    try:
        client = _get_claude_client()
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=AI_JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = next((block.text for block in response.content if block.type == "text"), "")
        result = json.loads(_strip_code_fences(raw_text))
        return {
            "score": result.get("score", 2),
            "groundedness": result.get("groundedness"),
            "completeness": result.get("completeness"),
            "accuracy_risk": result.get("accuracy_risk"),
            "justification": result.get("justification"),
        }
    except (anthropic.APIError, json.JSONDecodeError, StopIteration, RuntimeError) as exc:
        return {
            "score": 2,
            "groundedness": None,
            "completeness": None,
            "accuracy_risk": None,
            "justification": f"AI judge call failed: {exc}",
        }


def _build_recommendations(data_completeness, data_consistency, valuation_reliability, comps_quality, ai_output_quality):
    """Turn every failed check across all 5 dimensions into one flat list of
    plain-English things a user should watch out for before trusting the result."""
    recommendations = []

    if data_completeness["missing_fields"]:
        recommendations.append(
            f"Missing {len(data_completeness['missing_fields'])} critical financial field(s) "
            f"({', '.join(data_completeness['missing_fields'])}) - ratios relying on these are less reliable."
        )

    for failure in data_consistency["failed_checks"]:
        recommendations.append(failure["reason"])

    for failure in valuation_reliability["failed_checks"]:
        recommendations.append(failure["reason"])

    for failure in comps_quality["failed_checks"]:
        recommendations.append(failure["reason"])

    if ai_output_quality["score"] < 3:
        recommendations.append(
            f"Management guidance summary quality is low ({ai_output_quality['score']}/5): "
            f"{ai_output_quality.get('justification') or 'no justification provided'}."
        )

    if not recommendations:
        recommendations.append("No significant issues detected across any evaluation dimension.")

    return recommendations


def _detect_company_type(ticker):
    """Financial services companies (banks/insurers) need different critical
    fields and consistency/reliability checks than everyone else, since their
    financials dict (fetch_financials_financial_services) and valuation model
    (residual_income.py) use an entirely different schema than a normal
    company's (fetch_financials / dcf.py)."""
    try:
        sector = yf.Ticker(ticker).info.get("sector")
    except Exception:
        sector = None
    return "financial_services" if sector == "Financial Services" else "non_financial"


def evaluate_analysis(ticker, financials, verdict_result):
    """Score a complete verdict.run_verdict() output across all 5 quality
    dimensions and return a full evaluation report. Never raises - a failed
    AI-judge call degrades to a low score rather than crashing the report.
    """
    company_type = _detect_company_type(ticker)

    data_completeness = _evaluate_data_completeness(financials, company_type)
    data_consistency = _evaluate_data_consistency(financials, verdict_result, company_type)
    valuation_reliability = _evaluate_valuation_reliability(verdict_result, company_type)
    comps_quality = _evaluate_comps_quality(verdict_result.get("comps_result") or {})
    ai_output_quality = _evaluate_ai_output_quality(ticker, verdict_result)

    dimension_scores = {
        "data_completeness": data_completeness["score"],
        "data_consistency": data_consistency["score"],
        "valuation_reliability": valuation_reliability["score"],
        "comps_quality": comps_quality["score"],
        "ai_output_quality": ai_output_quality["score"],
    }

    overall_score = round(sum(dimension_scores.values()) / len(dimension_scores), 2)

    if overall_score >= 4.5:
        quality_rating = "EXCELLENT — High confidence in analysis quality"
    elif overall_score >= MINIMUM_ACCEPTABLE_SCORE:
        quality_rating = "GOOD — Analysis is reliable with minor caveats"
    elif overall_score >= 2.5:
        quality_rating = "MODERATE — Use with caution, review flagged issues"
    else:
        quality_rating = "POOR — Significant data quality issues detected"

    return {
        "ticker": ticker.upper(),
        "company_type": company_type,
        "overall_score": overall_score,
        "quality_rating": quality_rating,
        "show_warning": overall_score < MINIMUM_ACCEPTABLE_SCORE,
        "dimension_scores": dimension_scores,
        "dimension_details": {
            "data_completeness": data_completeness,
            "data_consistency": data_consistency,
            "valuation_reliability": valuation_reliability,
            "comps_quality": comps_quality,
            "ai_output_quality": ai_output_quality,
        },
        "recommendations": _build_recommendations(
            data_completeness, data_consistency, valuation_reliability, comps_quality, ai_output_quality
        ),
        "evaluation_timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    from data_fetch import fetch_financials, fetch_financials_financial_services
    from verdict import run_verdict

    def _run_and_evaluate(ticker, fetch_fn):
        financials = fetch_fn(ticker)
        market_price = yf.Ticker(ticker).fast_info["lastPrice"]
        verdict_result = run_verdict(ticker, financials, market_price)
        return evaluate_analysis(ticker, financials, verdict_result)

    print("=" * 60)
    print("AAPL Evaluation Report")
    print("=" * 60)
    aapl_eval = _run_and_evaluate("AAPL", fetch_financials)
    print(json.dumps(aapl_eval, indent=2))

    print()
    print("=" * 60)
    print("JPM Evaluation Report")
    print("=" * 60)
    jpm_eval = _run_and_evaluate("JPM", fetch_financials_financial_services)
    print(json.dumps(jpm_eval, indent=2))

    print()
    print("=" * 60)
    print("Summary Table")
    print("=" * 60)
    dimensions = ["data_completeness", "data_consistency", "valuation_reliability", "comps_quality", "ai_output_quality"]
    print(f"{'Dimension':<28}{'AAPL':>10}{'JPM':>10}")
    print("-" * 48)
    for dimension in dimensions:
        print(f"{dimension:<28}{aapl_eval['dimension_scores'][dimension]:>10}{jpm_eval['dimension_scores'][dimension]:>10}")
    print("-" * 48)
    print(f"{'OVERALL':<28}{aapl_eval['overall_score']:>10}{jpm_eval['overall_score']:>10}")
    print(f"{'Quality Rating':<28}{aapl_eval['quality_rating']}")
    print(f"{'':<28}{jpm_eval['quality_rating']}")
