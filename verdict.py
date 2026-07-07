"""Reconciled verdict: intrinsic value vs. peer comps, in one function call.

Ties together every other module in this project — the right valuation model
(DCF for normal companies, Residual Income for banks/insurers), comparable
company analysis, ratio analysis, and management guidance extraction — into a
single end-to-end read on whether a stock looks cheap, expensive, or fairly
priced, and whether the two independent signals (what the business is worth
vs. what the market pays for similar businesses) agree or conflict.
"""

import json
import sys

import yfinance as yf

# Some reconciled-verdict strings contain an em dash; reconfigure stdout to
# UTF-8 so it renders correctly on Windows consoles using a legacy codepage,
# without altering the actual string content.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from calculator import calculate_ratios
from calculator_financial_services import calculate_financial_ratios
from comps import run_comps
from dcf import run_dcf
from evaluator import evaluate_analysis
from management_extract import run_management_extraction
from residual_income import run_residual_income

# Only these four comps multiples are price-based valuation signals (the rest
# of comps.py's ratios are profitability/growth, which don't have a
# PREMIUM/DISCOUNT/IN LINE label to count here).
VALUATION_KEYS = ["pe_ratio", "pb_ratio", "ev_to_ebitda", "ps_ratio"]


def _get_sector_and_industry(ticker):
    """Best-effort sector/industry lookup; never raises."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    return info.get("sector"), info.get("industry")


def _reconcile_verdict(valuation_available, valuation_verdict, comps_available, premium_count, discount_count, inline_count):
    """Combine the intrinsic-value verdict with the peer-comps signal into one
    plain-English read. Two independent methods agreeing is a much stronger
    signal than either alone, so agreement is called out as STRONG, and
    disagreement is flagged explicitly as MIXED rather than silently
    defaulting to one side.
    """
    if valuation_available and comps_available:
        # Whichever label (PREMIUM/DISCOUNT/IN LINE) has the most of the 4
        # valuation multiples wins; ties favor DISCOUNT, then PREMIUM, then
        # IN LINE (an arbitrary but deterministic tie-break).
        majority = max(
            [("DISCOUNT", discount_count), ("PREMIUM", premium_count), ("IN LINE", inline_count)],
            key=lambda pair: pair[1],
        )[0]

        if valuation_verdict == "UNDERVALUED":
            if majority == "DISCOUNT":
                return "STRONG BUY SIGNAL — intrinsic value and peer multiples both suggest undervaluation"
            if majority == "PREMIUM":
                return "MIXED SIGNAL — trades at discount to intrinsic value but premium to peers"
            return "MODERATE BUY SIGNAL — intrinsic value suggests undervaluation, peer multiples are neutral"

        if valuation_verdict == "OVERVALUED":
            if majority == "PREMIUM":
                return "STRONG SELL SIGNAL — intrinsic value and peer multiples both suggest overvaluation"
            if majority == "DISCOUNT":
                return "MIXED SIGNAL — trades above intrinsic value but at discount to peers"
            return "MODERATE SELL SIGNAL — intrinsic value suggests overvaluation, peer multiples are neutral"

        # FAIRLY VALUED
        return "FAIRLY VALUED — trading near intrinsic value"

    if valuation_available and not comps_available:
        return f"{valuation_verdict} (comps unavailable)"

    if comps_available and not valuation_available:
        if discount_count > premium_count:
            return "DISCOUNT TO PEERS — no intrinsic value model available"
        if premium_count > discount_count:
            return "PREMIUM TO PEERS — no intrinsic value model available"
        return "IN LINE WITH PEERS — no intrinsic value model available"

    return "INSUFFICIENT DATA — analysis could not be completed"


def run_verdict(ticker, financials, market_price):
    """Run the full analysis pipeline for `ticker`: the valuation model that
    actually applies (DCF for non-financials, Residual Income for financial
    services), peer comps, ratio analysis, and management guidance
    extraction — then reconcile the valuation and comps signals into one
    final verdict. `financials` must already be fetched with the matching
    shape (data_fetch.fetch_financials for non-financials,
    fetch_financials_financial_services for financial services).
    """
    # ============================================================
    # STEP 1: What kind of company is this? Banks/insurers need a completely
    # different valuation approach (Residual Income) than everyone else (DCF).
    # ============================================================
    sector, industry = _get_sector_and_industry(ticker)
    company_type = "financial_services" if sector == "Financial Services" else "non_financial"

    # ============================================================
    # STEP 2: Run the valuation model that actually applies to this company.
    # ============================================================
    if company_type == "non_financial":
        valuation_model = "DCF"
        valuation_result = run_dcf(financials, market_price)
    else:
        valuation_model = "Residual Income"
        valuation_result = run_residual_income(financials, market_price)

    valuation_available = "error" not in valuation_result

    # ============================================================
    # STEP 3: Peer comps — how the market prices similar companies, entirely
    # independent of what the intrinsic-value model says.
    # ============================================================
    comps_result = run_comps(ticker)
    comps_available = "error" not in comps_result and bool(comps_result.get("peers"))

    # ============================================================
    # STEP 4: Pull the headline numbers out of each result.
    # ============================================================
    if valuation_available:
        intrinsic_value_per_share = valuation_result.get("intrinsic_value_per_share")
        valuation_verdict = valuation_result.get("verdict")
        margin_of_safety = valuation_result.get("margin_of_safety")
    else:
        intrinsic_value_per_share = None
        valuation_verdict = None
        margin_of_safety = None

    if comps_available:
        vs_median = comps_result.get("vs_median", {})
        labels = [vs_median.get(key) for key in VALUATION_KEYS]
        premium_count = labels.count("PREMIUM")
        discount_count = labels.count("DISCOUNT")
        inline_count = labels.count("IN LINE")
        peers = [peer["ticker"] for peer in comps_result.get("peers", [])]
        peer_medians = comps_result.get("peer_medians")
        target_ratios = comps_result.get("target_ratios")
    else:
        premium_count = discount_count = inline_count = 0
        peers = []
        peer_medians = None
        target_ratios = None

    # ============================================================
    # STEP 5: Reconcile the two independent signals into one final verdict.
    # ============================================================
    reconciled_verdict = _reconcile_verdict(
        valuation_available, valuation_verdict, comps_available, premium_count, discount_count, inline_count
    )

    # ============================================================
    # STEP 6: Ratio analysis, using whichever calculator matches the company type.
    # ============================================================
    if company_type == "non_financial":
        ratio_result = calculate_ratios(financials, market_price)
    else:
        ratio_result = calculate_financial_ratios(financials, market_price)

    # ============================================================
    # STEP 7: Management guidance extraction (10-K MD&A/risk factors via
    # Claude). This makes real network + LLM calls, so any failure here
    # degrades gracefully rather than taking down the whole verdict.
    # ============================================================
    try:
        management_result = run_management_extraction(ticker)
        management_available = "error" not in management_result
    except Exception as exc:
        management_result = {"error": str(exc)}
        management_available = False

    if management_available:
        management_summary = management_result.get("management_summary")
        management_recommended_overrides = management_result.get("recommended_overrides")
    else:
        management_summary = None
        management_recommended_overrides = None

    # ============================================================
    # STEP 8: Everything, in one place.
    # ============================================================
    result = {
        "ticker": ticker.upper(),
        "company_type": company_type,
        "sector": sector,
        "industry": industry,
        "market_price": market_price,
        "valuation_model": valuation_model,
        "valuation_available": valuation_available,
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "margin_of_safety": margin_of_safety,
        "valuation_verdict": valuation_verdict,
        "comps_available": comps_available,
        "premium_count": premium_count,
        "discount_count": discount_count,
        "inline_count": inline_count,
        "peers": peers,
        "peer_medians": peer_medians,
        "target_ratios": target_ratios,
        "reconciled_verdict": reconciled_verdict,
        "ratio_result": ratio_result,
        "management_available": management_available,
        "management_summary": management_summary,
        "management_recommended_overrides": management_recommended_overrides,
        "valuation_result": valuation_result,
        "comps_result": comps_result,
        # Not in the original spec's field list, but kept alongside
        # valuation_result/comps_result (which the spec already returns in
        # full) so callers - including the __main__ summary card below - can
        # show management's actual growth drivers/risks, not just the summary.
        "management_result": management_result,
    }

    # ============================================================
    # STEP 9: Score the whole analysis on the 5-dimension quality rubric
    # before it reaches a user. This runs last, on the complete result, so
    # the evaluator can sanity-check the valuation output and comps quality
    # together with the raw financials - never let a bug in the evaluator
    # itself take down an otherwise-good analysis.
    # ============================================================
    try:
        result["evaluation"] = evaluate_analysis(ticker, financials, result)
    except Exception:
        result["evaluation"] = {"error": "evaluation failed", "overall_score": None, "quality_rating": "UNKNOWN"}

    return result


def _print_summary_card(result):
    """A one-page, plain-English readout of the full verdict."""
    print("=" * 64)
    print(f"{result['ticker']}  |  {result['sector']} / {result['industry']}  |  {result['company_type']}")
    print("=" * 64)

    if result["valuation_available"]:
        print(f"Valuation model      : {result['valuation_model']}")
        print(f"Intrinsic value      : ${result['intrinsic_value_per_share']:.2f}")
        print(f"Market price         : ${result['market_price']:.2f}")
        print(f"Margin of safety     : {result['margin_of_safety']:.2f}%")
        print(f"Valuation verdict    : {result['valuation_verdict']}")
    else:
        print(f"Valuation model      : {result['valuation_model']} (unavailable)")
        print(f"Market price         : ${result['market_price']:.2f}")

    print()
    if result["comps_available"]:
        print(f"Peers                : {', '.join(result['peers'])}")
        print(
            f"Peer signal          : {result['premium_count']} PREMIUM / "
            f"{result['discount_count']} DISCOUNT / {result['inline_count']} IN LINE"
        )
    else:
        print("Peers                : comps unavailable")

    print()
    print("RECONCILED VERDICT")
    print(f">>> {result['reconciled_verdict']} <<<")

    if result["management_available"]:
        drivers = (
            result["management_result"].get("extracted_guidance", {}).get("key_growth_drivers") or []
        )
        if drivers:
            print()
            print("Top management guidance points:")
            for driver in drivers[:3]:
                print(f"  - {driver}")

    print()


if __name__ == "__main__":
    from data_fetch import fetch_financials, fetch_financials_financial_services

    # 1. AAPL - non-financial, should route to DCF.
    # 2. JPM - financial services, should route to Residual Income.
    test_cases = [
        ("AAPL", fetch_financials),
        ("JPM", fetch_financials_financial_services),
    ]

    for ticker, fetch_fn in test_cases:
        financials = fetch_fn(ticker)
        market_price = yf.Ticker(ticker).fast_info["lastPrice"]
        result = run_verdict(ticker, financials, market_price)
        _print_summary_card(result)
