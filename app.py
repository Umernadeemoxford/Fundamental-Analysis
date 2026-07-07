"""Equity Analysis Agent — Streamlit front end for the fundamental analysis
pipeline (SEC EDGAR + Yahoo Finance + Claude). This file only handles
presentation: every number on screen comes from data_fetch/dcf/comps/
calculator/management_extract/residual_income via verdict.run_verdict.
"""

import concurrent.futures
import time

import pandas as pd
import streamlit as st
import yfinance as yf

from data_fetch import fetch_financials, fetch_financials_financial_services
from verdict import run_verdict

st.set_page_config(layout="wide", page_title="Equity Analysis Agent", page_icon="📈")

PROGRESS_STEPS = [
    "Fetching financials from SEC EDGAR...",
    "Computing ratios...",
    "Running comparable company analysis...",
    "Running valuation model...",
    "Extracting management guidance...",
    "Generating verdict...",
]

# Rough, general rules of thumb for "good" vs "concerning" ratio values —
# not universal truths, just a sensible default so the ratio tables aren't
# flat black-and-white. (metric_key -> (direction, good_threshold, bad_threshold))
RATIO_RULES = {
    "gross_margin": ("higher", 40, 20),
    "operating_margin": ("higher", 20, 5),
    "net_margin": ("higher", 15, 3),
    "roe": ("higher", 15, 5),
    "roa": ("higher", 8, 2),
    "current_ratio": ("higher", 1.5, 1.0),
    "debt_to_equity": ("lower", 0.5, 2.0),
    "interest_coverage": ("higher", 5, 1.5),
    "fcf_yield": ("higher", 4, 0),
    "revenue_growth_1y": ("higher", 8, 0),
    "revenue_growth_2y": ("higher", 8, 0),
    "net_income_growth_1y": ("higher", 8, 0),
    "net_income_growth_2y": ("higher", 8, 0),
    "nim": ("higher", 3, 1.5),
    "efficiency_ratio": ("lower", 55, 70),
    "cost_to_income": ("lower", 150, 220),
    "npl_ratio": ("lower", 1.5, 4),
    "coverage_ratio": ("higher", 100, 50),
    "provision_to_loans": ("lower", 0.3, 1.0),
    "loan_loss_rate": ("lower", 0.3, 1.0),
    "tier1_capital_ratio": ("higher", 10, 6),
    "equity_to_assets": ("higher", 8, 4),
    "loss_ratio": ("lower", 60, 80),
    "expense_ratio": ("lower", 30, 40),
    "combined_ratio": ("lower", 100, 105),
    "investment_yield": ("higher", 3, 1),
    "float_to_equity": ("higher", 50, 10),
    "underwriting_profit_margin": ("higher", 0, -5),
    "dividend_yield": ("higher", 2, 0),
    "nii_growth_1y": ("higher", 5, 0),
    "premium_growth_1y": ("higher", 5, 0),
    "premium_growth_2y": ("higher", 5, 0),
}

GOOD_COLOR = "color: #4CAF50; font-weight: 600"
BAD_COLOR = "color: #F44336; font-weight: 600"


def _get_sector(ticker):
    """Best-effort sector lookup; never raises."""
    try:
        return yf.Ticker(ticker).info.get("sector")
    except Exception:
        return None


def _fetch_financials_for_ticker(ticker):
    """Fetch the right shape of financials for this ticker: the financial-
    services extraction for banks/insurers, the generic one for everyone else.
    """
    sector = _get_sector(ticker)
    if sector == "Financial Services":
        return fetch_financials_financial_services(ticker), sector
    return fetch_financials(ticker), sector


def _run_analysis(ticker):
    """Fetch financials, get a live price, and run the full verdict pipeline.
    Raises on failure — the caller is responsible for catching and showing a
    clean message instead of a traceback.
    """
    financials, _sector = _fetch_financials_for_ticker(ticker)
    market_price = yf.Ticker(ticker).fast_info["lastPrice"]
    return run_verdict(ticker, financials, market_price)


def _fmt_money(value, decimals=2):
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    return f"${value:,.{decimals}f}"


def _fmt_pct(value, decimals=2):
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    return f"{value:,.{decimals}f}%"


def _fmt_num(value, decimals=2):
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    return f"{value:,.{decimals}f}"


def _verdict_color(reconciled_verdict):
    """Pick a banner color from the reconciled verdict's leading keyword."""
    text = reconciled_verdict.upper()
    if "STRONG BUY" in text or "MODERATE BUY" in text:
        return "#1B4332", "#4CAF50"
    if "STRONG SELL" in text or "MODERATE SELL" in text:
        return "#4A1414", "#F44336"
    if "UNDERVALUED" in text:
        return "#1B4332", "#4CAF50"
    if "OVERVALUED" in text:
        return "#4A1414", "#F44336"
    if "MIXED" in text:
        return "#4A3B14", "#FFC107"
    if "FAIRLY VALUED" in text or "IN LINE" in text:
        return "#1B2A4A", "#64B5F6"
    return "#2B2B2B", "#BDBDBD"


QUALITY_RATING_COLORS = {
    "EXCELLENT": "#4CAF50",
    "GOOD": "#14B8A6",
    "MODERATE": "#FFC107",
    "POOR": "#F44336",
}

# Plain-English note on which ratios/checks each critical financial field
# feeds, so a missing field means something concrete to a non-technical user.
FIELD_EXPLANATIONS = {
    "revenue": "gross/operating/net margins, growth rates, P/S ratio",
    "net_income": "ROE, ROA, net margin, EPS-based ratios, growth rates",
    "gross_profit": "gross margin",
    "operating_income": "operating margin",
    "total_assets": "ROA, asset turnover, equity-to-assets",
    "total_liabilities": "solvency ratios",
    "shareholders_equity": "ROE, P/B ratio, debt-to-equity",
    "current_assets": "the current ratio and working capital",
    "current_liabilities": "the current ratio and working capital",
    "long_term_debt": "debt-to-equity and leverage ratios",
    "operating_cash_flow": "free cash flow and FCF yield",
    "capex": "free cash flow (FCF = OCF - capex)",
    "shares_outstanding": "EPS, market cap, and per-share valuation ratios",
    "eps": "the P/E ratio",
    "depreciation_amortisation": "the FCFF calculation (D&A add-back)",
    "interest_expense": "the interest coverage ratio and cost of debt",
    "net_interest_income": "net interest margin and total revenue for a bank",
    "total_deposits": "the loan-to-deposit liquidity ratio",
    "total_loans": "NPL ratio, loan-to-deposit ratio, provision-to-loans",
    "noninterest_expense": "the efficiency ratio",
    "noninterest_income": "the efficiency ratio and price-to-sales proxy",
    "provision_for_loan_losses": "provision-to-loans and loan-loss-rate",
    "tier1_capital": "the Tier 1 capital adequacy ratio",
}

DIMENSION_LABELS = {
    "data_completeness": "Data Completeness",
    "data_consistency": "Data Consistency",
    "valuation_reliability": "Valuation Reliability",
    "comps_quality": "Comps Quality",
    "ai_output_quality": "AI Output Quality",
}


def _quality_rating_label(quality_rating):
    """'GOOD — Analysis is reliable...' -> 'GOOD'."""
    return (quality_rating or "").split(" — ")[0].strip()


def _score_circles(score):
    """5-circle visual for a 1-5 score, filled = score rounded to nearest integer."""
    if score is None:
        return "○○○○○"
    filled = max(0, min(5, round(score)))
    return "●" * filled + "○" * (5 - filled)


def _ratio_color(key, value):
    """Green/red for a metric based on RATIO_RULES, or "" if there's no rule
    defined or the value isn't a usable number."""
    rule = RATIO_RULES.get(key)
    if rule is None or value is None or not isinstance(value, (int, float)):
        return ""
    direction, good, bad = rule
    if direction == "higher":
        return GOOD_COLOR if value >= good else BAD_COLOR if value <= bad else ""
    return GOOD_COLOR if value <= good else BAD_COLOR if value >= bad else ""


def _render_ratio_category(title, ratio_result, group_keys):
    """Build and render one ratio category as a 2-column (Metric, Display)
    table, colored via a separately-computed style matrix rather than extra
    DataFrame columns — st.dataframe doesn't honor Styler.hide(axis="columns"),
    so any helper column added to the frame would stay visible.
    """
    metrics, displays, styles = [], [], []
    for key, label, is_pct in group_keys:
        if key not in ratio_result:
            continue
        value = ratio_result.get(key)
        metrics.append(label)
        displays.append(_fmt_pct(value) if is_pct else _fmt_num(value))
        styles.append(_ratio_color(key, value))

    if not metrics:
        return

    st.subheader(title)
    df = pd.DataFrame({"Metric": metrics, "Display": displays})
    style_matrix = pd.DataFrame({"Metric": [""] * len(df), "Display": styles})
    styled = df.style.apply(lambda _: style_matrix, axis=None).hide(axis="index")
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ============================================================
# Persona selection (landing screen)
# ============================================================
PERSONAS = [
    {
        "key": "value_investor",
        "icon": "🏛️",
        "name": "Value Investor",
        "description": "Find undervalued stocks trading below intrinsic value. Buffett-style fundamental analysis with DCF valuation and margin of safety.",
        "active": True,
    },
    {
        "key": "growth_investor",
        "icon": "🚀",
        "name": "Growth Investor",
        "description": "Identify high-growth companies with strong revenue momentum and expanding markets.",
        "active": False,
    },
    {
        "key": "dividend_investor",
        "icon": "💰",
        "name": "Dividend Investor",
        "description": "Build a passive income portfolio with stable dividend-paying companies.",
        "active": False,
    },
    {
        "key": "passive_investor",
        "icon": "📊",
        "name": "Passive Investor",
        "description": "Simple buy or avoid signals without the complexity. Plain English verdicts.",
        "active": False,
    },
    {
        "key": "quant_investor",
        "icon": "⚡",
        "name": "Quantitative Investor",
        "description": "Systematic mispricing signals across multiple stocks. Data-driven screening.",
        "active": False,
    },
    {
        "key": "fund_manager",
        "icon": "🏦",
        "name": "Fund Manager",
        "description": "Institutional-grade analysis with full ratio sets, peer benchmarking, and exportable reports.",
        "active": False,
    },
]

PERSONA_ACTIVE_BORDER = "#14B8A6"  # teal


def _render_persona_card(col, persona):
    """One persona card: a full-HTML block for the visual (icon, name,
    description, border/opacity/badge), plus a real button underneath for
    the click — Streamlit can't embed an interactive widget inside raw HTML,
    so the button is what actually fires the selection."""
    active = persona["active"]
    border_color = PERSONA_ACTIVE_BORDER if active else "#444"
    opacity = "1" if active else "0.5"
    shadow = f"box-shadow: 0 4px 24px rgba(20,184,166,0.35);" if active else ""
    badge = (
        ""
        if active
        else (
            "<span style='position:absolute; top:10px; right:10px; background:#555; "
            "color:#ddd; font-size:0.7em; font-weight:600; padding:3px 10px; "
            "border-radius:12px;'>COMING SOON</span>"
        )
    )

    with col:
        # Built as one line, not an indented multi-line f-string: Streamlit's
        # markdown parser treats 4+ leading spaces as a code block, which
        # would render this HTML as literal text instead of parsing it.
        card_html = (
            f'<div style="position:relative; border:2px solid {border_color}; border-radius:14px; '
            f'padding:22px 18px; opacity:{opacity}; {shadow} min-height:230px; margin-bottom:10px;">'
            f"{badge}"
            f'<div style="font-size:2.6em; text-align:center;">{persona["icon"]}</div>'
            f'<div style="font-size:1.15em; font-weight:700; text-align:center; margin-top:10px;">{persona["name"]}</div>'
            f'<div style="font-size:0.85em; color:#aaa; text-align:center; margin-top:10px; line-height:1.4;">{persona["description"]}</div>'
            f"</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)
        if active:
            return st.button(f"Select {persona['name']}", key=f"select_{persona['key']}", use_container_width=True, type="primary")
        st.button("Coming Soon", key=f"select_{persona['key']}", use_container_width=True, disabled=True)
        return False


if "persona" not in st.session_state:
    st.title("Equity Analysis Agent")
    st.caption("Personalised stock analysis tailored to your investment style")
    st.divider()
    st.subheader("What type of investor are you?")
    st.write("")

    row1 = st.columns(3)
    row2 = st.columns(3)
    persona_columns = list(row1) + list(row2)

    selected_persona = None
    for col, persona in zip(persona_columns, PERSONAS):
        if _render_persona_card(col, persona):
            selected_persona = persona

    if selected_persona is not None:
        st.session_state.persona = selected_persona["key"]
        st.session_state.persona_name = selected_persona["name"]
        st.session_state.persona_icon = selected_persona["icon"]
        st.success(f"✅ {selected_persona['name']} selected — Buffett-style analysis activated")
        time.sleep(1)
        st.rerun()

    # Nothing below this point renders until a persona is picked.
    st.stop()


# ============================================================
# Section 1 — Header
# ============================================================
title_col, persona_col = st.columns([4, 1])
with title_col:
    st.title("Equity Analysis Agent")
    st.caption("Fundamental analysis powered by SEC EDGAR, Yahoo Finance, and Claude AI")
with persona_col:
    badge_html = (
        '<div style="text-align:right; margin-top: 18px;">'
        f'<span style="background:{PERSONA_ACTIVE_BORDER}22; border:1px solid {PERSONA_ACTIVE_BORDER}; '
        f'color:{PERSONA_ACTIVE_BORDER}; padding:6px 14px; border-radius:20px; font-weight:600; font-size:0.85em;">'
        f"{st.session_state.persona_icon} {st.session_state.persona_name} Mode"
        "</span>"
        "</div>"
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    if st.button("Change", key="change_persona", use_container_width=True):
        for key in ("persona", "persona_name", "persona_icon", "result", "ticker", "elapsed"):
            st.session_state.pop(key, None)
        st.rerun()

st.markdown(
    "<span style='font-size: 0.8em; color: #888;'>"
    "For educational and demonstration purposes only. Not financial advice."
    "</span>",
    unsafe_allow_html=True,
)
st.divider()

# ============================================================
# Section 2 — Input
# ============================================================
input_col, button_col = st.columns([4, 1])
with input_col:
    ticker_input = st.text_input(
        "Enter a stock ticker (e.g. AAPL, JPM, MSFT)", value="", label_visibility="visible"
    )
with button_col:
    st.write("")
    st.write("")
    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

if run_clicked:
    ticker_clean = ticker_input.strip().upper()
    if not ticker_clean:
        st.warning("Enter a ticker symbol first.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        started_at = time.monotonic()
        error_message = None
        result = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_analysis, ticker_clean)
            step_index = 0
            while not future.done():
                # These labels are a best-effort narration of what's likely
                # happening, not a real progress signal from inside
                # run_verdict (which runs as a single blocking call) - once
                # every label has been shown, hold on the last one rather
                # than looping back to "Fetching financials..." and looking
                # like the analysis restarted.
                current_step = min(step_index, len(PROGRESS_STEPS) - 1)
                status_text.text(PROGRESS_STEPS[current_step])
                progress_bar.progress(min(0.9, 0.1 + 0.8 * (current_step + 1) / len(PROGRESS_STEPS)))
                step_index += 1
                time.sleep(3)
            try:
                result = future.result()
            except Exception as exc:
                error_message = str(exc)

        progress_bar.progress(1.0)
        status_text.empty()
        elapsed = time.monotonic() - started_at

        if error_message:
            st.error(f"Analysis failed for '{ticker_clean}': {error_message}")
        else:
            st.session_state["result"] = result
            st.session_state["ticker"] = ticker_clean
            st.session_state["elapsed"] = elapsed
            st.rerun()

# ============================================================
# Section 3 — Results (only after a successful run)
# ============================================================
if "result" in st.session_state:
    result = st.session_state["result"]
    elapsed = st.session_state.get("elapsed")

    top_left, top_right = st.columns([5, 1])
    with top_left:
        if elapsed is not None:
            st.caption(f"Analysis completed in {elapsed:.1f}s")
    with top_right:
        if st.button("Run Another Analysis", use_container_width=True):
            for key in ("result", "ticker", "elapsed"):
                st.session_state.pop(key, None)
            st.rerun()

    summary_tab_label = (
        "🏛️ Value Investor Screen" if st.session_state.persona == "value_investor" else "Summary"
    )
    tab_summary, tab_ratios, tab_comps, tab_valuation, tab_management, tab_quality = st.tabs(
        [summary_tab_label, "Ratios", "Comparable Companies", "Valuation Model", "Management Guidance", "Data Quality"]
    )

    # --------------------------------------------------------
    # TAB 1 — Summary
    # --------------------------------------------------------
    with tab_summary:
        bg_color, text_color = _verdict_color(result["reconciled_verdict"])
        verdict_html = (
            f'<div style="background-color:{bg_color}; padding: 24px; border-radius: 10px; margin-bottom: 20px;">'
            '<div style="font-size: 0.9em; color: #ccc; margin-bottom: 6px;">RECONCILED VERDICT</div>'
            f'<div style="font-size: 1.6em; font-weight: 700; color: {text_color};">{result["reconciled_verdict"]}</div>'
            "</div>"
        )
        st.markdown(verdict_html, unsafe_allow_html=True)

        evaluation = result.get("evaluation") or {}
        overall_score = evaluation.get("overall_score")
        if overall_score is not None:
            st.caption(f"Analysis Quality: {overall_score:.1f}/5.0 — {_quality_rating_label(evaluation.get('quality_rating'))}")

        if st.session_state.persona == "value_investor":
            # Buffett-style screen: an undervalued price alone isn't enough —
            # a value investor also wants a real safety cushion (25%+ margin
            # of safety) and a balance sheet that isn't overleveraged
            # (debt/equity under 0.5), since debt can wipe out a "cheap" stock.
            debt_to_equity = (result.get("ratio_result") or {}).get("debt_to_equity")
            margin_of_safety = result.get("margin_of_safety")
            if margin_of_safety is not None and debt_to_equity is not None and margin_of_safety >= 25 and debt_to_equity < 0.5:
                buy_line = "✅ BUY CANDIDATE — meets your investment criteria"
            else:
                buy_line = "❌ NOT A BUY — does not meet your criteria"
            note = ""
            if margin_of_safety is None or debt_to_equity is None:
                note = " (some inputs unavailable — treated conservatively as not meeting criteria)"
            st.markdown(
                f"**Value Investor View:** {buy_line} based on your 25% margin of safety requirement "
                f"and debt/equity threshold{note}"
            )

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric(
            "Intrinsic Value",
            _fmt_money(result.get("intrinsic_value_per_share")) if result["valuation_available"] else "N/A",
        )
        metric_col2.metric("Market Price", _fmt_money(result.get("market_price")))
        metric_col3.metric(
            "Margin of Safety",
            _fmt_pct(result.get("margin_of_safety")) if result["valuation_available"] else "N/A",
        )

        st.divider()

        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        info_col1.markdown(f"**Ticker**\n\n{result['ticker']}")
        info_col2.markdown(f"**Sector**\n\n{result.get('sector') or 'N/A'}")
        info_col3.markdown(f"**Industry**\n\n{result.get('industry') or 'N/A'}")
        info_col4.markdown(f"**Valuation Model**\n\n{result['valuation_model']}")

        st.divider()

        if result["management_available"] and result.get("management_summary"):
            st.subheader("Management Summary")
            st.write(result["management_summary"])

            extracted = result.get("management_result", {}).get("extracted_guidance", {}) or {}
            drivers = extracted.get("key_growth_drivers") or []
            risks = extracted.get("key_risks") or []

            driver_col, risk_col = st.columns(2)
            with driver_col:
                st.markdown("**Top Growth Drivers**")
                if drivers:
                    for driver in drivers:
                        st.markdown(f"- {driver}")
                else:
                    st.caption("None extracted.")
            with risk_col:
                st.markdown("**Top Risks**")
                if risks:
                    for risk in risks:
                        st.markdown(f"- {risk}")
                else:
                    st.caption("None extracted.")
        else:
            st.info("Management guidance extraction was unavailable for this ticker.")

    # --------------------------------------------------------
    # TAB 2 — Ratios
    # --------------------------------------------------------
    with tab_ratios:
        ratio_result = result.get("ratio_result") or {}

        if result["company_type"] == "non_financial":
            _render_ratio_category(
                "Profitability",
                ratio_result,
                [
                    ("gross_margin", "Gross Margin", True),
                    ("operating_margin", "Operating Margin", True),
                    ("net_margin", "Net Margin", True),
                    ("roe", "Return on Equity", True),
                    ("roa", "Return on Assets", True),
                ],
            )
            _render_ratio_category(
                "Liquidity & Solvency",
                ratio_result,
                [
                    ("current_ratio", "Current Ratio", False),
                    ("debt_to_equity", "Debt to Equity", False),
                    ("interest_coverage", "Interest Coverage", False),
                ],
            )
            _render_ratio_category(
                "Valuation",
                ratio_result,
                [
                    ("pe_ratio", "P/E Ratio", False),
                    ("pb_ratio", "P/B Ratio", False),
                    ("ps_ratio", "P/S Ratio", False),
                    ("ev_to_ebitda", "EV / EBITDA", False),
                    ("market_cap", "Market Cap ($)", False),
                    ("ev", "Enterprise Value ($)", False),
                ],
            )
            _render_ratio_category(
                "Cash Flow",
                ratio_result,
                [
                    ("free_cash_flow", "Free Cash Flow ($)", False),
                    ("fcf_yield", "FCF Yield", True),
                ],
            )
            _render_ratio_category(
                "Growth",
                ratio_result,
                [
                    ("revenue_growth_1y", "Revenue Growth (1Y)", True),
                    ("revenue_growth_2y", "Revenue Growth (2Y)", True),
                    ("net_income_growth_1y", "Net Income Growth (1Y)", True),
                ],
            )
        else:
            st.caption(f"Sub-type: {ratio_result.get('sub_type', 'N/A')}")
            _render_ratio_category(
                "Profitability",
                ratio_result,
                [
                    ("roe", "Return on Equity", True),
                    ("roa", "Return on Assets", True),
                    ("nim", "Net Interest Margin (proxy)", True),
                    ("efficiency_ratio", "Efficiency Ratio", True),
                    ("cost_to_income", "Cost to Income", True),
                    ("loss_ratio", "Loss Ratio", True),
                    ("expense_ratio", "Expense Ratio", True),
                    ("combined_ratio", "Combined Ratio", True),
                    ("investment_yield", "Investment Yield", True),
                    ("underwriting_profit_margin", "Underwriting Profit Margin", True),
                ],
            )
            _render_ratio_category(
                "Liquidity & Solvency",
                ratio_result,
                [
                    ("npl_ratio", "NPL Ratio", True),
                    ("coverage_ratio", "Coverage Ratio", True),
                    ("provision_to_loans", "Provision to Loans", True),
                    ("loan_loss_rate", "Loan Loss Rate", True),
                    ("tier1_capital_ratio", "Tier 1 Capital Ratio", True),
                    ("equity_to_assets", "Equity to Assets", True),
                    ("ldr", "Loan to Deposit Ratio", True),
                    ("float_to_equity", "Float to Equity", True),
                ],
            )
            _render_ratio_category(
                "Valuation",
                ratio_result,
                [
                    ("pe_ratio", "P/E Ratio", False),
                    ("pb_ratio", "P/B Ratio", False),
                    ("ps_ratio", "P/S Ratio", False),
                    ("p_to_float", "Price to Float", False),
                    ("dividend_yield", "Dividend Yield", True),
                ],
            )
            _render_ratio_category(
                "Cash Flow",
                ratio_result,
                [("float_proxy", "Float (Insurance Reserves, $)", False)],
            )
            _render_ratio_category(
                "Growth",
                ratio_result,
                [
                    ("net_income_growth_1y", "Net Income Growth (1Y)", True),
                    ("net_income_growth_2y", "Net Income Growth (2Y)", True),
                    ("nii_growth_1y", "Net Interest Income Growth (1Y)", True),
                    ("premium_growth_1y", "Premium Growth (1Y)", True),
                    ("premium_growth_2y", "Premium Growth (2Y)", True),
                ],
            )
            if ratio_result.get("casa_ratio_note"):
                st.caption(f"Note: {ratio_result['casa_ratio_note']}")

    # --------------------------------------------------------
    # TAB 3 — Comparable Companies
    # --------------------------------------------------------
    with tab_comps:
        comps_result = result.get("comps_result") or {}
        if not result["comps_available"]:
            st.info("Comparable company analysis was unavailable for this ticker (no peers found).")
        else:
            st.markdown(f"**Sector:** {comps_result.get('sector') or 'N/A'} &nbsp;|&nbsp; **Industry:** {comps_result.get('industry') or 'N/A'}")

            comp_keys = ["pe_ratio", "pb_ratio", "ps_ratio", "ev_to_ebitda", "net_margin", "roe", "revenue_growth"]
            comp_labels = ["P/E", "P/B", "P/S", "EV/EBITDA", "Net Margin", "ROE", "Rev Growth"]

            rows = []
            target_ratios = comps_result.get("target_ratios") or {}
            rows.append({"Ticker": f"{result['ticker']} (target)", **{label: target_ratios.get(key) for key, label in zip(comp_keys, comp_labels)}})
            for peer in comps_result.get("peers", []):
                peer_ratios = peer.get("ratios") or {}
                rows.append({"Ticker": peer["ticker"], **{label: peer_ratios.get(key) for key, label in zip(comp_keys, comp_labels)}})
            medians = comps_result.get("peer_medians") or {}
            rows.append({"Ticker": "Peer Median", **{label: medians.get(key) for key, label in zip(comp_keys, comp_labels)}})

            comps_df = pd.DataFrame(rows)

            def _highlight_target(row):
                if "(target)" in row["Ticker"]:
                    return ["background-color: #1B2A4A"] * len(row)
                if row["Ticker"] == "Peer Median":
                    return ["font-style: italic"] * len(row)
                return [""] * len(row)

            st.dataframe(
                comps_df.style.apply(_highlight_target, axis=1).format(precision=2, na_rep="N/A"),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**vs. Peer Median Signal**")
            vs_median = comps_result.get("vs_median") or {}
            badge_cols = st.columns(4)
            badge_colors = {"PREMIUM": "#F44336", "DISCOUNT": "#4CAF50", "IN LINE": "#64B5F6"}
            for col, key, label in zip(badge_cols, ["pe_ratio", "pb_ratio", "ev_to_ebitda", "ps_ratio"], ["P/E", "P/B", "EV/EBITDA", "P/S"]):
                signal = vs_median.get(key)
                color = badge_colors.get(signal, "#555")
                col.markdown(
                    f"<div style='text-align:center; padding:8px; border-radius:6px; background-color:{color}22; border:1px solid {color};'>"
                    f"<div style='font-size:0.8em; color:#ccc;'>{label}</div>"
                    f"<div style='font-weight:700; color:{color};'>{signal or 'N/A'}</div></div>",
                    unsafe_allow_html=True,
                )

    # --------------------------------------------------------
    # TAB 4 — Valuation Model
    # --------------------------------------------------------
    with tab_valuation:
        valuation_result = result.get("valuation_result") or {}
        if not result["valuation_available"]:
            st.warning(f"Valuation model unavailable: {valuation_result.get('error', 'unknown error')}")
        elif result["company_type"] == "non_financial":
            st.subheader("WACC Breakdown")
            wacc_col1, wacc_col2, wacc_col3, wacc_col4 = st.columns(4)
            wacc_col1.metric("Risk-Free Rate", _fmt_pct(valuation_result.get("risk_free_rate", 0) * 100 if valuation_result.get("risk_free_rate") is not None else None))
            wacc_col2.metric("Beta", _fmt_num(valuation_result.get("beta")))
            wacc_col3.metric("Cost of Equity", _fmt_pct(valuation_result.get("cost_of_equity", 0) * 100 if valuation_result.get("cost_of_equity") is not None else None))
            wacc_col4.metric("Cost of Debt", _fmt_pct(valuation_result.get("cost_of_debt", 0) * 100 if valuation_result.get("cost_of_debt") is not None else None))
            st.caption(
                f"WACC: {_fmt_pct(valuation_result.get('wacc', 0) * 100 if valuation_result.get('wacc') is not None else None)}  |  "
                f"Terminal Growth Rate: {_fmt_pct(valuation_result.get('terminal_growth_rate', 0) * 100 if valuation_result.get('terminal_growth_rate') is not None else None)}  |  "
                f"Revenue Growth Rate ({valuation_result.get('growth_source', 'N/A')}): "
                f"{_fmt_pct(valuation_result.get('revenue_growth_rate', 0) * 100 if valuation_result.get('revenue_growth_rate') is not None else None)}"
            )

            st.divider()
            st.subheader("5-Year Projection")
            years = [f"Year {i}" for i in range(1, 6)]
            projected_revenues = valuation_result.get("projected_revenues") or []
            projected_fcfs = valuation_result.get("projected_fcfs") or []
            if projected_revenues and projected_fcfs:
                chart_df = pd.DataFrame(
                    {"Projected Revenue": projected_revenues, "Projected FCFF": projected_fcfs}, index=years
                )
                st.bar_chart(chart_df)

            st.divider()
            st.subheader("Terminal Value & Enterprise Value Build-Up")
            build_col1, build_col2, build_col3 = st.columns(3)
            build_col1.metric("Terminal Value", _fmt_money(valuation_result.get("terminal_value"), 0))
            build_col2.metric("Enterprise Value", _fmt_money(valuation_result.get("enterprise_value"), 0))
            build_col3.metric("Equity Value", _fmt_money(valuation_result.get("equity_value"), 0))
        else:
            st.subheader("Residual Income Model Assumptions")
            ri_col1, ri_col2, ri_col3, ri_col4 = st.columns(4)
            ri_col1.metric("ROE", _fmt_pct(valuation_result.get("roe", 0) * 100 if valuation_result.get("roe") is not None else None))
            ri_col2.metric("Cost of Equity", _fmt_pct(valuation_result.get("cost_of_equity", 0) * 100 if valuation_result.get("cost_of_equity") is not None else None))
            ri_col3.metric("Payout Ratio", _fmt_pct(valuation_result.get("historical_payout_ratio", 0) * 100 if valuation_result.get("historical_payout_ratio") is not None else None))
            ri_col4.metric("Book Value / Share", _fmt_money(valuation_result.get("book_value_per_share")))
            st.caption(f"Payout ratio source: {valuation_result.get('payout_ratio_source', 'N/A')}")

            st.divider()
            st.subheader("5-Year Forecast")
            forecasted_eps = valuation_result.get("forecasted_eps") or []
            forecasted_book_values = valuation_result.get("forecasted_book_values") or []
            forecasted_dividends = valuation_result.get("forecasted_dividends") or []
            forecasted_residual_incomes = valuation_result.get("forecasted_residual_incomes") or []
            if forecasted_eps:
                forecast_df = pd.DataFrame(
                    {
                        "EPS": forecasted_eps,
                        "Book Value / Share": forecasted_book_values,
                        "Dividends / Share": forecasted_dividends,
                        "Residual Income": forecasted_residual_incomes,
                    },
                    index=[f"Year {i}" for i in range(1, 6)],
                )
                st.dataframe(forecast_df.style.format(precision=2), use_container_width=True)

            st.divider()
            st.metric("Terminal Residual Value", _fmt_money(valuation_result.get("terminal_residual_value")))

    # --------------------------------------------------------
    # TAB 5 — Management Guidance
    # --------------------------------------------------------
    with tab_management:
        management_result = result.get("management_result") or {}
        if not result["management_available"]:
            st.warning(f"Management guidance extraction unavailable: {management_result.get('error', 'unknown error')}")
        else:
            st.subheader("10-K Sections Analyzed")
            sections_found = management_result.get("sections_found") or []
            word_counts = management_result.get("section_word_counts") or {}
            if sections_found:
                section_rows = [
                    {
                        "Section": section,
                        "Raw Words": word_counts.get(section, {}).get("raw_word_count"),
                        "Words Used": word_counts.get(section, {}).get("final_word_count"),
                    }
                    for section in sections_found
                ]
                st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True)
            st.caption(f"Filing: {management_result.get('filing_url', 'N/A')} (filed {management_result.get('filing_date', 'N/A')})")

            st.divider()
            st.subheader("Extracted Guidance")
            extracted = management_result.get("extracted_guidance") or {}
            g1, g2 = st.columns(2)
            with g1:
                st.markdown(f"**Revenue Growth Guidance:** {extracted.get('revenue_growth_guidance') if extracted.get('revenue_growth_guidance') is not None else 'Not stated'}")
                st.markdown(f"**Revenue Growth (Qualitative):** {extracted.get('revenue_growth_qualitative') or 'Not stated'}")
                st.markdown(f"**Operating Margin Guidance:** {extracted.get('operating_margin_guidance') if extracted.get('operating_margin_guidance') is not None else 'Not stated'}")
                st.markdown(f"**Operating Margin Trend:** {extracted.get('operating_margin_trend') or 'N/A'}")
                st.markdown(f"**Gross Margin Trend:** {extracted.get('gross_margin_trend') or 'N/A'}")
                st.markdown(f"**Economies of Scale Mentioned:** {extracted.get('economies_of_scale_mentioned')}")
            with g2:
                st.markdown(f"**Capex Guidance:** {extracted.get('capex_guidance') if extracted.get('capex_guidance') is not None else 'Not stated'}")
                st.markdown(f"**Capex Trend:** {extracted.get('capex_trend') or 'N/A'}")
                st.markdown(f"**Management Tone:** {extracted.get('management_tone') or 'N/A'}")
                st.markdown(f"**Capital Return Policy:** {extracted.get('capital_return_policy') or 'Not stated'}")
                new_markets = extracted.get("new_products_or_markets") or []
                st.markdown(f"**New Products / Markets:** {', '.join(new_markets) if new_markets else 'None extracted'}")

            st.divider()
            confidence_score = extracted.get("confidence_score")
            st.subheader("Confidence Score")
            if confidence_score is not None:
                st.progress(min(max(confidence_score, 0.0), 1.0))
                if confidence_score >= 0.8:
                    explanation = "Management gave explicit numerical targets."
                elif confidence_score >= 0.4:
                    explanation = "Clear qualitative direction, but no hard numbers."
                else:
                    explanation = "Vague or boilerplate language — treat with caution."
                st.caption(f"{confidence_score:.2f} — {explanation}")

            st.divider()
            st.subheader("Recommended DCF Overrides")
            overrides = result.get("management_recommended_overrides") or {}
            if confidence_score is not None and confidence_score > 0.6 and overrides:
                for field, value in overrides.items():
                    st.markdown(f"- **{field}**: {value}")
            else:
                st.caption("No overrides recommended (confidence below 0.6 or guidance was qualitative only).")

            st.divider()
            st.subheader("Full Management Summary")
            st.write(management_result.get("management_summary") or "Not available.")

    # --------------------------------------------------------
    # TAB 6 — Data Quality
    # --------------------------------------------------------
    with tab_quality:
        evaluation = result.get("evaluation") or {}

        if "error" in evaluation or evaluation.get("overall_score") is None:
            st.warning(f"Quality evaluation unavailable: {evaluation.get('error', 'unknown error')}")
        else:
            # --- Section 1: overall quality score ---------------------------
            if evaluation.get("show_warning"):
                st.markdown(
                    "<div style='background-color:#4A3B14; border:1px solid #FFC107; padding:14px; "
                    "border-radius:8px; margin-bottom:16px; color:#FFC107; font-weight:600;'>"
                    "⚠️ Data quality concerns detected — review the details below before making investment decisions"
                    "</div>",
                    unsafe_allow_html=True,
                )

            rating_label = _quality_rating_label(evaluation.get("quality_rating"))
            rating_color = QUALITY_RATING_COLORS.get(rating_label, "#888")

            score_col, badge_col = st.columns([1, 2])
            with score_col:
                st.metric("Overall Quality Score", f"{evaluation['overall_score']:.2f} / 5.0")
            with badge_col:
                st.markdown(
                    f"<div style='margin-top:22px;'><span style='background:{rating_color}22; "
                    f"border:1px solid {rating_color}; color:{rating_color}; padding:8px 18px; "
                    f"border-radius:20px; font-weight:700; font-size:1.1em;'>{evaluation.get('quality_rating')}</span></div>",
                    unsafe_allow_html=True,
                )

            st.divider()

            # --- Section 2: dimension scores table ---------------------------
            st.subheader("Dimension Scores")
            dimension_scores = evaluation.get("dimension_scores") or {}
            dimension_rows = [
                {"Dimension": DIMENSION_LABELS.get(key, key), "Score": score, "Rating": _score_circles(score)}
                for key, score in dimension_scores.items()
            ]
            st.dataframe(pd.DataFrame(dimension_rows), use_container_width=True, hide_index=True)

            st.divider()

            # --- Section 3: issues and recommendations ------------------------
            st.subheader("Issues and Recommendations")
            dimension_details = evaluation.get("dimension_details") or {}
            any_failed = False
            for dimension_key, details in dimension_details.items():
                for failure in details.get("failed_checks") or []:
                    any_failed = True
                    st.markdown(f"⚠️ **{DIMENSION_LABELS.get(dimension_key, dimension_key)}:** {failure['reason']}")
            if not any_failed:
                st.caption("No failed checks across any dimension.")

            st.write("")
            st.markdown("**Recommendations:**")
            for recommendation in evaluation.get("recommendations") or []:
                st.markdown(f"- {recommendation}")

            st.divider()

            # --- Section 4: missing fields ------------------------------------
            missing_fields = (dimension_details.get("data_completeness") or {}).get("missing_fields") or []
            if missing_fields:
                st.subheader("Missing Fields")
                st.markdown("The following data fields were unavailable from SEC EDGAR for this ticker:")
                for field in missing_fields:
                    affects = FIELD_EXPLANATIONS.get(field, "one or more ratios in this analysis")
                    st.markdown(f"- **{field}** — affects {affects}")

            st.divider()

            # --- Section 5: evaluation metadata --------------------------------
            st.subheader("Evaluation Metadata")
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.markdown(f"**Ticker**\n\n{evaluation.get('ticker')}")
            meta_col2.markdown(f"**Company Type**\n\n{evaluation.get('company_type')}")
            meta_col3.markdown(f"**Evaluated At**\n\n{evaluation.get('evaluation_timestamp')}")
