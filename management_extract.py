"""Extract targeted, high-value sections from a company's latest 10-K and use
Claude to pull structured forward-looking management guidance out of them.

Rather than sending an entire 10-K (often 100k+ words) to Claude, this module
locates just the sections that actually carry forward-looking signal — MD&A,
Risk Factors, the market risk disclosure, and the business overview — caps
each to a token budget, and sends only that.
"""

import io
import json
import os
import re

import anthropic
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from data_fetch import SEC_HEADERS

load_dotenv()

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
CLAUDE_MODEL = "claude-sonnet-4-6"

MAX_SECTION_WORDS = 3000
MAX_TOTAL_WORDS = 12000

# Every genuine "Item N" heading in a 10-K starts at the beginning of a line
# and carries its title on that same line (e.g. "Item 7.\xa0\xa0Management's
# Discussion..."). Table-of-contents entries put the title on a separate
# line/cell instead, and cross-references ("as discussed in Item 7...") never
# start a line at all — so anchoring on line-start plus requiring same-line
# title text filters out both without any extra bookkeeping.
GENERIC_ITEM_HEADING_REGEX = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\.?", re.IGNORECASE | re.MULTILINE)

# (item number as it appears in the regex capture, internal key, output label,
# keywords that must appear in the heading's own title text, priority tier).
# Item 2 in a standard 10-K is "Properties", not MD&A — that pairing is kept
# here only as a defensive catch-all for non-standard filers; the keyword gate
# means it will simply never fire for a normal large-filer 10-K.
TARGET_SECTIONS = [
    ("7", "item_7", "ITEM 7: MD&A",
     ["management", "discussion", "analysis", "financial condition", "results of operations"], "high"),
    ("2", "item_2", "ITEM 2: MD&A",
     ["management", "discussion", "analysis", "financial condition", "results of operations"], "high"),
    ("7A", "item_7a", "ITEM 7A: QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
     ["quantitative", "qualitative", "market risk"], "high"),
    ("1A", "item_1a", "ITEM 1A: RISK FACTORS",
     ["risk", "risk factors"], "high"),
    ("1", "item_1", "ITEM 1: BUSINESS OVERVIEW",
     ["business", "overview", "strategy", "products", "services", "competition", "market"], "medium"),
]

# Truncation priority when the combined text exceeds MAX_TOTAL_WORDS: trim
# lowest-priority sections first (Item 1, then Item 7A, then Item 1A, then
# the MD&A sections), per the requested "Item 7/2 > Item 1A > Item 7A > Item 1".
TRUNCATION_PRIORITY_ORDER = ["item_7", "item_2", "item_1a", "item_7a", "item_1"]

EXTRACTION_SYSTEM_PROMPT = (
    "You are a senior equity research analyst. Extract forward-looking quantitative "
    "guidance and qualitative signals from the following 10-K sections. These sections "
    "may include MD&A, Risk Factors, and Business Overview. Return ONLY a valid JSON "
    "object with no markdown, no code fences, no other text."
)

EXTRACTION_SCHEMA_INSTRUCTIONS = """Extract the following fields as a single JSON object:
{
  "revenue_growth_guidance": number or null (explicit management guidance as decimal e.g. 0.08),
  "revenue_growth_qualitative": string or null (e.g. "expects double digit growth driven by Services"),
  "operating_margin_guidance": number or null (explicit target as decimal),
  "operating_margin_trend": "expanding" or "contracting" or "stable" or null,
  "capex_guidance": number or null (in dollars, explicit figure only),
  "capex_trend": "increasing" or "decreasing" or "stable" or null,
  "gross_margin_trend": "expanding" or "contracting" or "stable" or null,
  "economies_of_scale_mentioned": true or false,
  "key_growth_drivers": list of max 4 strings (most important only),
  "key_risks": list of max 4 strings (most important only),
  "new_products_or_markets": list of max 3 strings or null,
  "management_tone": "optimistic" or "cautious" or "neutral",
  "capital_return_policy": string or null (dividends, buybacks mentioned),
  "confidence_score": number 0 to 1 (1 = very explicit numerical guidance, 0 = vague qualitative only)
}

Return null for any field where guidance is not explicitly stated. Do not infer or
hallucinate numbers. For confidence_score: use 0.8-1.0 only if management gave explicit
numerical targets, 0.4-0.7 for clear qualitative direction, 0.1-0.3 for vague or
boilerplate language."""

# Only these guidance fields are hard numbers eligible to override a DCF
# assumption; the *_trend/*_qualitative/list fields are directional color, not
# something that should silently overwrite a numeric model input.
OVERRIDE_FIELD_MAP = {
    "revenue_growth_guidance": "revenue_growth_rate",
    "operating_margin_guidance": "operating_margin",
    "capex_guidance": "capex",
}
OVERRIDE_CONFIDENCE_THRESHOLD = 0.6


def _get_cik(ticker):
    """Best-effort CIK lookup; returns None (never raises) if it can't be found."""
    try:
        response = requests.get(TICKERS_URL, headers=SEC_HEADERS, timeout=15)
        response.raise_for_status()
        ticker_map = response.json()
    except (requests.RequestException, ValueError):
        return None

    ticker_upper = ticker.upper()
    for entry in ticker_map.values():
        if str(entry.get("ticker", "")).upper() == ticker_upper:
            return str(entry.get("cik_str")).zfill(10)
    return None


def _get_latest_10k_url(ticker):
    """Locate the most recently filed 10-K's primary document on SEC EDGAR.
    Returns (url, filing_date) or (None, None) if it can't be found.
    """
    cik = _get_cik(ticker)
    if cik is None:
        return None, None

    try:
        response = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=SEC_HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None, None

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    for form, accession, primary_doc, filing_date in zip(forms, accession_numbers, primary_documents, filing_dates):
        if form == "10-K":
            accession_no_dashes = accession.replace("-", "")
            cik_no_zeros = str(int(cik))
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_no_dashes}/{primary_doc}"
            return url, filing_date
    return None, None


def _find_real_headings(full_text):
    """Every genuine Item-N heading in the document, in document order, as
    {"item_number", "start", "title"} dicts. Filters out table-of-contents
    entries and cross-references (see GENERIC_ITEM_HEADING_REGEX comment).
    """
    headings = []
    for match in GENERIC_ITEM_HEADING_REGEX.finditer(full_text):
        start = match.start()
        line_end = full_text.find("\n", start)
        if line_end == -1:
            line_end = len(full_text)
        title = full_text[match.end():line_end].strip(" .\xa0\t")
        if len(title) > 3:
            headings.append({"item_number": match.group(1).upper(), "start": start, "title": title})
    return headings


def _extract_target_sections(full_text):
    """Slice out the text of each TARGET_SECTIONS entry found in `full_text`,
    bounded by the next real heading of any kind (so e.g. Item 1A's text stops
    at Item 1B, not at the next tracked section). Returns
    {key: {"label", "priority", "text", "word_count"}}.
    """
    headings = _find_real_headings(full_text)
    sections = {}
    for i, heading in enumerate(headings):
        for item_number, key, label, keywords, priority in TARGET_SECTIONS:
            if heading["item_number"] != item_number:
                continue
            if not any(keyword in heading["title"].lower() for keyword in keywords):
                continue
            end = headings[i + 1]["start"] if i + 1 < len(headings) else len(full_text)
            text = full_text[heading["start"]:end].strip()
            sections[key] = {"label": label, "priority": priority, "text": text, "word_count": len(text.split())}
    return sections


def _cap_words(text, max_words):
    words = text.split()
    return " ".join(words[:max_words])


def _build_combined_text(sections):
    """Cap each section to MAX_SECTION_WORDS, then trim lowest-priority
    sections first if the combined total still exceeds MAX_TOTAL_WORDS.
    Returns (combined_text, sections_found, section_word_counts) where
    section_word_counts maps each label to (raw_word_count, final_word_count).
    """
    capped_words = {key: _cap_words(data["text"], MAX_SECTION_WORDS).split() for key, data in sections.items()}

    total_words = sum(len(words) for words in capped_words.values())
    if total_words > MAX_TOTAL_WORDS:
        for key in reversed(TRUNCATION_PRIORITY_ORDER):
            if key not in capped_words or total_words <= MAX_TOTAL_WORDS:
                continue
            excess = total_words - MAX_TOTAL_WORDS
            words = capped_words[key]
            new_len = max(0, len(words) - excess)
            total_words -= len(words) - new_len
            capped_words[key] = words[:new_len]

    combined_parts = []
    sections_found = []
    section_word_counts = {}
    for key in TRUNCATION_PRIORITY_ORDER:
        if key not in sections:
            continue
        final_words = capped_words[key]
        if not final_words:
            continue
        label = sections[key]["label"]
        combined_parts.append(f"=== {label} ===\n{' '.join(final_words)}")
        sections_found.append(label)
        section_word_counts[label] = {"raw_word_count": sections[key]["word_count"], "final_word_count": len(final_words)}

    return "\n\n".join(combined_parts), sections_found, section_word_counts


def extract_10k_sections(ticker):
    """Fetch the latest 10-K for `ticker` and pull out just the targeted
    high-value sections. Returns a dict with combined_text, sections_found,
    and section_word_counts, or {"error": ...} if the filing can't be
    located, fetched, or contains none of the target sections.
    """
    url, filing_date = _get_latest_10k_url(ticker)
    if url is None:
        return {"error": f"could not locate a 10-K filing for {ticker}"}

    try:
        response = requests.get(url, headers=SEC_HEADERS, timeout=60)
        response.raise_for_status()
        html = response.text
    except requests.RequestException as exc:
        return {"error": f"failed to fetch 10-K document: {exc}"}

    soup = BeautifulSoup(html, "html.parser")
    full_text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n"))

    sections = _extract_target_sections(full_text)
    combined_text, sections_found, section_word_counts = _build_combined_text(sections)

    if not combined_text.strip():
        return {
            "error": "no target sections could be located in the 10-K document",
            "filing_url": url,
        }

    return {
        "ticker": ticker.upper(),
        "filing_url": url,
        "filing_date": filing_date,
        "combined_text": combined_text,
        "sections_found": sections_found,
        "section_word_counts": section_word_counts,
    }


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
    """Claude occasionally wraps JSON in ```json ... ``` despite being told not
    to — strip that off before parsing rather than trusting prompt compliance."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_management_guidance(combined_text):
    """Send the combined 10-K sections to Claude and parse the structured
    guidance extraction. Raises RuntimeError if the API call or JSON parsing fails.
    """
    client = _get_claude_client()
    user_prompt = f"{EXTRACTION_SCHEMA_INSTRUCTIONS}\n\n10-K SECTIONS:\n\n{combined_text}"

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = next((block.text for block in response.content if block.type == "text"), "")
        return json.loads(_strip_code_fences(raw_text))
    except (anthropic.APIError, json.JSONDecodeError, StopIteration) as exc:
        raise RuntimeError(f"failed to extract management guidance: {exc}") from exc


def generate_management_summary(extracted_guidance):
    """A second, smaller Claude call that turns the structured JSON extraction
    into a 3-sentence plain-English summary for a human reader.
    """
    client = _get_claude_client()
    prompt = (
        "In exactly 3 sentences, summarize the key takeaways for an investor from "
        "this structured extraction of a company's 10-K management guidance. Plain "
        "English, no jargon, no bullet points.\n\n" + json.dumps(extracted_guidance)
    )
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return next((block.text for block in response.content if block.type == "text"), "").strip()
    except anthropic.APIError as exc:
        return f"(summary unavailable: {exc})"


def apply_management_guidance(extracted_guidance, management_summary=None):
    """Decide which extracted guidance fields are strong enough to act as DCF
    assumption overrides: confidence_score must exceed OVERRIDE_CONFIDENCE_THRESHOLD
    AND the field must be a hard number, not qualitative color. This only
    returns the override recommendations — it does not modify dcf.py or any
    financials dict itself.
    """
    confidence_score = extracted_guidance.get("confidence_score") or 0
    overrides = {}
    if confidence_score > OVERRIDE_CONFIDENCE_THRESHOLD:
        for guidance_field, dcf_field in OVERRIDE_FIELD_MAP.items():
            value = extracted_guidance.get(guidance_field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                overrides[dcf_field] = value

    return {
        "overrides": overrides,
        "confidence_score": confidence_score,
        "management_summary": management_summary,
    }


def run_management_extraction(ticker):
    """Full pipeline: locate + extract 10-K sections, run the Claude guidance
    extraction and summary, and compute recommended DCF overrides.
    """
    section_data = extract_10k_sections(ticker)
    if "error" in section_data:
        return section_data

    extracted_guidance = extract_management_guidance(section_data["combined_text"])
    management_summary = generate_management_summary(extracted_guidance)
    override_info = apply_management_guidance(extracted_guidance, management_summary)

    return {
        "ticker": section_data["ticker"],
        "filing_url": section_data["filing_url"],
        "filing_date": section_data["filing_date"],
        "sections_found": section_data["sections_found"],
        "section_word_counts": section_data["section_word_counts"],
        "extracted_guidance": extracted_guidance,
        "management_summary": management_summary,
        "confidence_score": override_info["confidence_score"],
        "recommended_overrides": override_info["overrides"],
    }


if __name__ == "__main__":
    result = run_management_extraction("AAPL")

    if "error" in result:
        print("Error:", result["error"])
    else:
        print(f"Filing: {result['filing_url']} (filed {result['filing_date']})")
        print()
        print("Sections found:", result["sections_found"])
        print()
        print("Word counts per section (raw -> after capping):")
        for label, counts in result["section_word_counts"].items():
            print(f"  {label}: {counts['raw_word_count']} -> {counts['final_word_count']}")
        print()
        print("Extracted JSON:")
        print(json.dumps(result["extracted_guidance"], indent=2))
        print()
        print(f"Confidence score: {result['confidence_score']}")
        print()
        print("Management summary:")
        print(result["management_summary"])
        print()
        print("Recommended DCF overrides (confidence > 0.6, hard numbers only):")
        if result["recommended_overrides"]:
            for field, value in result["recommended_overrides"].items():
                print(f"  {field}: {value}")
        else:
            print("  (none - confidence below threshold or guidance was qualitative only)")
