"""Robust shares-outstanding lookup with fallback across multiple data
sources, since some companies (dual-class structures, foreign filers, certain
financials) don't cleanly report CommonStockSharesOutstanding in SEC EDGAR XBRL.
"""

import requests
import yfinance as yf

from data_fetch import COMPANY_FACTS_URL, SEC_HEADERS, TICKERS_URL, _latest_value


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


def _get_edgar_weighted_average_shares(ticker):
    """Most recent 10-K WeightedAverageNumberOfSharesOutstandingBasic from SEC
    EDGAR — the average share count during the year rather than a point-in-time
    count, but a reasonable last resort. Returns None (never raises) if
    unavailable.
    """
    cik = _get_cik(ticker)
    if cik is None:
        return None
    try:
        response = requests.get(COMPANY_FACTS_URL.format(cik=cik), headers=SEC_HEADERS, timeout=15)
        response.raise_for_status()
        company_facts = response.json()
    except (requests.RequestException, ValueError):
        return None

    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    return _latest_value(us_gaap, ["WeightedAverageNumberOfSharesOutstandingBasic"], unit_keys=("shares",))


def get_shares_outstanding(ticker, financials):
    """Return (shares_outstanding, source_tag) for `ticker`, trying each
    source in order and returning the first valid (not None, > 0) result:

    1. edgar_xbrl            - financials["shares_outstanding"] (already fetched)
    2. yfinance_fast_info    - yfinance.Ticker(ticker).fast_info["shares"]
    3. yfinance_info         - yfinance.Ticker(ticker).info["sharesOutstanding"]
    4. computed_from_market_cap - fast_info marketCap / fast_info lastPrice
    5. edgar_weighted_average - EDGAR's WeightedAverageNumberOfSharesOutstandingBasic

    Shares outstanding is too foundational an input to silently return None
    for, so if every source fails this raises ValueError instead.
    """
    edgar_shares = financials.get("shares_outstanding")
    if edgar_shares is not None and edgar_shares > 0:
        return edgar_shares, "edgar_xbrl"

    try:
        fast_shares = yf.Ticker(ticker).fast_info.get("shares")
    except Exception:
        fast_shares = None
    if fast_shares is not None and fast_shares > 0:
        return fast_shares, "yfinance_fast_info"

    try:
        info_shares = yf.Ticker(ticker).info.get("sharesOutstanding")
    except Exception:
        info_shares = None
    if info_shares is not None and info_shares > 0:
        return info_shares, "yfinance_info"

    try:
        fast_info = yf.Ticker(ticker).fast_info
        market_cap = fast_info.get("marketCap")
        price = fast_info.get("lastPrice")
    except Exception:
        market_cap = None
        price = None
    if market_cap is not None and price is not None and price > 0:
        computed_shares = market_cap / price
        if computed_shares > 0:
            return computed_shares, "computed_from_market_cap"

    weighted_average_shares = _get_edgar_weighted_average_shares(ticker)
    if weighted_average_shares is not None and weighted_average_shares > 0:
        return weighted_average_shares, "edgar_weighted_average"

    raise ValueError(f"could not determine shares outstanding for {ticker} from any source")


def _get_raw_edgar_shares(ticker):
    """Raw CommonStockSharesOutstanding from SEC EDGAR with no fallback
    applied — used only to build a realistic, uncorrected test input below
    (data_fetch.fetch_financials already applies the correction internally,
    so calling it here would just re-confirm Source 1 trivially).
    """
    cik = _get_cik(ticker)
    if cik is None:
        return None
    try:
        response = requests.get(COMPANY_FACTS_URL.format(cik=cik), headers=SEC_HEADERS, timeout=15)
        response.raise_for_status()
        company_facts = response.json()
    except (requests.RequestException, ValueError):
        return None
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    return _latest_value(us_gaap, ["CommonStockSharesOutstanding"], unit_keys=("shares",))


if __name__ == "__main__":
    test_tickers = ["AAPL", "BRK-B", "JPM", "GOOGL"]

    print(f"{'Ticker':<10}{'Shares Outstanding (B)':>26}{'Source':>26}")
    print("-" * 62)
    for ticker in test_tickers:
        raw_shares = _get_raw_edgar_shares(ticker)
        shares, source = get_shares_outstanding(ticker, {"shares_outstanding": raw_shares})
        print(f"{ticker:<10}{shares / 1e9:>26.3f}{source:>26}")
