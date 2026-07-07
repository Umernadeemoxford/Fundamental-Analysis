"""Peer comparison: benchmark a ticker's valuation multiples against sector peers."""

import json
import statistics

import yfinance as yf

# Yahoo/yfinance has no stable, public "related tickers" field — recommendationKey
# is an analyst buy/hold/sell rating, not a peer list, and info payloads don't
# reliably expose peers either. So peers are looked up from this hardcoded map,
# keyed by the `sector` string yfinance's .info returns. Six candidates per sector
# so the target ticker can be filtered out and 5 peers still remain.
SECTOR_PEER_MAP = {
    "Technology": ["AAPL", "MSFT", "NVDA", "ORCL", "CRM", "ADBE"],
    "Healthcare": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST", "PM"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX"],
    "Industrials": ["CAT", "HON", "UPS", "BA", "GE", "LMT"],
    "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "VZ", "T"],
}

# yfinance/Yahoo sometimes labels the sector "Financial Services" instead of
# "Financials" - normalize so both resolve to the same map entry.
SECTOR_ALIASES = {"Financial Services": "Financials"}

RATIO_KEYS = ["pe_ratio", "pb_ratio", "ps_ratio", "ev_to_ebitda", "net_margin", "roe", "revenue_growth"]
VALUATION_KEYS = ["pe_ratio", "pb_ratio", "ev_to_ebitda", "ps_ratio"]


def _get_info(ticker):
    """Best-effort fetch of yfinance's info dict; never raises."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def _get_live_price(ticker):
    """Best-effort current price lookup; returns None rather than raising."""
    try:
        return yf.Ticker(ticker).fast_info["lastPrice"]
    except Exception:
        info = _get_info(ticker)
        return info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")


def _get_sector_and_industry(ticker):
    info = _get_info(ticker)
    return info.get("sector"), info.get("industry")


def _round(value):
    return round(value, 2) if isinstance(value, (int, float)) else None


def _pct(value):
    return round(value * 100, 2) if isinstance(value, (int, float)) else None


def get_peers(ticker):
    """Return up to 5 peer ticker symbols in the same sector as `ticker`."""
    sector, _industry = _get_sector_and_industry(ticker)
    sector = SECTOR_ALIASES.get(sector, sector)
    candidates = SECTOR_PEER_MAP.get(sector, [])
    return [t for t in candidates if t.upper() != ticker.upper()][:5]


def get_peer_ratios(ticker, market_price=None):
    """Pull key valuation/profitability multiples for `ticker` from yfinance's info.

    Returns None for any field that is missing or non-numeric; never raises.
    """
    info = _get_info(ticker)

    # pe_ratio: price paid per dollar of trailing earnings - how expensive vs. profit.
    pe_ratio = _round(info.get("trailingPE"))
    if pe_ratio is None and market_price is not None:
        eps = info.get("trailingEps")
        if isinstance(eps, (int, float)) and eps != 0:
            pe_ratio = round(market_price / eps, 2)

    # pb_ratio: price paid per dollar of book equity - premium over net asset value.
    pb_ratio = _round(info.get("priceToBook"))
    if pb_ratio is None and market_price is not None:
        book_value = info.get("bookValue")
        if isinstance(book_value, (int, float)) and book_value != 0:
            pb_ratio = round(market_price / book_value, 2)

    # ps_ratio: price paid per dollar of revenue - useful when earnings are unstable.
    ps_ratio = _round(info.get("priceToSalesTrailing12Months"))
    if ps_ratio is None and market_price is not None:
        revenue_per_share = info.get("revenuePerShare")
        if isinstance(revenue_per_share, (int, float)) and revenue_per_share != 0:
            ps_ratio = round(market_price / revenue_per_share, 2)

    # ev_to_ebitda: valuation multiple normalized for capital structure/leverage.
    ev_to_ebitda = _round(info.get("enterpriseToEbitda"))

    # net_margin: % of revenue that becomes bottom-line profit - profitability quality.
    net_margin = _pct(info.get("profitMargins"))

    # roe: return generated on shareholders' invested capital.
    roe = _pct(info.get("returnOnEquity"))

    # revenue_growth: year-over-year top-line growth - momentum relative to peers.
    revenue_growth = _pct(info.get("revenueGrowth"))

    return {
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
        "ps_ratio": ps_ratio,
        "ev_to_ebitda": ev_to_ebitda,
        "net_margin": net_margin,
        "roe": roe,
        "revenue_growth": revenue_growth,
    }


def _vs_median_label(target_value, median_value):
    """PREMIUM if >15% above peer median, DISCOUNT if >15% below, else IN LINE."""
    if target_value is None or median_value is None or median_value == 0:
        return None
    diff_pct = (target_value - median_value) / median_value * 100
    if diff_pct > 15:
        return "PREMIUM"
    if diff_pct < -15:
        return "DISCOUNT"
    return "IN LINE"


def run_comps(ticker):
    """Benchmark `ticker` against 5 sector peers on valuation/profitability multiples."""
    sector, industry = _get_sector_and_industry(ticker)

    peer_tickers = get_peers(ticker)
    target_ratios = get_peer_ratios(ticker, _get_live_price(ticker))

    peers = []
    for peer_ticker in peer_tickers:
        peer_ratios = get_peer_ratios(peer_ticker, _get_live_price(peer_ticker))
        peers.append({"ticker": peer_ticker, "ratios": peer_ratios})

    peer_medians = {}
    for key in RATIO_KEYS:
        values = [p["ratios"][key] for p in peers if p["ratios"][key] is not None]
        peer_medians[key] = round(statistics.median(values), 2) if values else None

    # PREMIUM/DISCOUNT/IN LINE only makes sense for price-based valuation multiples,
    # not profitability/growth metrics, so it's limited to the four valuation keys.
    vs_median = {
        key: _vs_median_label(target_ratios.get(key), peer_medians.get(key)) for key in VALUATION_KEYS
    }

    return {
        "ticker": ticker.upper(),
        "sector": sector,
        "industry": industry,
        "target_ratios": target_ratios,
        "peers": peers,
        "peer_medians": peer_medians,
        "vs_median": vs_median,
    }


if __name__ == "__main__":
    result = run_comps("AAPL")
    print(json.dumps(result, indent=2))

    print()
    print(f"{result['ticker']} ({result['sector']} / {result['industry']}) vs. peer median:")
    print(f"{'Metric':<15}{'AAPL':>12}{'Peer Median':>15}{'Verdict':>12}")
    for key in VALUATION_KEYS:
        target_val = result["target_ratios"].get(key)
        median_val = result["peer_medians"].get(key)
        verdict = result["vs_median"].get(key) or "N/A"
        target_str = "N/A" if target_val is None else str(target_val)
        median_str = "N/A" if median_val is None else str(median_val)
        print(f"{key:<15}{target_str:>12}{median_str:>15}{verdict:>12}")
