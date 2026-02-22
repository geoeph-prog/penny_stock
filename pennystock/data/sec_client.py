"""SEC EDGAR client for insider transactions and filing data. Free, no API key."""

import time
from datetime import datetime, timedelta

import requests
from loguru import logger

SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

HEADERS = {
    "User-Agent": "PennyStockAnalyzer research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Module-level cache for ticker->CIK mapping
_ticker_cik_map = None


def _load_ticker_map():
    """Load SEC ticker-to-CIK mapping (cached)."""
    global _ticker_cik_map
    if _ticker_cik_map is not None:
        return _ticker_cik_map

    try:
        resp = requests.get(SEC_TICKERS_URL, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        _ticker_cik_map = {}
        for entry in data.values():
            ticker = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", ""))
            if ticker and cik:
                _ticker_cik_map[ticker] = cik.zfill(10)
        logger.debug(f"Loaded {len(_ticker_cik_map)} ticker-CIK mappings")
        return _ticker_cik_map
    except Exception as e:
        logger.debug(f"Failed to load SEC ticker map: {e}")
        return {}


def get_insider_transactions(ticker: str, days_back: int = 90) -> dict:
    """
    Get recent insider transactions (Form 4 filings) for a ticker.

    Returns:
        {
            "ticker": str,
            "total_filings": int,
            "recent_buys": int,
            "recent_sells": int,
            "net_direction": str,  # "buying", "selling", "neutral"
            "filings": list[dict],
        }
    """
    ticker_map = _load_ticker_map()
    cik = ticker_map.get(ticker.upper())

    if not cik:
        return _empty_insider_result(ticker)

    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        params = {
            "q": f'"{ticker}"',
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "forms": "4",
        }
        resp = requests.get(SEC_SEARCH_URL, headers=HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            return _empty_insider_result(ticker)

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])

        filings = []
        buys = 0
        sells = 0

        for hit in hits[:20]:  # Cap at 20 most recent
            source = hit.get("_source", {})
            form_type = source.get("form_type", "")
            filed = source.get("file_date", "")
            display = source.get("display_names", [""])[0] if source.get("display_names") else ""

            # Try to determine buy vs sell from filing text
            title = (source.get("display_date_filed", "") + " " + display).lower()
            is_buy = any(w in title for w in ["purchase", "acquisition", "buy", "grant"])
            is_sell = any(w in title for w in ["sale", "sold", "sell", "disposition"])

            if is_buy:
                buys += 1
            elif is_sell:
                sells += 1

            filings.append({
                "form": form_type,
                "date": filed,
                "filer": display,
                "direction": "buy" if is_buy else ("sell" if is_sell else "unknown"),
            })

        if buys > sells:
            net = "buying"
        elif sells > buys:
            net = "selling"
        else:
            net = "neutral"

        return {
            "ticker": ticker,
            "total_filings": len(filings),
            "recent_buys": buys,
            "recent_sells": sells,
            "net_direction": net,
            "filings": filings,
        }

    except Exception as e:
        logger.debug(f"SEC insider lookup failed for {ticker}: {e}")
        return _empty_insider_result(ticker)


def get_recent_filings(ticker: str, forms: str = "8-K,10-Q,10-K", days_back: int = 90) -> dict:
    """
    Get recent SEC filings (8-K material events, 10-Q quarterlies, etc.).

    Returns:
        {
            "ticker": str,
            "total_filings": int,
            "filing_types": dict[str, int],  # count by type
            "filings": list[dict],
        }
    """
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        params = {
            "q": f'"{ticker}"',
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "forms": forms,
        }
        resp = requests.get(SEC_SEARCH_URL, headers=HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            return {"ticker": ticker, "total_filings": 0, "filing_types": {}, "filings": []}

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])

        filings = []
        type_counts = {}

        for hit in hits[:30]:
            source = hit.get("_source", {})
            form_type = source.get("form_type", "unknown")
            type_counts[form_type] = type_counts.get(form_type, 0) + 1

            filings.append({
                "form": form_type,
                "date": source.get("file_date", ""),
                "description": source.get("display_names", [""])[0] if source.get("display_names") else "",
            })

        return {
            "ticker": ticker,
            "total_filings": len(filings),
            "filing_types": type_counts,
            "filings": filings,
        }

    except Exception as e:
        logger.debug(f"SEC filings lookup failed for {ticker}: {e}")
        return {"ticker": ticker, "total_filings": 0, "filing_types": {}, "filings": []}


def search_filing_text(ticker: str, search_term: str, forms: str = "10-K,10-Q",
                       days_back: int = 365) -> dict:
    """
    Search SEC EDGAR full-text index for a specific term in recent filings.

    This is used by kill filters to detect "going concern" language,
    restatements, or other red-flag disclosures in 10-K/10-Q filings.

    Args:
        ticker: Stock ticker symbol.
        search_term: Text to search for (e.g., "going concern").
        forms: Comma-separated SEC form types to search.
        days_back: How far back to search.

    Returns:
        {
            "found": bool,          # Whether the term was found
            "match_count": int,     # Number of filings containing the term
            "filings": list[dict],  # Matching filing details
        }
    """
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        params = {
            "q": f'"{ticker}" "{search_term}"',
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "forms": forms,
        }
        resp = requests.get(SEC_SEARCH_URL, headers=HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            return {"found": False, "match_count": 0, "filings": []}

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])

        filings = []
        for hit in hits[:10]:
            source = hit.get("_source", {})
            filings.append({
                "form": source.get("form_type", ""),
                "date": source.get("file_date", ""),
                "description": (source.get("display_names", [""])[0]
                                if source.get("display_names") else ""),
            })

        return {
            "found": len(filings) > 0,
            "match_count": len(filings),
            "filings": filings,
        }

    except Exception as e:
        logger.debug(f"SEC text search failed for '{search_term}' in {ticker}: {e}")
        return {"found": False, "match_count": 0, "filings": []}


def check_going_concern(ticker: str) -> bool:
    """
    Check if a company has 'going concern' language in recent SEC filings.
    Checks both 10-K/10-Q (US filers) and 20-F (foreign filers).
    Returns True if going concern was found (= BAD, should be killed).
    """
    # Check US filers (10-K, 10-Q)
    result = search_filing_text(ticker, "going concern", forms="10-K,10-Q", days_back=365)
    if result["found"]:
        logger.info(f"KILL FLAG: {ticker} has 'going concern' in {result['match_count']} "
                     f"recent 10-K/10-Q filing(s)")
        return True

    # Check foreign filers (20-F, 6-K)
    result_foreign = search_filing_text(ticker, "going concern", forms="20-F,6-K", days_back=365)
    if result_foreign["found"]:
        logger.info(f"KILL FLAG: {ticker} has 'going concern' in {result_foreign['match_count']} "
                     f"recent 20-F/6-K filing(s)")
        return True

    return False


def check_dilution_filings(ticker: str, days_back: int = 180) -> dict:
    """
    Check for S-1, S-3 registration statements (share issuance / dilution).
    Returns count of dilution-related filings.
    """
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        params = {
            "q": f'"{ticker}"',
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "forms": "S-1,S-3,424B",
        }
        resp = requests.get(SEC_SEARCH_URL, headers=HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            return {"dilution_filings": 0}

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])

        return {"dilution_filings": len(hits)}

    except Exception as e:
        logger.debug(f"SEC dilution check failed for {ticker}: {e}")
        return {"dilution_filings": 0}


def _empty_insider_result(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "total_filings": 0,
        "recent_buys": 0,
        "recent_sells": 0,
        "net_direction": "neutral",
        "filings": [],
    }
