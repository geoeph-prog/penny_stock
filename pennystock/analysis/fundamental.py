"""
Fundamental analysis: financial health, CEO/insider activity, SEC filings, sector context.
"""

from loguru import logger

from pennystock.data.yahoo_client import get_stock_info, get_quarterly_financials
from pennystock.data.sec_client import get_insider_transactions, get_recent_filings


def analyze(ticker: str) -> dict:
    """
    Full fundamental analysis for a single stock.

    Returns dict with sub-scores and raw data for:
    - financial_health: cash, debt, current ratio
    - insider_activity: insider/institutional ownership + SEC Form 4 data
    - sec_filings: recent 8-K material events, filing activity
    - revenue_momentum: revenue growth trends
    """
    info = get_stock_info(ticker)

    health = _score_financial_health(info)
    insider = _score_insider_activity(ticker, info)
    sec = _score_sec_filings(ticker)
    revenue = _score_revenue_momentum(ticker, info)

    score = (
        health["score"] * 0.25 +
        insider["score"] * 0.30 +
        sec["score"] * 0.15 +
        revenue["score"] * 0.30
    )

    return {
        "score": round(max(0, min(100, score)), 1),
        "financial_health": health,
        "insider_activity": insider,
        "sec_filings": sec,
        "revenue_momentum": revenue,
    }


def extract_features(ticker: str) -> dict:
    """
    Extract numerical fundamental features for algorithm learning.
    Returns flat dict of feature_name: value.
    """
    info = get_stock_info(ticker)
    insider = get_insider_transactions(ticker, days_back=90)
    sec = get_recent_filings(ticker, days_back=90)

    return {
        "total_cash": info.get("total_cash", 0) or 0,
        "total_debt": info.get("total_debt", 0) or 0,
        "debt_to_equity": info.get("debt_to_equity", 0) or 0,
        "current_ratio": info.get("current_ratio", 0) or 0,
        "revenue_growth": info.get("revenue_growth", 0) or 0,
        "profit_margins": info.get("profit_margins", 0) or 0,
        "insider_pct": info.get("insider_percent_held", 0) or 0,
        "institution_pct": info.get("institution_percent_held", 0) or 0,
        "short_pct_float": info.get("short_percent_of_float", 0) or 0,
        "short_ratio": info.get("short_ratio", 0) or 0,
        "float_shares": info.get("float_shares", 0) or 0,
        "insider_buys_90d": insider.get("recent_buys", 0),
        "insider_sells_90d": insider.get("recent_sells", 0),
        "insider_net_buys": insider.get("recent_buys", 0) - insider.get("recent_sells", 0),
        "sec_filings_90d": sec.get("total_filings", 0),
        "sec_8k_count": sec.get("filing_types", {}).get("8-K", 0),
    }


def _score_financial_health(info: dict) -> dict:
    cash = info.get("total_cash", 0) or 0
    debt = info.get("total_debt", 0) or 0
    dte = info.get("debt_to_equity", 0) or 0
    current_ratio = info.get("current_ratio", 0) or 0

    score = 50

    if cash > 10_000_000:
        score += 15
    elif cash > 1_000_000:
        score += 8
    elif cash > 0:
        score += 3

    if dte == 0 or dte is None:
        score += 10
    elif dte < 50:
        score += 8
    elif dte < 100:
        score += 3
    elif dte > 200:
        score -= 15
    elif dte > 100:
        score -= 5

    if current_ratio > 2:
        score += 10
    elif current_ratio > 1:
        score += 5
    elif 0 < current_ratio < 1:
        score -= 10

    return {
        "score": max(0, min(100, score)),
        "cash": cash, "debt": debt,
        "debt_to_equity": dte, "current_ratio": current_ratio,
    }


def _score_insider_activity(ticker: str, info: dict) -> dict:
    insider_pct = info.get("insider_percent_held", 0) or 0
    inst_pct = info.get("institution_percent_held", 0) or 0
    insider_data = get_insider_transactions(ticker, days_back=90)

    score = 50

    if insider_pct > 0.30:
        score += 12
    elif insider_pct > 0.10:
        score += 7
    elif insider_pct > 0.05:
        score += 3

    if 0.10 <= inst_pct <= 0.50:
        score += 8
    elif inst_pct > 0.50:
        score += 2

    net_buys = insider_data["recent_buys"] - insider_data["recent_sells"]
    if net_buys > 3:
        score += 20
    elif net_buys > 0:
        score += 12
    elif net_buys < -3:
        score -= 15
    elif net_buys < 0:
        score -= 8

    return {
        "score": max(0, min(100, score)),
        "insider_pct": insider_pct, "institution_pct": inst_pct,
        "insider_buys_90d": insider_data["recent_buys"],
        "insider_sells_90d": insider_data["recent_sells"],
        "net_direction": insider_data["net_direction"],
    }


def _score_sec_filings(ticker: str) -> dict:
    filings = get_recent_filings(ticker, days_back=90)
    score = 50
    total = filings["total_filings"]
    type_counts = filings.get("filing_types", {})

    if total > 5:
        score += 10
    elif total > 2:
        score += 5
    elif total == 0:
        score -= 5

    eight_k = type_counts.get("8-K", 0)
    if eight_k > 0:
        score += min(10, eight_k * 3)

    if type_counts.get("10-Q", 0) > 0 or type_counts.get("10-K", 0) > 0:
        score += 5

    return {
        "score": max(0, min(100, score)),
        "total_filings": total, "filing_types": type_counts,
    }


def _score_revenue_momentum(ticker: str, info: dict) -> dict:
    rev_growth = info.get("revenue_growth", 0) or 0
    total_rev = info.get("total_revenue", 0) or 0
    score = 50

    if rev_growth > 0.50:
        score += 25
    elif rev_growth > 0.20:
        score += 15
    elif rev_growth > 0:
        score += 8
    elif rev_growth < -0.20:
        score -= 15
    elif rev_growth < 0:
        score -= 5

    if total_rev > 10_000_000:
        score += 10
    elif total_rev > 1_000_000:
        score += 5
    elif total_rev > 0:
        score += 2

    qf = get_quarterly_financials(ticker)
    if qf is not None and not qf.empty:
        try:
            rev_row = None
            for label in ["Total Revenue", "TotalRevenue"]:
                if label in qf.index:
                    rev_row = qf.loc[label]
                    break
            if rev_row is not None and len(rev_row) >= 2:
                recent = rev_row.dropna().values[:4]
                if len(recent) >= 2 and recent[0] > recent[1]:
                    score += 5
                if len(recent) >= 3 and recent[0] > recent[2]:
                    score += 3
        except Exception:
            pass

    return {
        "score": max(0, min(100, score)),
        "revenue_growth": rev_growth, "total_revenue": total_rev,
    }
