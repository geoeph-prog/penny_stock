"""
Fundamental analysis: financial health, insider activity, short interest, float.
"""

from loguru import logger

from pennystock.data.yahoo_client import get_stock_info, get_quarterly_financials


def analyze(ticker: str) -> dict:
    """
    Run fundamental analysis on a stock.

    Returns:
        {
            "score": float (0-100),
            "financial_health": dict,
            "short_squeeze_potential": dict,
            "insider_signal": dict,
            "revenue_momentum": dict,
        }
    """
    info = get_stock_info(ticker)

    health = _score_financial_health(info)
    squeeze = _score_short_squeeze(info)
    insider = _score_insider_activity(info)
    revenue = _score_revenue_momentum(ticker, info)

    # Composite: weighted blend of sub-scores
    score = (
        health["score"] * 0.25 +
        squeeze["score"] * 0.30 +
        insider["score"] * 0.20 +
        revenue["score"] * 0.25
    )

    return {
        "score": round(max(0, min(100, score)), 1),
        "financial_health": health,
        "short_squeeze_potential": squeeze,
        "insider_signal": insider,
        "revenue_momentum": revenue,
    }


def _score_financial_health(info: dict) -> dict:
    """Score based on cash, debt, and current ratio."""
    cash = info.get("total_cash", 0) or 0
    debt = info.get("total_debt", 0) or 0
    dte = info.get("debt_to_equity", 0) or 0
    current_ratio = info.get("current_ratio", 0) or 0

    score = 50  # Neutral baseline

    # Cash position (more = better survival odds)
    if cash > 10_000_000:
        score += 15
    elif cash > 1_000_000:
        score += 8
    elif cash > 0:
        score += 3

    # Debt-to-equity (lower is better for penny stocks)
    if dte == 0 or dte is None:
        score += 10  # No debt is good
    elif dte < 50:
        score += 8
    elif dte < 100:
        score += 3
    elif dte > 200:
        score -= 15
    elif dte > 100:
        score -= 5

    # Current ratio (>1 means can cover short-term obligations)
    if current_ratio > 2:
        score += 10
    elif current_ratio > 1:
        score += 5
    elif current_ratio > 0 and current_ratio < 1:
        score -= 10

    return {
        "score": max(0, min(100, score)),
        "cash": cash,
        "debt": debt,
        "debt_to_equity": dte,
        "current_ratio": current_ratio,
    }


def _score_short_squeeze(info: dict) -> dict:
    """
    Score short squeeze potential.
    High short interest + low float + increasing volume = squeeze potential.
    """
    short_pct = info.get("short_percent_of_float", 0) or 0
    short_ratio = info.get("short_ratio", 0) or 0  # Days to cover
    float_shares = info.get("float_shares", 0) or 0

    score = 50

    # Short interest as % of float (higher = more squeeze potential)
    if short_pct > 0.30:
        score += 25  # Very high -- squeeze candidate
    elif short_pct > 0.20:
        score += 18
    elif short_pct > 0.10:
        score += 10
    elif short_pct > 0.05:
        score += 5

    # Days to cover (higher = harder for shorts to exit)
    if short_ratio > 5:
        score += 15
    elif short_ratio > 3:
        score += 10
    elif short_ratio > 1:
        score += 5

    # Low float amplifies moves
    if 0 < float_shares < 10_000_000:
        score += 15  # Very low float
    elif 0 < float_shares < 30_000_000:
        score += 8
    elif 0 < float_shares < 50_000_000:
        score += 3

    return {
        "score": max(0, min(100, score)),
        "short_pct_float": short_pct,
        "short_ratio_days": short_ratio,
        "float_shares": float_shares,
    }


def _score_insider_activity(info: dict) -> dict:
    """
    Score based on insider and institutional ownership.
    High insider ownership = aligned interests.
    """
    insider_pct = info.get("insider_percent_held", 0) or 0
    inst_pct = info.get("institution_percent_held", 0) or 0

    score = 50

    # Insider ownership (higher = skin in the game)
    if insider_pct > 0.30:
        score += 15
    elif insider_pct > 0.10:
        score += 8
    elif insider_pct > 0.05:
        score += 3

    # Institutional ownership (some is good, too much = manipulation risk)
    if 0.10 <= inst_pct <= 0.50:
        score += 10  # Sweet spot
    elif inst_pct > 0.50:
        score += 3  # Heavy institutional -- could dump
    elif inst_pct > 0:
        score += 5

    return {
        "score": max(0, min(100, score)),
        "insider_pct": insider_pct,
        "institution_pct": inst_pct,
    }


def _score_revenue_momentum(ticker: str, info: dict) -> dict:
    """Score based on revenue growth trends."""
    rev_growth = info.get("revenue_growth", 0) or 0
    total_rev = info.get("total_revenue", 0) or 0

    score = 50

    # Revenue growth rate
    if rev_growth > 0.50:
        score += 25  # Explosive growth
    elif rev_growth > 0.20:
        score += 15
    elif rev_growth > 0:
        score += 8
    elif rev_growth < -0.20:
        score -= 15
    elif rev_growth < 0:
        score -= 5

    # Having any revenue at all is a positive for penny stocks
    if total_rev > 10_000_000:
        score += 10
    elif total_rev > 1_000_000:
        score += 5
    elif total_rev > 0:
        score += 2

    # Try to get quarterly trend
    qf = get_quarterly_financials(ticker)
    if qf is not None and not qf.empty:
        try:
            rev_row = None
            for label in ["Total Revenue", "TotalRevenue"]:
                if label in qf.index:
                    rev_row = qf.loc[label]
                    break

            if rev_row is not None and len(rev_row) >= 2:
                recent_vals = rev_row.dropna().values[:4]  # Most recent 4 quarters
                if len(recent_vals) >= 2:
                    # Check if revenue is accelerating
                    if recent_vals[0] > recent_vals[1]:
                        score += 5  # QoQ growth
                    if len(recent_vals) >= 3 and recent_vals[0] > recent_vals[2]:
                        score += 3  # Growing vs 2 quarters ago
        except Exception:
            pass

    return {
        "score": max(0, min(100, score)),
        "revenue_growth": rev_growth,
        "total_revenue": total_rev,
    }
