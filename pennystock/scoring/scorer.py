"""
Multi-factor composite scoring engine.

Combines technical, sentiment, fundamental, catalyst, and market context
scores into a single composite score. Weights are configurable and can
be optimized through backtesting.
"""

from loguru import logger

from pennystock.config import WEIGHTS
from pennystock.analysis import technical, sentiment, fundamental, catalyst, market_context
from pennystock.data.yahoo_client import get_price_history, get_stock_info


def score_stock(ticker: str, hist=None, sector: str = "", weights: dict = None) -> dict:
    """
    Run full multi-factor analysis and scoring on a single stock.

    Args:
        ticker: Stock ticker symbol.
        hist: Pre-loaded price history DataFrame (optional, fetched if None).
        sector: Stock's sector for market context analysis.
        weights: Custom scoring weights (uses config defaults if None).

    Returns:
        {
            "ticker": str,
            "final_score": float (0-100),
            "technical": dict,
            "sentiment": dict,
            "fundamental": dict,
            "catalyst": dict,
            "market_ctx": dict,
            "sub_scores": dict,
        }
    """
    weights = weights or WEIGHTS

    # ── Technical Analysis ──────────────────────────────────────────
    if hist is None:
        hist = get_price_history(ticker)

    tech = technical.analyze(hist)
    tech_score = tech.get("score", 0) if tech.get("valid") else 0

    # ── Sentiment Analysis ──────────────────────────────────────────
    try:
        sent = sentiment.analyze(ticker)
        sent_score = sent.get("score", 50)
    except Exception as e:
        logger.debug(f"Sentiment analysis failed for {ticker}: {e}")
        sent = {"score": 50}
        sent_score = 50

    # ── Fundamental Analysis ────────────────────────────────────────
    try:
        fund = fundamental.analyze(ticker)
        fund_score = fund.get("score", 50)
    except Exception as e:
        logger.debug(f"Fundamental analysis failed for {ticker}: {e}")
        fund = {"score": 50}
        fund_score = 50

    # ── Catalyst Detection ──────────────────────────────────────────
    try:
        cat = catalyst.analyze(ticker)
        cat_score = cat.get("score", 50)
    except Exception as e:
        logger.debug(f"Catalyst analysis failed for {ticker}: {e}")
        cat = {"score": 50}
        cat_score = 50

    # ── Market Context ──────────────────────────────────────────────
    try:
        mkt = market_context.analyze(sector)
        mkt_score = mkt.get("score", 50)
    except Exception as e:
        logger.debug(f"Market context analysis failed: {e}")
        mkt = {"score": 50}
        mkt_score = 50

    # ── Short Squeeze (from fundamental analysis) ───────────────────
    squeeze_score = fund.get("short_squeeze_potential", {}).get("score", 50)

    # ── Composite Score ─────────────────────────────────────────────
    final = (
        tech_score * weights.get("technical", 0.25) +
        sent_score * weights.get("sentiment", 0.20) +
        fund_score * weights.get("fundamental", 0.15) +
        cat_score * weights.get("catalyst", 0.15) +
        mkt_score * weights.get("market_ctx", 0.10) +
        squeeze_score * weights.get("short_squeeze", 0.15)
    )

    return {
        "ticker": ticker,
        "final_score": round(max(0, min(100, final)), 1),
        "technical": tech,
        "sentiment": sent,
        "fundamental": fund,
        "catalyst": cat,
        "market_ctx": mkt,
        "sub_scores": {
            "technical": round(tech_score, 1),
            "sentiment": round(sent_score, 1),
            "fundamental": round(fund_score, 1),
            "catalyst": round(cat_score, 1),
            "market_ctx": round(mkt_score, 1),
            "short_squeeze": round(squeeze_score, 1),
        },
    }


def score_stock_technical_only(ticker: str, hist=None) -> dict:
    """
    Fast technical-only scoring for Stage 1 filtering.
    Skips sentiment, fundamentals, catalysts, and market context.
    """
    if hist is None:
        hist = get_price_history(ticker)

    tech = technical.analyze(hist)

    return {
        "ticker": ticker,
        "technical_score": tech.get("score", 0),
        "valid": tech.get("valid", False),
        "indicators": tech,
    }
