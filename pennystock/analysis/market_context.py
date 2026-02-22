"""
Market context analysis: VIX, sector performance, and overall market conditions.

Provides a "should we be buying penny stocks right now?" signal.
"""

import numpy as np
import yfinance as yf
from loguru import logger

from pennystock.config import VIX_TICKER, SECTOR_ETFS, FEAR_GREED_FAVORABLE_RANGE
from pennystock.data.cache import cache_get, cache_set


def analyze(sector: str = "") -> dict:
    """
    Assess overall market conditions for penny stock buying.

    Args:
        sector: Optional sector name to check sector-specific conditions.

    Returns:
        {
            "score": float (0-100),  # Higher = more favorable for buying
            "vix": dict,
            "market_trend": dict,
            "sector_performance": dict,
        }
    """
    vix = _analyze_vix()
    market = _analyze_market_trend()
    sector_perf = _analyze_sector(sector) if sector else {"score": 50}

    # Composite: VIX conditions + market trend + sector
    score = (
        vix["score"] * 0.35 +
        market["score"] * 0.40 +
        sector_perf["score"] * 0.25
    )

    return {
        "score": round(max(0, min(100, score)), 1),
        "vix": vix,
        "market_trend": market,
        "sector_performance": sector_perf,
    }


def _analyze_vix() -> dict:
    """
    Analyze VIX (fear gauge).
    Low VIX = complacent market (OK to buy)
    Moderate VIX = some fear (good buying opportunity)
    High VIX = panic (risky for penny stocks)
    """
    cache_key = "vix_analysis"
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        vix = yf.Ticker(VIX_TICKER)
        hist = vix.history(period="1mo")

        if hist.empty:
            return {"score": 50, "level": None, "trend": "unknown"}

        current_vix = hist["Close"].iloc[-1]
        vix_5d_ago = hist["Close"].iloc[-5] if len(hist) >= 5 else current_vix
        vix_trend = "rising" if current_vix > vix_5d_ago * 1.05 else (
            "falling" if current_vix < vix_5d_ago * 0.95 else "stable"
        )

        # Score: lower VIX is generally better for risk assets
        if current_vix < 15:
            score = 75  # Low fear, calm markets
        elif current_vix < 20:
            score = 70  # Normal conditions
        elif current_vix < 25:
            score = 55  # Elevated uncertainty
        elif current_vix < 30:
            score = 40  # High fear -- be cautious
        else:
            score = 20  # Panic mode -- avoid penny stocks

        # Falling VIX is a positive signal
        if vix_trend == "falling":
            score += 10
        elif vix_trend == "rising":
            score -= 10

        result = {
            "score": max(0, min(100, score)),
            "level": round(current_vix, 1),
            "trend": vix_trend,
        }
        cache_set(cache_key, result)
        return result

    except Exception as e:
        logger.warning(f"VIX analysis failed: {e}")
        return {"score": 50, "level": None, "trend": "unknown"}


def _analyze_market_trend() -> dict:
    """
    Analyze broad market trend using SPY (S&P 500 ETF).
    Bull market = more favorable for penny stocks.
    """
    cache_key = "market_trend_analysis"
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="3mo")

        if hist.empty:
            return {"score": 50, "trend": "unknown"}

        close = hist["Close"]

        # Calculate moving averages
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
        current = close.iloc[-1]

        # Price relative to MAs
        above_20 = current > sma_20
        above_50 = current > sma_50

        # Recent performance
        pct_change_5d = ((current - close.iloc[-5]) / close.iloc[-5] * 100) if len(close) >= 5 else 0
        pct_change_20d = ((current - close.iloc[-20]) / close.iloc[-20] * 100) if len(close) >= 20 else 0

        score = 50

        if above_20 and above_50:
            score += 20  # Strong uptrend
        elif above_20:
            score += 10  # Above short-term MA
        elif not above_20 and not above_50:
            score -= 15  # Below both MAs -- bearish

        if pct_change_5d > 1:
            score += 10
        elif pct_change_5d < -2:
            score -= 10

        if pct_change_20d > 3:
            score += 5
        elif pct_change_20d < -5:
            score -= 10

        trend = "bullish" if score > 60 else ("bearish" if score < 40 else "neutral")

        result = {
            "score": max(0, min(100, score)),
            "trend": trend,
            "spy_price": round(float(current), 2),
            "above_20sma": bool(above_20),
            "above_50sma": bool(above_50),
            "change_5d_pct": round(float(pct_change_5d), 1),
            "change_20d_pct": round(float(pct_change_20d), 1),
        }
        cache_set(cache_key, result)
        return result

    except Exception as e:
        logger.warning(f"Market trend analysis failed: {e}")
        return {"score": 50, "trend": "unknown"}


def _analyze_sector(sector: str) -> dict:
    """Analyze sector-specific momentum."""
    etf = SECTOR_ETFS.get(sector)
    if not etf:
        return {"score": 50, "sector": sector, "etf": None}

    try:
        t = yf.Ticker(etf)
        hist = t.history(period="1mo")

        if hist.empty:
            return {"score": 50, "sector": sector, "etf": etf}

        close = hist["Close"]
        pct_change = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100)

        if pct_change > 5:
            score = 80  # Hot sector
        elif pct_change > 2:
            score = 65
        elif pct_change > 0:
            score = 55
        elif pct_change > -3:
            score = 40
        else:
            score = 25  # Weak sector

        return {
            "score": max(0, min(100, score)),
            "sector": sector,
            "etf": etf,
            "change_1mo_pct": round(pct_change, 1),
        }

    except Exception as e:
        logger.debug(f"Sector analysis failed for {sector}: {e}")
        return {"score": 50, "sector": sector, "etf": etf}
