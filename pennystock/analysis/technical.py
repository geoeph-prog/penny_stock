"""
Technical analysis engine.

Computes multiple indicators and returns a composite technical score.
Unlike the original design that was calibrated to a single stock (OPTT),
this uses normalized indicator scores that don't assume specific targets.
"""

import numpy as np
import pandas as pd
from loguru import logger

from pennystock.config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLLINGER_PERIOD, BOLLINGER_STD, VOLUME_AVG_PERIOD,
    STOCHASTIC_PERIOD, STOCHRSI_PERIOD, TECH_WEIGHTS,
    RSI_SCORE_MAP, STOCHRSI_THRESHOLDS,
)


def compute_rsi(prices: pd.Series, period: int = None) -> pd.Series:
    """Compute Relative Strength Index."""
    period = period or RSI_PERIOD
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices: pd.Series) -> dict:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = prices.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = prices.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def compute_bollinger_bands(prices: pd.Series) -> dict:
    """Compute Bollinger Bands."""
    sma = prices.rolling(window=BOLLINGER_PERIOD).mean()
    std = prices.rolling(window=BOLLINGER_PERIOD).std()
    upper = sma + BOLLINGER_STD * std
    lower = sma - BOLLINGER_STD * std
    return {"upper": upper, "middle": sma, "lower": lower}


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """Compute Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(window=STOCHASTIC_PERIOD).min()
    highest_high = high.rolling(window=STOCHASTIC_PERIOD).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(window=3).mean()
    return {"k": k, "d": d}


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (volume * direction).cumsum()
    return obv


def compute_stochrsi(prices: pd.Series, rsi_period: int = None,
                     stoch_period: int = None) -> pd.Series:
    """
    Compute Stochastic RSI.

    StochRSI = (RSI - min(RSI, N)) / (max(RSI, N) - min(RSI, N)) * 100

    This is the stochastic oscillator applied to RSI values rather than price.
    It ranges 0-100 where < 20 is oversold and > 80 is overbought.

    RIME had StochRSI of 5.76 before its +325% run -- deeply oversold.
    """
    rsi_period = rsi_period or RSI_PERIOD
    stoch_period = stoch_period or STOCHRSI_PERIOD

    rsi = compute_rsi(prices, rsi_period)

    lowest_rsi = rsi.rolling(window=stoch_period).min()
    highest_rsi = rsi.rolling(window=stoch_period).max()

    rsi_range = highest_rsi - lowest_rsi
    stochrsi = ((rsi - lowest_rsi) / rsi_range.replace(0, np.nan)) * 100

    return stochrsi


def compute_volume_spike(volume: pd.Series, period: int = None) -> float:
    """Current volume relative to N-day average."""
    period = period or VOLUME_AVG_PERIOD
    if len(volume) < period + 1:
        return 1.0
    avg_vol = volume.iloc[-(period + 1):-1].mean()
    if avg_vol == 0:
        return 1.0
    return volume.iloc[-1] / avg_vol


def compute_price_trend(close: pd.Series, days: int = 20) -> float:
    """Price change percentage over N days."""
    if len(close) < days + 1:
        return 0.0
    old_price = close.iloc[-(days + 1)]
    new_price = close.iloc[-1]
    if old_price == 0:
        return 0.0
    return ((new_price - old_price) / old_price) * 100


def analyze(hist: pd.DataFrame) -> dict:
    """
    Run full technical analysis on a price history DataFrame.

    Args:
        hist: DataFrame with Open, High, Low, Close, Volume columns.

    Returns:
        Dict with individual indicator values, sub-scores, and composite score.
    """
    if hist is None or hist.empty or len(hist) < 30:
        return {"score": 0, "valid": False, "reason": "insufficient data"}

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]

    # ── Compute indicators ──────────────────────────────────────────
    rsi = compute_rsi(close)
    macd = compute_macd(close)
    bb = compute_bollinger_bands(close)
    stoch = compute_stochastic(high, low, close)
    stochrsi = compute_stochrsi(close)
    obv = compute_obv(close, volume)
    vol_spike = compute_volume_spike(volume)
    price_trend_20d = compute_price_trend(close, 20)
    price_trend_5d = compute_price_trend(close, 5)

    current_rsi = rsi.iloc[-1] if not rsi.empty else 50
    current_macd_hist = macd["histogram"].iloc[-1] if not macd["histogram"].empty else 0
    current_price = close.iloc[-1]
    current_bb_upper = bb["upper"].iloc[-1] if not bb["upper"].empty else current_price
    current_bb_lower = bb["lower"].iloc[-1] if not bb["lower"].empty else current_price
    current_bb_mid = bb["middle"].iloc[-1] if not bb["middle"].empty else current_price
    current_stoch_k = stoch["k"].iloc[-1] if not stoch["k"].empty else 50
    current_stochrsi = stochrsi.iloc[-1] if not stochrsi.empty else 50

    # ── Score each indicator (0-100) ────────────────────────────────

    # RSI Score: Best when 30-50 (oversold bounce zone, like RIME at 42-47)
    if np.isnan(current_rsi):
        rsi_score = 50
    elif 30 <= current_rsi <= 50:
        rsi_score = RSI_SCORE_MAP["30_50"]
    elif 50 < current_rsi <= 60:
        rsi_score = RSI_SCORE_MAP["50_60"]
    elif current_rsi < 30:
        rsi_score = RSI_SCORE_MAP["below_30"]
    elif 60 < current_rsi <= 70:
        rsi_score = RSI_SCORE_MAP["60_70"]
    else:
        rsi_score = RSI_SCORE_MAP["above_70"]

    # MACD Score: Positive histogram = bullish momentum
    if np.isnan(current_macd_hist):
        macd_score = 50
    else:
        # Normalize by price to make comparable across stocks
        macd_pct = (current_macd_hist / current_price * 100) if current_price > 0 else 0
        if macd_pct > 0:
            macd_score = min(100, 60 + macd_pct * 10)  # Bullish
        else:
            macd_score = max(0, 50 + macd_pct * 10)  # Bearish

        # Bonus: MACD line just crossed above signal (bullish crossover)
        if len(macd["histogram"]) >= 2:
            prev_hist = macd["histogram"].iloc[-2]
            if not np.isnan(prev_hist) and prev_hist < 0 and current_macd_hist > 0:
                macd_score = min(100, macd_score + 15)

    # Volume Spike Score: Higher is better for penny stocks, but extreme = pump risk
    if vol_spike >= 2.0 and vol_spike <= 10.0:
        vol_score = min(100, 50 + vol_spike * 8)  # Sweet spot
    elif vol_spike > 10.0:
        vol_score = max(60, 100 - (vol_spike - 10) * 2)  # Diminishing returns, possible pump
    elif vol_spike >= 1.0:
        vol_score = 30 + vol_spike * 20  # Below average volume
    else:
        vol_score = max(0, vol_spike * 30)  # Very low volume

    # OBV Trend Score: Rising OBV = accumulation
    if len(obv) >= 20:
        obv_recent = obv.iloc[-10:].values
        obv_older = obv.iloc[-20:-10].values
        obv_trend = obv_recent.mean() - obv_older.mean()
        if obv_trend > 0:
            obv_score = min(100, 60 + min(40, abs(obv_trend) / (volume.mean() + 1) * 100))
        else:
            obv_score = max(0, 40 - min(40, abs(obv_trend) / (volume.mean() + 1) * 100))
    else:
        obv_score = 50

    # Bollinger Band Score: Price near lower band = potential bounce
    bb_range = current_bb_upper - current_bb_lower
    if bb_range > 0 and not np.isnan(bb_range):
        bb_position = (current_price - current_bb_lower) / bb_range  # 0=lower, 1=upper
        if 0.2 <= bb_position <= 0.5:
            bb_score = 85  # Near lower band but not collapsing -- bounce potential
        elif 0.0 <= bb_position < 0.2:
            bb_score = 65  # At lower band -- could bounce or break down
        elif 0.5 < bb_position <= 0.8:
            bb_score = 60  # Middle to upper -- neutral to slightly hot
        else:
            bb_score = 35  # Above upper band -- overbought
    else:
        bb_score = 50

    # Stochastic Score: Similar to RSI -- sweet spot is 30-50 (oversold bounce)
    if np.isnan(current_stoch_k):
        stoch_score = 50
    elif 20 <= current_stoch_k <= 50:
        stoch_score = 80  # Oversold-to-neutral -- room to run
    elif 50 < current_stoch_k <= 70:
        stoch_score = 60  # Getting warm
    elif current_stoch_k > 70:
        stoch_score = max(20, 60 - (current_stoch_k - 70))  # Overbought
    else:
        stoch_score = 50  # Deep oversold

    # StochRSI Score: Lower = more oversold = better entry point
    # RIME had StochRSI of 5.76 before its +325% move -- deeply oversold.
    if np.isnan(current_stochrsi):
        stochrsi_score = 50
    else:
        stochrsi_score = 50  # default
        for threshold, score_val in STOCHRSI_THRESHOLDS:
            if current_stochrsi <= threshold:
                stochrsi_score = score_val
                break

    # Price Trend Score: Moderate uptrend is ideal
    # We want stocks starting to move, not ones that already exploded
    if 5 <= price_trend_20d <= 30:
        trend_score = 85  # Early momentum -- ideal entry
    elif 30 < price_trend_20d <= 60:
        trend_score = 70  # Good momentum but getting extended
    elif 0 <= price_trend_20d < 5:
        trend_score = 55  # Flat -- could go either way
    elif price_trend_20d > 60:
        trend_score = max(20, 60 - (price_trend_20d - 60) * 0.5)  # Chasing, late entry
    else:
        trend_score = max(0, 40 + price_trend_20d)  # Downtrend penalty

    # ── Composite technical score ───────────────────────────────────
    composite = (
        rsi_score * TECH_WEIGHTS["rsi"] +
        macd_score * TECH_WEIGHTS["macd"] +
        stochrsi_score * TECH_WEIGHTS["stochrsi"] +
        vol_score * TECH_WEIGHTS["volume_spike"] +
        obv_score * TECH_WEIGHTS["obv_trend"] +
        bb_score * TECH_WEIGHTS["bollinger"] +
        trend_score * TECH_WEIGHTS["price_trend"]
    )

    return {
        "score": round(composite, 1),
        "valid": True,
        # Raw indicator values
        "rsi": round(current_rsi, 1) if not np.isnan(current_rsi) else None,
        "stochrsi": round(current_stochrsi, 2) if not np.isnan(current_stochrsi) else None,
        "macd_histogram": round(current_macd_hist, 4) if not np.isnan(current_macd_hist) else None,
        "volume_spike": round(vol_spike, 2),
        "price_trend_20d": round(price_trend_20d, 1),
        "price_trend_5d": round(price_trend_5d, 1),
        "stochastic_k": round(current_stoch_k, 1) if not np.isnan(current_stoch_k) else None,
        "bb_position": round((current_price - current_bb_lower) / bb_range, 2) if bb_range > 0 and not np.isnan(bb_range) else None,
        "obv_trend_direction": "up" if obv_score > 55 else ("down" if obv_score < 45 else "flat"),
        # Sub-scores
        "sub_scores": {
            "rsi": round(rsi_score, 1),
            "macd": round(macd_score, 1),
            "stochrsi": round(stochrsi_score, 1),
            "volume_spike": round(vol_score, 1),
            "obv_trend": round(obv_score, 1),
            "bollinger": round(bb_score, 1),
            "price_trend": round(trend_score, 1),
        },
    }


def extract_features(hist: pd.DataFrame) -> dict:
    """
    Extract raw technical features for pattern learning / backtesting.
    Returns numerical features only (no scores), suitable for comparison.
    """
    if hist is None or hist.empty or len(hist) < 30:
        return {}

    close = hist["Close"]
    volume = hist["Volume"]

    rsi = compute_rsi(close)
    stochrsi = compute_stochrsi(close)
    macd = compute_macd(close)
    vol_spike = compute_volume_spike(volume)

    current_stochrsi = stochrsi.iloc[-1] if not stochrsi.empty else None
    if current_stochrsi is not None and np.isnan(current_stochrsi):
        current_stochrsi = None

    return {
        "rsi": float(rsi.iloc[-1]) if not rsi.empty and not np.isnan(rsi.iloc[-1]) else None,
        "stochrsi": float(current_stochrsi) if current_stochrsi is not None else None,
        "macd_histogram": float(macd["histogram"].iloc[-1]) if not macd["histogram"].empty else None,
        "volume_spike": float(vol_spike),
        "price_trend_5d": float(compute_price_trend(close, 5)),
        "price_trend_20d": float(compute_price_trend(close, 20)),
        "price_trend_60d": float(compute_price_trend(close, min(60, len(close) - 1))),
        "volatility_20d": float(close.pct_change().tail(20).std() * 100) if len(close) > 20 else None,
        "avg_volume": float(volume.tail(VOLUME_AVG_PERIOD).mean()),
        "price": float(close.iloc[-1]),
    }
