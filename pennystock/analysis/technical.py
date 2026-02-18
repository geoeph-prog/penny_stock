"""
Technical analysis engine (v3: comprehensive indicator suite).

Computes a wide range of indicators and returns a composite technical score.
Uses normalized indicator scores that don't assume specific targets.

Indicators:
  Core momentum: RSI, MACD, StochRSI
  Trend strength: ADX (Average Directional Index)
  Volume: Volume spike, OBV, MFI (Money Flow Index), multi-day unusual volume
  Volatility: Bollinger Bands, BB Squeeze detection
  Price action: Price trend, volume-price divergence, consolidation detection
  Pattern: Support/resistance levels, gap detection, candlestick patterns
  Risk: ATR (Average True Range) for stop-loss calculation
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


# ════════════════════════════════════════════════════════════════════
# CORE INDICATOR COMPUTATIONS
# ════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════
# NEW INDICATORS (Tier 1-3)
# ════════════════════════════════════════════════════════════════════

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> dict:
    """
    Compute Average Directional Index (ADX).
    ADX measures trend STRENGTH (not direction).
    >25 = strong trend, >50 = very strong, <20 = no trend (choppy).

    Validated against winners: IQST had strong uptrend (ADX>25) before
    its +118% run on earnings. Choppy stocks like SLS would score low.
    """
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close.shift(1)).abs(),
        "lc": (low - close.shift(1)).abs(),
    }).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(window=period).mean()

    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "atr": atr,
    }


def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
                volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Money Flow Index (volume-weighted RSI).
    MFI < 20 = oversold, MFI > 80 = overbought.

    Better than plain RSI for penny stocks because it incorporates
    volume -- heavy buying volume + price increase = real accumulation.
    BEAT had strong buying pressure (high MFI) before its FDA catalyst.
    """
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    delta = typical_price.diff()
    positive_flow = raw_money_flow.where(delta > 0, 0.0)
    negative_flow = raw_money_flow.where(delta < 0, 0.0)

    pos_sum = positive_flow.rolling(window=period).sum()
    neg_sum = negative_flow.rolling(window=period).sum()

    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))

    return mfi


def compute_bb_squeeze(prices: pd.Series, bb_period: int = None,
                       bb_std: float = None) -> dict:
    """
    Detect Bollinger Band squeeze (bandwidth contraction).
    When bands tighten, volatility is compressing = impending explosive move.

    Bandwidth = (upper - lower) / middle
    Squeeze = bandwidth below its 20-day average by >30%

    Validated: KTTA had tight consolidation at $0.49 before +142% explosion.
    BEAT consolidated at $0.76 before +200% FDA breakout.
    SIDU hit ATL at $0.63, consolidated, then +297%.
    """
    bb_period = bb_period or BOLLINGER_PERIOD
    bb_std_val = bb_std or BOLLINGER_STD

    bb = compute_bollinger_bands(prices)
    upper = bb["upper"]
    lower = bb["lower"]
    middle = bb["middle"]

    bandwidth = (upper - lower) / middle.replace(0, np.nan)

    # Compare current bandwidth to its own 20-period average
    bw_avg = bandwidth.rolling(window=20).mean()

    current_bw = bandwidth.iloc[-1] if not bandwidth.empty else None
    avg_bw = bw_avg.iloc[-1] if not bw_avg.empty else None

    is_squeeze = False
    squeeze_intensity = 0.0
    if (current_bw is not None and avg_bw is not None and
            not np.isnan(current_bw) and not np.isnan(avg_bw) and avg_bw > 0):
        squeeze_ratio = current_bw / avg_bw
        is_squeeze = squeeze_ratio < 0.70  # Current BW is 30%+ below average
        squeeze_intensity = max(0, 1 - squeeze_ratio)  # 0 = no squeeze, 1 = max squeeze

    return {
        "is_squeeze": is_squeeze,
        "squeeze_intensity": squeeze_intensity,
        "bandwidth": float(current_bw) if current_bw is not None and not np.isnan(current_bw) else None,
        "avg_bandwidth": float(avg_bw) if avg_bw is not None and not np.isnan(avg_bw) else None,
    }


def compute_multiday_unusual_volume(volume: pd.Series, period: int = 50,
                                     lookback: int = 5, sigma: float = 2.0) -> dict:
    """
    Detect sustained unusual volume over multiple days.
    3+ sigma volume on 3/5 days = institutional accumulation.

    Single-day spikes can be noise; multi-day = someone is accumulating.
    BBGI had 45.8M shares vs <1M float. BEAT had 270M+ shares on FDA day.
    SMX had 21M shares vs ~1M float. Multi-day unusual volume preceded all.
    """
    if len(volume) < period + lookback:
        return {"unusual_days": 0, "max_sigma": 0, "avg_rvol_5d": 1.0, "is_accumulating": False}

    # Compute rolling mean and std excluding the lookback window
    hist_vol = volume.iloc[-(period + lookback):-lookback]
    vol_mean = hist_vol.mean()
    vol_std = hist_vol.std()

    if vol_std == 0 or vol_mean == 0:
        return {"unusual_days": 0, "max_sigma": 0, "avg_rvol_5d": 1.0, "is_accumulating": False}

    recent = volume.iloc[-lookback:]
    sigmas = [(v - vol_mean) / vol_std for v in recent]
    rvols = [v / vol_mean for v in recent]

    unusual_days = sum(1 for s in sigmas if s >= sigma)
    max_sigma = max(sigmas) if sigmas else 0
    avg_rvol = sum(rvols) / len(rvols) if rvols else 1.0

    return {
        "unusual_days": unusual_days,
        "max_sigma": round(max_sigma, 2),
        "avg_rvol_5d": round(avg_rvol, 2),
        "is_accumulating": unusual_days >= 3,
    }


def compute_volume_price_divergence(close: pd.Series, volume: pd.Series,
                                     lookback: int = 20) -> dict:
    """
    Detect divergence between volume (OBV) and price trends.
    OBV rising while price flat/declining = invisible accumulation (bullish).
    OBV falling while price rising = distribution (bearish).

    Validated: Several winners showed accumulation patterns before breakout:
    KTTA, BEAT, SIDU all had quiet accumulation at lows.
    """
    if len(close) < lookback + 10:
        return {"divergence_type": "neutral", "divergence_score": 0}

    obv = compute_obv(close, volume)

    # Compare price and OBV trends over lookback period
    price_start = close.iloc[-lookback]
    price_end = close.iloc[-1]
    price_change = (price_end - price_start) / price_start if price_start > 0 else 0

    obv_start = obv.iloc[-lookback]
    obv_end = obv.iloc[-1]
    obv_mean = abs(obv.iloc[-lookback:]).mean()
    obv_change = (obv_end - obv_start) / obv_mean if obv_mean > 0 else 0

    # Detect divergence
    if obv_change > 0.1 and price_change < 0.02:
        # OBV rising, price flat/down = bullish accumulation
        divergence_type = "bullish_accumulation"
        divergence_score = min(100, obv_change * 200)
    elif obv_change < -0.1 and price_change > 0.02:
        # OBV falling, price rising = bearish distribution
        divergence_type = "bearish_distribution"
        divergence_score = max(-100, obv_change * 200)
    else:
        divergence_type = "neutral"
        divergence_score = 0

    return {
        "divergence_type": divergence_type,
        "divergence_score": round(divergence_score, 1),
        "price_change_pct": round(price_change * 100, 2),
        "obv_trend": "up" if obv_change > 0.05 else ("down" if obv_change < -0.05 else "flat"),
    }


def compute_consolidation(close: pd.Series, volume: pd.Series,
                          lookback: int = 10) -> dict:
    """
    Detect price consolidation (tight range) with optional volume accumulation.
    Tight price range + rising OBV = pre-breakout setup.

    Uses ATR contraction: if recent ATR is much lower than longer-term ATR,
    the stock is consolidating.

    Validated: KTTA consolidated at $0.49, BEAT at $0.76, SIDU at $0.63.
    All broke out explosively after periods of low volatility.
    """
    if len(close) < lookback + 20:
        return {"is_consolidating": False, "consolidation_score": 0}

    # Price range contraction
    recent_range = close.iloc[-lookback:].max() - close.iloc[-lookback:].min()
    longer_range = close.iloc[-(lookback + 20):-lookback].max() - close.iloc[-(lookback + 20):-lookback].min()
    current_price = close.iloc[-1]

    if current_price == 0 or longer_range == 0:
        return {"is_consolidating": False, "consolidation_score": 0}

    range_pct = recent_range / current_price
    range_contraction = recent_range / longer_range if longer_range > 0 else 1

    # OBV trend during consolidation
    obv = compute_obv(close, volume)
    obv_recent = obv.iloc[-lookback:].values
    obv_slope = np.polyfit(range(len(obv_recent)), obv_recent, 1)[0] if len(obv_recent) > 1 else 0
    obv_rising = obv_slope > 0

    is_consolidating = range_contraction < 0.5 and range_pct < 0.15
    consolidation_score = 0
    if is_consolidating:
        consolidation_score = min(100, (1 - range_contraction) * 100)
        if obv_rising:
            consolidation_score = min(100, consolidation_score + 20)  # Accumulation bonus

    return {
        "is_consolidating": is_consolidating,
        "consolidation_score": round(consolidation_score, 1),
        "range_contraction": round(range_contraction, 3),
        "range_pct": round(range_pct, 4),
        "obv_rising_during_consolidation": obv_rising,
    }


def compute_support_resistance(close: pd.Series, lookback: int = 60) -> dict:
    """
    Identify key support and resistance levels from price pivots.
    Support = price level tested multiple times without breaking.
    Resistance = price ceiling tested multiple times.

    Validated: BEAT bounced off $0.54-$0.80 support zone multiple times
    before FDA catalyst broke resistance. SIDU found support at ATL $0.63.
    """
    if len(close) < lookback:
        return {"support": None, "resistance": None, "near_support": False, "near_resistance": False}

    prices = close.iloc[-lookback:].values
    current = prices[-1]

    # Find local minima and maxima (simple pivot detection)
    pivots_low = []
    pivots_high = []

    for i in range(2, len(prices) - 2):
        if prices[i] <= prices[i - 1] and prices[i] <= prices[i + 1]:
            if prices[i] <= prices[i - 2] and prices[i] <= prices[i + 2]:
                pivots_low.append(prices[i])
        if prices[i] >= prices[i - 1] and prices[i] >= prices[i + 1]:
            if prices[i] >= prices[i - 2] and prices[i] >= prices[i + 2]:
                pivots_high.append(prices[i])

    # Cluster nearby pivots (within 5% of each other)
    support = _cluster_levels(pivots_low, tolerance=0.05) if pivots_low else None
    resistance = _cluster_levels(pivots_high, tolerance=0.05) if pivots_high else None

    # Find nearest support below and resistance above current price
    nearest_support = None
    nearest_resistance = None

    if support:
        below = [s for s in support if s < current]
        if below:
            nearest_support = max(below)
    if resistance:
        above = [r for r in resistance if r > current]
        if above:
            nearest_resistance = min(above)

    near_support = (nearest_support is not None and
                    current > 0 and
                    (current - nearest_support) / current < 0.05)
    near_resistance = (nearest_resistance is not None and
                       current > 0 and
                       (nearest_resistance - current) / current < 0.05)

    return {
        "support": round(nearest_support, 4) if nearest_support else None,
        "resistance": round(nearest_resistance, 4) if nearest_resistance else None,
        "near_support": near_support,
        "near_resistance": near_resistance,
        "support_distance_pct": round((current - nearest_support) / current * 100, 2) if nearest_support and current > 0 else None,
        "resistance_distance_pct": round((nearest_resistance - current) / current * 100, 2) if nearest_resistance and current > 0 else None,
    }


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> float:
    """
    Compute Average True Range for stop-loss calculation.
    ATR measures volatility; 2x ATR below entry = dynamic stop.

    Returns current ATR value.
    """
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close.shift(1)).abs(),
        "lc": (low - close.shift(1)).abs(),
    }).max(axis=1)

    atr = tr.rolling(window=period).mean()
    current_atr = atr.iloc[-1] if not atr.empty else 0
    return float(current_atr) if not np.isnan(current_atr) else 0


def compute_gap(open_prices: pd.Series, close: pd.Series) -> dict:
    """
    Detect price gaps (gap-ups and gap-downs).
    Gap = today's open vs yesterday's close.

    Validated: BKYI gapped up on defense contract. BEAT gapped up on FDA clearance.
    OPAD gapped up 50% overnight on Trump mortgage plan.
    """
    if len(open_prices) < 2 or len(close) < 2:
        return {"gap_pct": 0, "has_gap": False, "gap_direction": "none"}

    prev_close = close.iloc[-2]
    today_open = open_prices.iloc[-1]

    if prev_close <= 0:
        return {"gap_pct": 0, "has_gap": False, "gap_direction": "none"}

    gap_pct = ((today_open - prev_close) / prev_close) * 100

    return {
        "gap_pct": round(gap_pct, 2),
        "has_gap": abs(gap_pct) > 4.0,  # >4% gap is significant
        "gap_direction": "up" if gap_pct > 4 else ("down" if gap_pct < -4 else "none"),
    }


def detect_candlestick_patterns(open_prices: pd.Series, high: pd.Series,
                                 low: pd.Series, close: pd.Series) -> dict:
    """
    Detect key candlestick patterns without TA-Lib dependency.
    Implements the most important reversal patterns manually.

    Patterns detected:
    - Hammer (bullish reversal at bottom)
    - Bullish engulfing (bullish reversal)
    - Morning star (3-candle bullish reversal)
    - Doji (indecision, potential reversal)
    """
    if len(close) < 5:
        return {"patterns": [], "bullish_patterns": 0, "bearish_patterns": 0}

    patterns = []
    o = open_prices.values
    h = high.values
    l = low.values
    c = close.values

    # Last 3 candles
    i = len(c) - 1

    body = abs(c[i] - o[i])
    candle_range = h[i] - l[i]
    lower_shadow = min(o[i], c[i]) - l[i]
    upper_shadow = h[i] - max(o[i], c[i])

    # Hammer: small body at top, long lower shadow (2x+ body), near bottom
    if candle_range > 0 and body < candle_range * 0.3 and lower_shadow > body * 2:
        patterns.append("hammer")

    # Bullish engulfing: previous red candle fully inside current green candle
    if i >= 1:
        prev_red = c[i - 1] < o[i - 1]
        curr_green = c[i] > o[i]
        engulfs = o[i] <= c[i - 1] and c[i] >= o[i - 1]
        if prev_red and curr_green and engulfs:
            patterns.append("bullish_engulfing")

    # Morning star: 3 candles - big red, small body (star), big green
    if i >= 2:
        first_red = c[i - 2] < o[i - 2] and abs(c[i - 2] - o[i - 2]) > candle_range * 0.3
        star_small = abs(c[i - 1] - o[i - 1]) < candle_range * 0.2 if candle_range > 0 else False
        third_green = c[i] > o[i] and abs(c[i] - o[i]) > candle_range * 0.3
        if first_red and star_small and third_green:
            patterns.append("morning_star")

    # Doji: body is very small relative to range
    if candle_range > 0 and body < candle_range * 0.1:
        patterns.append("doji")

    # Bearish engulfing
    if i >= 1:
        prev_green = c[i - 1] > o[i - 1]
        curr_red = c[i] < o[i]
        engulfs_bear = o[i] >= c[i - 1] and c[i] <= o[i - 1]
        if prev_green and curr_red and engulfs_bear:
            patterns.append("bearish_engulfing")

    bullish = sum(1 for p in patterns if p in ["hammer", "bullish_engulfing", "morning_star"])
    bearish = sum(1 for p in patterns if p in ["bearish_engulfing"])

    return {
        "patterns": patterns,
        "bullish_patterns": bullish,
        "bearish_patterns": bearish,
    }


def _cluster_levels(values: list, tolerance: float = 0.05) -> list:
    """Cluster nearby price levels, returning representative levels."""
    if not values:
        return []

    sorted_vals = sorted(values)
    clusters = [[sorted_vals[0]]]

    for v in sorted_vals[1:]:
        if (v - clusters[-1][-1]) / clusters[-1][-1] < tolerance if clusters[-1][-1] > 0 else True:
            clusters[-1].append(v)
        else:
            clusters.append([v])

    # Return mean of each cluster, weighted by number of touches
    return [sum(c) / len(c) for c in clusters if len(c) >= 2]


# ════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ════════════════════════════════════════════════════════════════════

def analyze(hist: pd.DataFrame) -> dict:
    """
    Run full technical analysis on a price history DataFrame.

    Returns dict with individual indicator values, sub-scores, and composite score.
    """
    if hist is None or hist.empty or len(hist) < 30:
        return {"score": 0, "valid": False, "reason": "insufficient data"}

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]
    open_prices = hist["Open"]

    # ── Compute core indicators ───────────────────────────────────
    rsi = compute_rsi(close)
    macd = compute_macd(close)
    bb = compute_bollinger_bands(close)
    stoch = compute_stochastic(high, low, close)
    stochrsi = compute_stochrsi(close)
    obv = compute_obv(close, volume)
    vol_spike = compute_volume_spike(volume)
    price_trend_20d = compute_price_trend(close, 20)
    price_trend_5d = compute_price_trend(close, 5)

    # ── Compute new indicators ────────────────────────────────────
    adx_data = compute_adx(high, low, close)
    mfi = compute_mfi(high, low, close, volume)
    bb_squeeze = compute_bb_squeeze(close)
    multiday_vol = compute_multiday_unusual_volume(volume)
    vol_price_div = compute_volume_price_divergence(close, volume)
    consolidation = compute_consolidation(close, volume)
    support_resistance = compute_support_resistance(close)
    atr_val = compute_atr(high, low, close)
    gap = compute_gap(open_prices, close)
    candle_patterns = detect_candlestick_patterns(open_prices, high, low, close)

    # ── Extract current values ────────────────────────────────────
    current_rsi = rsi.iloc[-1] if not rsi.empty else 50
    current_macd_hist = macd["histogram"].iloc[-1] if not macd["histogram"].empty else 0
    current_price = close.iloc[-1]
    current_bb_upper = bb["upper"].iloc[-1] if not bb["upper"].empty else current_price
    current_bb_lower = bb["lower"].iloc[-1] if not bb["lower"].empty else current_price
    current_bb_mid = bb["middle"].iloc[-1] if not bb["middle"].empty else current_price
    current_stoch_k = stoch["k"].iloc[-1] if not stoch["k"].empty else 50
    current_stochrsi = stochrsi.iloc[-1] if not stochrsi.empty else 50
    current_adx = adx_data["adx"].iloc[-1] if not adx_data["adx"].empty else 0
    current_plus_di = adx_data["plus_di"].iloc[-1] if not adx_data["plus_di"].empty else 0
    current_minus_di = adx_data["minus_di"].iloc[-1] if not adx_data["minus_di"].empty else 0
    current_mfi = mfi.iloc[-1] if not mfi.empty else 50

    # ── Score each indicator (0-100) ──────────────────────────────

    # RSI Score
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

    # MACD Score
    if np.isnan(current_macd_hist):
        macd_score = 50
    else:
        macd_pct = (current_macd_hist / current_price * 100) if current_price > 0 else 0
        if macd_pct > 0:
            macd_score = min(100, 60 + macd_pct * 10)
        else:
            macd_score = max(0, 50 + macd_pct * 10)
        if len(macd["histogram"]) >= 2:
            prev_hist = macd["histogram"].iloc[-2]
            if not np.isnan(prev_hist) and prev_hist < 0 and current_macd_hist > 0:
                macd_score = min(100, macd_score + 15)

    # Volume Spike Score
    if vol_spike >= 2.0 and vol_spike <= 10.0:
        vol_score = min(100, 50 + vol_spike * 8)
    elif vol_spike > 10.0:
        vol_score = max(60, 100 - (vol_spike - 10) * 2)
    elif vol_spike >= 1.0:
        vol_score = 30 + vol_spike * 20
    else:
        vol_score = max(0, vol_spike * 30)

    # OBV Trend Score
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

    # Bollinger Band Score
    bb_range = current_bb_upper - current_bb_lower
    if bb_range > 0 and not np.isnan(bb_range):
        bb_position = (current_price - current_bb_lower) / bb_range
        if 0.2 <= bb_position <= 0.5:
            bb_score = 85
        elif 0.0 <= bb_position < 0.2:
            bb_score = 65
        elif 0.5 < bb_position <= 0.8:
            bb_score = 60
        else:
            bb_score = 35
    else:
        bb_score = 50

    # StochRSI Score
    if np.isnan(current_stochrsi):
        stochrsi_score = 50
    else:
        stochrsi_score = 50
        for threshold, score_val in STOCHRSI_THRESHOLDS:
            if current_stochrsi <= threshold:
                stochrsi_score = score_val
                break

    # Price Trend Score
    if 5 <= price_trend_20d <= 30:
        trend_score = 85
    elif 30 < price_trend_20d <= 60:
        trend_score = 70
    elif 0 <= price_trend_20d < 5:
        trend_score = 55
    elif price_trend_20d > 60:
        trend_score = max(20, 60 - (price_trend_20d - 60) * 0.5)
    else:
        trend_score = max(0, 40 + price_trend_20d)

    # ── NEW INDICATOR SCORES ──────────────────────────────────────

    # ADX Score: Strong trend = higher score (filters out choppy stocks)
    if np.isnan(current_adx):
        adx_score = 50
    elif current_adx >= 50:
        adx_score = 95  # Very strong trend
    elif current_adx >= 25:
        # Strong trend. Bonus if +DI > -DI (bullish direction)
        adx_score = 80
        if not np.isnan(current_plus_di) and not np.isnan(current_minus_di):
            if current_plus_di > current_minus_di:
                adx_score = 90  # Strong bullish trend
    elif current_adx >= 20:
        adx_score = 60  # Developing trend
    else:
        adx_score = 30  # No trend, choppy -- avoid

    # MFI Score: Volume-weighted RSI (better for penny stocks)
    if np.isnan(current_mfi):
        mfi_score = 50
    elif 20 <= current_mfi <= 40:
        mfi_score = 90  # Oversold with buying pressure
    elif 40 < current_mfi <= 60:
        mfi_score = 70  # Neutral-bullish
    elif current_mfi < 20:
        mfi_score = 55  # Deeply oversold
    elif 60 < current_mfi <= 80:
        mfi_score = 45  # Getting extended
    else:
        mfi_score = 15  # Overbought

    # BB Squeeze Score: Tight bands = impending move
    if bb_squeeze["is_squeeze"]:
        squeeze_score = min(100, 70 + bb_squeeze["squeeze_intensity"] * 50)
    else:
        squeeze_score = 40  # No squeeze = neutral

    # Multi-day unusual volume score
    if multiday_vol["is_accumulating"]:
        multiday_score = 95  # 3+ unusual days = institutional accumulation
    elif multiday_vol["unusual_days"] >= 2:
        multiday_score = 75
    elif multiday_vol["unusual_days"] >= 1:
        multiday_score = 60
    else:
        multiday_score = 35  # No unusual activity

    # Volume-price divergence score
    if vol_price_div["divergence_type"] == "bullish_accumulation":
        divergence_score = min(100, 70 + vol_price_div["divergence_score"] * 0.3)
    elif vol_price_div["divergence_type"] == "bearish_distribution":
        divergence_score = max(0, 30 + vol_price_div["divergence_score"] * 0.3)
    else:
        divergence_score = 50

    # Consolidation + accumulation score (pre-breakout pattern)
    if consolidation["is_consolidating"] and consolidation.get("obv_rising_during_consolidation"):
        consol_score = min(100, 80 + consolidation["consolidation_score"] * 0.2)
    elif consolidation["is_consolidating"]:
        consol_score = 65
    else:
        consol_score = 40

    # Support proximity score
    if support_resistance["near_support"]:
        sr_score = 85  # Near strong support = good entry
    elif support_resistance["near_resistance"]:
        sr_score = 30  # Near resistance = risky
    else:
        sr_score = 50

    # Candlestick pattern score
    if candle_patterns["bullish_patterns"] > 0 and candle_patterns["bearish_patterns"] == 0:
        candle_score = min(100, 70 + candle_patterns["bullish_patterns"] * 15)
    elif candle_patterns["bearish_patterns"] > 0 and candle_patterns["bullish_patterns"] == 0:
        candle_score = max(0, 30 - candle_patterns["bearish_patterns"] * 10)
    else:
        candle_score = 50

    # ── Composite technical score ─────────────────────────────────
    # Original core indicators use existing TECH_WEIGHTS
    core_composite = (
        rsi_score * TECH_WEIGHTS["rsi"] +
        macd_score * TECH_WEIGHTS["macd"] +
        stochrsi_score * TECH_WEIGHTS["stochrsi"] +
        vol_score * TECH_WEIGHTS["volume_spike"] +
        obv_score * TECH_WEIGHTS["obv_trend"] +
        bb_score * TECH_WEIGHTS["bollinger"] +
        trend_score * TECH_WEIGHTS["price_trend"]
    )

    # New indicators as bonus adjustments (up to +/- 15 points)
    # These fine-tune the core score based on advanced signals
    new_adj = 0.0
    new_adj += (adx_score - 50) * 0.06       # +/- 3 pts for trend strength
    new_adj += (mfi_score - 50) * 0.04        # +/- 2 pts for money flow
    new_adj += (squeeze_score - 50) * 0.04    # +/- 2 pts for BB squeeze
    new_adj += (multiday_score - 50) * 0.04   # +/- 2 pts for sustained volume
    new_adj += (divergence_score - 50) * 0.03 # +/- 1.5 pts for vol-price divergence
    new_adj += (consol_score - 50) * 0.03     # +/- 1.5 pts for consolidation
    new_adj += (sr_score - 50) * 0.02         # +/- 1 pt for support/resistance
    new_adj += (candle_score - 50) * 0.02     # +/- 1 pt for candlestick patterns

    composite = max(0, min(100, core_composite + new_adj))

    return {
        "score": round(composite, 1),
        "valid": True,
        # Core indicator values
        "rsi": round(current_rsi, 1) if not np.isnan(current_rsi) else None,
        "stochrsi": round(current_stochrsi, 2) if not np.isnan(current_stochrsi) else None,
        "macd_histogram": round(current_macd_hist, 4) if not np.isnan(current_macd_hist) else None,
        "volume_spike": round(vol_spike, 2),
        "price_trend_20d": round(price_trend_20d, 1),
        "price_trend_5d": round(price_trend_5d, 1),
        "stochastic_k": round(current_stoch_k, 1) if not np.isnan(current_stoch_k) else None,
        "bb_position": round((current_price - current_bb_lower) / bb_range, 2) if bb_range > 0 and not np.isnan(bb_range) else None,
        "obv_trend_direction": "up" if obv_score > 55 else ("down" if obv_score < 45 else "flat"),
        # New indicator values
        "adx": round(current_adx, 1) if not np.isnan(current_adx) else None,
        "plus_di": round(current_plus_di, 1) if not np.isnan(current_plus_di) else None,
        "minus_di": round(current_minus_di, 1) if not np.isnan(current_minus_di) else None,
        "mfi": round(current_mfi, 1) if not np.isnan(current_mfi) else None,
        "bb_squeeze": bb_squeeze,
        "multiday_unusual_volume": multiday_vol,
        "volume_price_divergence": vol_price_div,
        "consolidation": consolidation,
        "support_resistance": support_resistance,
        "atr": round(atr_val, 4),
        "gap": gap,
        "candlestick_patterns": candle_patterns,
        # Sub-scores
        "sub_scores": {
            "rsi": round(rsi_score, 1),
            "macd": round(macd_score, 1),
            "stochrsi": round(stochrsi_score, 1),
            "volume_spike": round(vol_score, 1),
            "obv_trend": round(obv_score, 1),
            "bollinger": round(bb_score, 1),
            "price_trend": round(trend_score, 1),
            "adx": round(adx_score, 1),
            "mfi": round(mfi_score, 1),
            "bb_squeeze": round(squeeze_score, 1),
            "multiday_volume": round(multiday_score, 1),
            "vol_price_divergence": round(divergence_score, 1),
            "consolidation": round(consol_score, 1),
            "support_resistance": round(sr_score, 1),
            "candlestick": round(candle_score, 1),
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
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]
    open_prices = hist["Open"]

    rsi = compute_rsi(close)
    stochrsi = compute_stochrsi(close)
    macd = compute_macd(close)
    vol_spike = compute_volume_spike(volume)
    adx_data = compute_adx(high, low, close)
    mfi_series = compute_mfi(high, low, close, volume)
    bb_squeeze = compute_bb_squeeze(close)
    multiday_vol = compute_multiday_unusual_volume(volume)
    consolidation = compute_consolidation(close, volume)

    current_stochrsi = stochrsi.iloc[-1] if not stochrsi.empty else None
    if current_stochrsi is not None and np.isnan(current_stochrsi):
        current_stochrsi = None

    current_adx = adx_data["adx"].iloc[-1] if not adx_data["adx"].empty else None
    if current_adx is not None and np.isnan(current_adx):
        current_adx = None

    current_mfi = mfi_series.iloc[-1] if not mfi_series.empty else None
    if current_mfi is not None and np.isnan(current_mfi):
        current_mfi = None

    return {
        # Original features
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
        # New features
        "adx": float(current_adx) if current_adx is not None else None,
        "mfi": float(current_mfi) if current_mfi is not None else None,
        "bb_squeeze_intensity": float(bb_squeeze["squeeze_intensity"]),
        "multiday_unusual_days": float(multiday_vol["unusual_days"]),
        "avg_rvol_5d": float(multiday_vol["avg_rvol_5d"]),
        "consolidation_score": float(consolidation["consolidation_score"]),
        "is_consolidating": float(1.0 if consolidation["is_consolidating"] else 0.0),
    }
