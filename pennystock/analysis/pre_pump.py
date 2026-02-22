"""
Pre-Pump Signal Detection: The MEGA-ALGORITHM layer.

v3.0: Combines signals from 10+ strategies to detect stocks
BEFORE they pump, not after.

Strategies integrated:
  1. Short Interest Change Rate (YIBO's SI dropped 74.9% before pump)
  2. Float Rotation Potential (volume/float ratio capacity)
  3. Nasdaq Compliance Risk ($1 threshold = management incentive to pump)
  4. Volume Acceleration (multi-day sigma patterns)
  5. Signal Confluence (count independent bullish signals)
  6. Insider Ownership Lock (>70% = supply constrained)
  7. Short Squeeze Setup (low float + high SI + high DTC)

Key insight from YIBO (2/18/2026):
  - Short interest collapsed 74.9% (reported 5 days before pump)
  - Ultra-low float (7.13M) + 93% insider ownership
  - Below $1 Nasdaq compliance risk + reverse split authority
  - NO specific news catalyst -- pure setup/supply-demand play
  - Our algorithm found it as #1 but scored it too conservatively

Key insight from other 2/18/2026 pumps:
  - RXT (+242%): Partnership catalyst on extremely beaten-down stock
  - IBRX (+42%): EU approval -- PREDICTABLE from Dec CHMP positive opinion
  - RIME (+325%): AI narrative + contract wins + conference presentations
"""

import numpy as np
from loguru import logger

from pennystock.data.yahoo_client import get_stock_info


def score_pre_pump(ticker: str, info: dict = None, tech_features: dict = None) -> dict:
    """
    Score pre-pump setup potential using 7 independent signals.

    Returns:
        {
            "score": float (0-100),
            "signals": dict of individual signal results,
            "confluence_count": int (how many signals are bullish),
            "confidence": str ("HIGH", "MEDIUM", "LOW"),
        }
    """
    if info is None:
        info = get_stock_info(ticker)
    if tech_features is None:
        tech_features = {}

    signals = {}

    # ── Signal 1: Short Interest Change Rate ──────────────────────
    # YIBO's short interest dropped 74.9% before its 60% pump.
    # A collapsing short interest means shorts are covering (reducing
    # selling pressure) or anticipating something.
    si_result = _score_short_interest_change(info)
    signals["short_interest_change"] = si_result

    # ── Signal 2: Float Rotation Potential ────────────────────────
    # When recent volume approaches or exceeds the float, the stock
    # can make explosive moves. Volume/float > 0.5 = significant.
    fr_result = _score_float_rotation(info)
    signals["float_rotation"] = fr_result

    # ── Signal 3: Nasdaq Compliance Risk ─────────────────────────
    # Stocks below $1.00 with low floats have a strong compliance
    # incentive to boost the price above $1 (Nasdaq listing threshold).
    # Only applies to the sub-$1 subset of our $0.50-$5.00 range.
    nc_result = _score_compliance_risk(info)
    signals["compliance_risk"] = nc_result

    # ── Signal 4: Volume Acceleration ────────────────────────────
    # Use multi-day unusual volume from tech features if available.
    vol_result = _score_volume_acceleration(info, tech_features)
    signals["volume_acceleration"] = vol_result

    # ── Signal 5: Supply Lock (Insider + Low Float) ───────────────
    # >70% insider ownership + <10M float = extremely constrained supply.
    # Any demand spike = explosive move. YIBO: 93% insider, 7.13M float.
    lock_result = _score_supply_lock(info)
    signals["supply_lock"] = lock_result

    # ── Signal 6: Short Squeeze Setup ─────────────────────────────
    # High short interest + low float + high days-to-cover.
    sq_result = _score_squeeze_setup(info)
    signals["squeeze_setup"] = sq_result

    # ── Signal 7: Beaten Down Bounce Setup ────────────────────────
    # Near 52w low + recent stabilization = potential reversal candidate.
    bd_result = _score_beaten_down_setup(info)
    signals["beaten_down"] = bd_result

    # ── Confluence Scoring ────────────────────────────────────────
    # Count how many independent signals are bullish (>= 65 score).
    # More concurrent signals = higher conviction.
    bullish_count = sum(1 for s in signals.values() if s["score"] >= 65)
    total_signals = len(signals)

    # Weighted composite of all signals
    weights = {
        "short_interest_change": 0.20,   # Strongest predictor (YIBO proof)
        "supply_lock": 0.20,             # Supply constraint = explosive moves
        "float_rotation": 0.15,          # Volume capacity
        "squeeze_setup": 0.15,           # Squeeze fuel
        "compliance_risk": 0.10,         # Incentive alignment
        "volume_acceleration": 0.10,     # Early accumulation
        "beaten_down": 0.10,             # Reversal setup
    }

    composite = sum(
        signals[key]["score"] * weight
        for key, weight in weights.items()
    )

    # Confluence bonus: if 4+ signals bullish, add up to 15 points
    if bullish_count >= 5:
        composite += 15
    elif bullish_count >= 4:
        composite += 10
    elif bullish_count >= 3:
        composite += 5

    composite = max(0, min(100, composite))

    # Confidence level
    if bullish_count >= 5:
        confidence = "HIGH"
    elif bullish_count >= 3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "score": round(composite, 1),
        "signals": signals,
        "confluence_count": bullish_count,
        "total_signals": total_signals,
        "confidence": confidence,
    }


# ════════════════════════════════════════════════════════════════════
# INDIVIDUAL SIGNAL SCORERS
# ════════════════════════════════════════════════════════════════════

def _score_short_interest_change(info: dict) -> dict:
    """
    Score based on short interest change (current vs prior month).
    A DROP in short interest = shorts are covering = less selling pressure.
    YIBO: 301,657 -> 75,593 (74.9% decrease) -> pumped 60% five days later.
    """
    shares_short = info.get("shares_short", 0) or 0
    shares_short_prior = info.get("shares_short_prior", 0) or 0

    if shares_short_prior <= 0 or shares_short <= 0:
        return {"score": 50, "change_pct": 0, "signal": "no_data"}

    change_pct = ((shares_short - shares_short_prior) / shares_short_prior) * 100

    if change_pct <= -50:
        score = 95   # Massive short covering (YIBO-like)
    elif change_pct <= -30:
        score = 85   # Significant short covering
    elif change_pct <= -15:
        score = 70   # Moderate short covering
    elif change_pct <= 0:
        score = 55   # Slight reduction
    elif change_pct <= 20:
        score = 40   # Slight increase in shorts
    elif change_pct <= 50:
        score = 25   # Shorts piling in
    else:
        score = 15   # Massive short increase (bearish)

    signal = "bullish_covering" if change_pct < -15 else "neutral" if abs(change_pct) < 15 else "bearish_buildup"

    return {
        "score": score,
        "change_pct": round(change_pct, 1),
        "shares_short": shares_short,
        "shares_short_prior": shares_short_prior,
        "signal": signal,
    }


def _score_float_rotation(info: dict) -> dict:
    """
    Score based on recent volume relative to float.
    When volume approaches float size, supply gets tight.
    Float rotation > 1.0 = entire float turned over.
    """
    float_shares = info.get("float_shares", 0) or 0
    avg_volume = info.get("avg_volume", 0) or 0
    avg_volume_10d = info.get("avg_volume_10d", 0) or 0

    # Use 10-day average if available (more recent), else regular average
    volume = avg_volume_10d if avg_volume_10d > 0 else avg_volume

    if float_shares <= 0 or volume <= 0:
        return {"score": 50, "rotation": 0, "signal": "no_data"}

    rotation = volume / float_shares

    if rotation >= 1.0:
        score = 95   # Full float rotation daily
    elif rotation >= 0.5:
        score = 85   # Half float rotating daily
    elif rotation >= 0.2:
        score = 70   # Significant volume relative to float
    elif rotation >= 0.1:
        score = 55   # Moderate
    elif rotation >= 0.05:
        score = 40   # Low
    else:
        score = 25   # Very low activity relative to float

    return {
        "score": score,
        "rotation": round(rotation, 3),
        "volume": volume,
        "float": float_shares,
        "signal": "high_rotation" if rotation >= 0.2 else "normal",
    }


def _score_compliance_risk(info: dict) -> dict:
    """
    Score Nasdaq compliance risk as a POSITIVE pump catalyst.
    Stocks below $1 need to get above $1 or face delisting.
    Management has strong incentive to pump. This is bullish for
    short-term price action even if the company is garbage.
    """
    price = info.get("price", 0) or 0
    market_cap = info.get("market_cap", 0) or 0
    float_shares = info.get("float_shares", 0) or 0

    if price <= 0:
        return {"score": 50, "below_dollar": False, "signal": "no_data"}

    below_dollar = price < 1.00
    tiny_float = float_shares > 0 and float_shares < 15_000_000
    micro_cap = 0 < market_cap < 50_000_000

    score = 50
    if below_dollar:
        score += 20  # Strong compliance incentive
    if below_dollar and tiny_float:
        score += 15  # Easy to pump with low float
    if below_dollar and micro_cap:
        score += 10  # Small market cap = less capital needed to move

    score = min(100, score)

    return {
        "score": score,
        "below_dollar": below_dollar,
        "price": price,
        "signal": "compliance_pump_candidate" if below_dollar and tiny_float else "normal",
    }


def _score_volume_acceleration(info: dict, tech_features: dict) -> dict:
    """
    Score based on recent volume acceleration.
    Rising volume even before a news catalyst = accumulation.
    """
    avg_volume = info.get("avg_volume", 0) or 0
    avg_volume_10d = info.get("avg_volume_10d", 0) or 0

    # Check if 10-day average is significantly above overall average
    # This means recent volume is ACCELERATING
    if avg_volume <= 0 or avg_volume_10d <= 0:
        return {"score": 50, "ratio": 0, "signal": "no_data"}

    ratio = avg_volume_10d / avg_volume

    # Multi-day unusual volume from technical analysis
    multiday_unusual = tech_features.get("multiday_unusual_vol_days", 0)

    score = 50
    if ratio >= 3.0:
        score = 90   # 3x volume acceleration
    elif ratio >= 2.0:
        score = 80   # 2x volume acceleration
    elif ratio >= 1.5:
        score = 65   # 50% volume increase
    elif ratio >= 1.2:
        score = 55   # Slight increase
    else:
        score = 40   # Volume declining or flat

    # Bonus for multi-day unusual volume pattern
    if multiday_unusual >= 3:
        score = min(100, score + 15)
    elif multiday_unusual >= 2:
        score = min(100, score + 8)

    return {
        "score": score,
        "ratio": round(ratio, 2),
        "multiday_unusual": multiday_unusual,
        "signal": "accelerating" if ratio >= 1.5 else "normal",
    }


def _score_supply_lock(info: dict) -> dict:
    """
    Score supply constraint from insider ownership + float tightness.
    YIBO: 93% insider ownership + 7.13M float = extreme supply lock.
    High insider lock = very few shares available to trade =
    any demand spike creates explosive price moves.
    """
    insider_pct = info.get("insider_percent_held", 0) or 0
    float_shares = info.get("float_shares", 0) or 0
    shares_outstanding = info.get("shares_outstanding", 0) or 0

    score = 50

    # Insider ownership scoring
    if insider_pct >= 0.80:
        score += 25   # Extreme lock (YIBO-like: 93%)
    elif insider_pct >= 0.60:
        score += 20
    elif insider_pct >= 0.40:
        score += 12
    elif insider_pct >= 0.20:
        score += 5

    # Float tightness scoring
    if 0 < float_shares < 5_000_000:
        score += 25   # Ultra-tight float
    elif float_shares < 10_000_000:
        score += 18
    elif float_shares < 20_000_000:
        score += 10

    # Float as percentage of outstanding
    if shares_outstanding > 0 and float_shares > 0:
        float_pct = float_shares / shares_outstanding
        if float_pct < 0.15:
            score += 10   # Less than 15% of shares are tradeable

    score = min(100, score)

    return {
        "score": score,
        "insider_pct": round(insider_pct * 100, 1) if insider_pct else 0,
        "float_shares": float_shares,
        "signal": "locked" if score >= 75 else "normal",
    }


def _score_squeeze_setup(info: dict) -> dict:
    """
    Score short squeeze potential.
    Combines: short % of float + days to cover + float size.
    """
    short_pct = info.get("short_percent_of_float", 0) or 0
    short_ratio = info.get("short_ratio", 0) or 0  # days to cover
    float_shares = info.get("float_shares", 0) or 0

    score = 30  # Baseline

    # Short percent of float
    if short_pct >= 0.30:
        score += 30
    elif short_pct >= 0.20:
        score += 25
    elif short_pct >= 0.10:
        score += 15
    elif short_pct >= 0.05:
        score += 8

    # Days to cover
    if short_ratio >= 7:
        score += 25  # Shorts are trapped
    elif short_ratio >= 5:
        score += 18
    elif short_ratio >= 3:
        score += 10

    # Float size multiplier (low float amplifies squeeze)
    if 0 < float_shares < 10_000_000:
        score += 15
    elif float_shares < 20_000_000:
        score += 8

    score = min(100, score)
    is_setup = short_pct >= 0.10 and short_ratio >= 3 and float_shares < 20_000_000

    return {
        "score": score,
        "short_pct_float": round(short_pct * 100, 1) if short_pct else 0,
        "days_to_cover": round(short_ratio, 1),
        "is_squeeze_setup": is_setup,
        "signal": "squeeze_ready" if is_setup else "normal",
    }


def _score_beaten_down_setup(info: dict) -> dict:
    """
    Score reversal potential from beaten-down stocks.
    Near 52w low with signs of stabilization.
    """
    price = info.get("price", 0) or 0
    low_52w = info.get("52w_low", 0) or 0
    high_52w = info.get("52w_high", 0) or 0

    if price <= 0 or low_52w <= 0 or high_52w <= low_52w:
        return {"score": 50, "position_52w": None, "signal": "no_data"}

    # Position within 52-week range (0 = at low, 1 = at high)
    position = (price - low_52w) / (high_52w - low_52w)

    # Near bottom = higher score (contrarian value)
    if position <= 0.10:
        score = 85   # Within 10% of 52w low
    elif position <= 0.20:
        score = 75   # Near bottom 20%
    elif position <= 0.35:
        score = 60   # Bottom third
    elif position <= 0.50:
        score = 45   # Middle
    else:
        score = 25   # Upper half (less upside potential)

    return {
        "score": score,
        "position_52w": round(position, 3),
        "price": price,
        "52w_low": low_52w,
        "52w_high": high_52w,
        "signal": "near_bottom" if position <= 0.20 else "normal",
    }
