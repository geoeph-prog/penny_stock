"""
Fundamental analysis: two-layer scoring system.

Layer 1 (kill filters) lives in quality_gate.py.
Layer 2 (positive scoring) is here, organized into:

  A. SETUP QUALITY (40% of total score)
     - Float tightness: Ultra-low float = explosive moves
     - Insider ownership: High lock = supply constraint
     - Proximity to 52-week low: Near bottom = max upside
     - Price-to-book: Below book = margin of safety

  B. FUNDAMENTAL QUALITY (25% of total score, applied in algorithm.py)
     - Revenue growth: Real business momentum
     - Short interest setup: Squeeze fuel
     - Cash position: Survival + opportunity

Designed around the RIME/ORKT pattern:
  RIME: 5.7M float, near ATL, RSI 42-47, StochRSI 5.76, 300% ARR growth
  ORKT: 7.8M float, 67% insider lock, ATL, P/B 0.47, 353% SI increase
"""

from loguru import logger

from pennystock.config import (
    SETUP_WEIGHTS,
    FUNDAMENTAL_WEIGHTS,
    FLOAT_THRESHOLDS,
    INSIDER_THRESHOLDS,
    PROXIMITY_LOW_THRESHOLDS,
    PB_THRESHOLDS,
    REVENUE_GROWTH_THRESHOLDS,
    SHORT_INTEREST_THRESHOLDS,
)
from pennystock.data.yahoo_client import get_stock_info, get_quarterly_financials
from pennystock.data.sec_client import (
    get_insider_transactions, get_recent_filings, check_dilution_filings,
)


# ════════════════════════════════════════════════════════════════════
# SETUP QUALITY SCORING (40% of total)
# ════════════════════════════════════════════════════════════════════

def score_setup(ticker: str, info: dict = None) -> dict:
    """
    Score the structural setup quality of a stock.
    These are the supply/demand dynamics that enable explosive moves.

    Returns:
        {
            "score": float (0-100),
            "float_tightness": dict,
            "insider_ownership": dict,
            "proximity_to_low": dict,
            "price_to_book": dict,
        }
    """
    if info is None:
        info = get_stock_info(ticker)

    ft = _score_float_tightness(info)
    io = _score_insider_ownership(info)
    pl = _score_proximity_to_low(info)
    pb = _score_price_to_book(info)

    composite = (
        ft["score"] * SETUP_WEIGHTS["float_tightness"] +
        io["score"] * SETUP_WEIGHTS["insider_ownership"] +
        pl["score"] * SETUP_WEIGHTS["proximity_to_low"] +
        pb["score"] * SETUP_WEIGHTS["price_to_book"]
    )

    return {
        "score": round(max(0, min(100, composite)), 1),
        "float_tightness": ft,
        "insider_ownership": io,
        "proximity_to_low": pl,
        "price_to_book": pb,
    }


def _score_float_tightness(info: dict) -> dict:
    """
    Score based on float size. Ultra-low float = stock can move violently
    on relatively small volume.

    RIME: 5.7M float -> score 85
    ORKT: 7.8M free float -> score 85
    """
    float_shares = info.get("float_shares", 0) or 0
    shares_out = info.get("shares_outstanding", 0) or 0

    score = 15  # Default for unknown/very high float
    for threshold, pts in FLOAT_THRESHOLDS:
        if float_shares > 0 and float_shares <= threshold:
            score = pts
            break
        elif float_shares == 0 and shares_out > 0 and shares_out <= threshold:
            # Fallback to shares outstanding if float not available
            score = max(15, pts - 10)  # Slight penalty for using shares_out
            break

    # Bonus: if we can compute float ratio (float / outstanding)
    float_ratio = None
    if float_shares > 0 and shares_out > 0:
        float_ratio = float_shares / shares_out
        # Low float ratio means insiders/institutions hold a lot
        if float_ratio < 0.30:
            score = min(100, score + 10)

    return {
        "score": max(0, min(100, score)),
        "float_shares": float_shares,
        "shares_outstanding": shares_out,
        "float_ratio": round(float_ratio, 3) if float_ratio else None,
    }


def _score_insider_ownership(info: dict) -> dict:
    """
    Score based on insider ownership percentage.
    High insider ownership = management has skin in the game,
    plus it reduces effective supply (insider-locked shares rarely trade).

    ORKT: 67% insider-locked -> score 100
    """
    insider_pct = info.get("insider_percent_held", 0) or 0
    inst_pct = info.get("institution_percent_held", 0) or 0

    score = 20  # Default for unknown/low insider ownership
    for threshold, pts in INSIDER_THRESHOLDS:
        if insider_pct >= threshold:
            score = pts
            break

    # Modest bonus for institutional validation (not too much -- institutions
    # in penny stocks can also mean they're bagholding)
    if 0.05 <= inst_pct <= 0.40:
        score = min(100, score + 5)

    return {
        "score": max(0, min(100, score)),
        "insider_pct": round(insider_pct, 3),
        "institution_pct": round(inst_pct, 3),
    }


def _score_proximity_to_low(info: dict) -> dict:
    """
    Score based on how close the price is to its 52-week low.
    Near the bottom = maximum upside potential, limited downside.

    position = (price - 52w_low) / (52w_high - 52w_low)
    0.0 = at the low, 1.0 = at the high

    RIME: $0.83, low $0.73 -> position ~0.13 -> score 80
    ORKT: $0.62, ATL $0.62 -> position 0.0 -> score 95
    """
    price = info.get("price", 0) or 0
    low_52w = info.get("52w_low", 0) or 0
    high_52w = info.get("52w_high", 0) or 0

    if high_52w <= low_52w or price <= 0:
        return {"score": 50, "position": None, "price": price,
                "52w_low": low_52w, "52w_high": high_52w}

    position = (price - low_52w) / (high_52w - low_52w)
    position = max(0.0, min(1.0, position))

    score = 15  # Default for near the high
    for threshold, pts in PROXIMITY_LOW_THRESHOLDS:
        if position <= threshold:
            score = pts
            break

    return {
        "score": max(0, min(100, score)),
        "position": round(position, 3),
        "price": price,
        "52w_low": low_52w,
        "52w_high": high_52w,
    }


def _score_price_to_book(info: dict) -> dict:
    """
    Score based on price-to-book ratio.
    Below book value = you're buying assets at a discount.

    ORKT: P/B 0.47 -> score 100 (below half of book value)
    """
    price = info.get("price", 0) or 0
    book_value = info.get("book_value", 0) or 0
    ptb = info.get("price_to_book", 0) or 0

    # Compute P/B if not provided
    if ptb == 0 and book_value > 0 and price > 0:
        ptb = price / book_value

    if ptb <= 0 or book_value <= 0:
        # No book value data -- neutral score
        return {"score": 40, "price_to_book": None, "book_value": book_value}

    score = 15  # Default for very high P/B
    for threshold, pts in PB_THRESHOLDS:
        if ptb <= threshold:
            score = pts
            break

    return {
        "score": max(0, min(100, score)),
        "price_to_book": round(ptb, 3),
        "book_value": book_value,
    }


# ════════════════════════════════════════════════════════════════════
# FUNDAMENTAL QUALITY SCORING (25% of total)
# ════════════════════════════════════════════════════════════════════

def score_fundamentals(ticker: str, info: dict = None) -> dict:
    """
    Score the fundamental quality of a stock's business.
    Revenue growth, short interest setup, cash position, squeeze potential,
    and dilution risk.

    Returns:
        {
            "score": float (0-100),
            "revenue_growth": dict,
            "short_interest": dict,
            "cash_position": dict,
            "squeeze_composite": dict,
            "dilution_risk": dict,
        }
    """
    if info is None:
        info = get_stock_info(ticker)

    rev = _score_revenue_growth(ticker, info)
    si = _score_short_interest(info)
    cash = _score_cash_position(info)
    squeeze = _score_squeeze_composite(info)
    dilution = _score_dilution_risk(ticker)

    composite = (
        rev["score"] * FUNDAMENTAL_WEIGHTS["revenue_growth"] +
        si["score"] * FUNDAMENTAL_WEIGHTS["short_interest"] +
        cash["score"] * FUNDAMENTAL_WEIGHTS["cash_position"]
    )

    # Squeeze composite as bonus adjustment (+/- 5 pts max)
    squeeze_adj = (squeeze["score"] - 50) * 0.10
    # Dilution risk as penalty (-0 to -8 pts)
    dilution_adj = -dilution["penalty"]

    composite = max(0, min(100, composite + squeeze_adj + dilution_adj))

    return {
        "score": round(composite, 1),
        "revenue_growth": rev,
        "short_interest": si,
        "cash_position": cash,
        "squeeze_composite": squeeze,
        "dilution_risk": dilution,
    }


def _score_revenue_growth(ticker: str, info: dict) -> dict:
    """
    Score based on YoY revenue growth.
    Accelerating revenue is the single best fundamental signal for penny stocks.

    RIME: 300% ARR growth ($2.5M -> $9.7M) -> score 100
    OPAD: declining revenue -> score ~10 (this is a kill signal too)
    """
    rev_growth = info.get("revenue_growth", 0) or 0
    total_rev = info.get("total_revenue", 0) or 0

    score = 45  # Default neutral
    # Walk thresholds from highest to lowest
    for threshold, pts in REVENUE_GROWTH_THRESHOLDS:
        if rev_growth >= threshold:
            score = pts
            break

    # Bonus for absolute revenue scale (actual revenue > $1M is more credible)
    if total_rev > 10_000_000:
        score = min(100, score + 8)
    elif total_rev > 1_000_000:
        score = min(100, score + 4)

    # Check for accelerating quarterly revenue
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
                    score = min(100, score + 5)
                if len(recent) >= 3 and recent[0] > recent[1] > recent[2]:
                    score = min(100, score + 5)  # Accelerating growth bonus
        except Exception:
            pass

    # Bonus for EPS/earnings growth (rare in penny stocks but very powerful)
    # IQST had record quarterly revenue AND earnings beat -> +118%
    earnings_growth = info.get("earnings_growth", 0) or 0
    if earnings_growth > 0.5:
        score = min(100, score + 8)  # EPS growing 50%+
    elif earnings_growth > 0.2:
        score = min(100, score + 4)  # EPS growing 20%+
    elif earnings_growth > 0:
        score = min(100, score + 2)  # Any positive EPS growth

    return {
        "score": max(0, min(100, score)),
        "revenue_growth": rev_growth,
        "total_revenue": total_rev,
        "earnings_growth": earnings_growth,
    }


def _score_short_interest(info: dict) -> dict:
    """
    Score based on short interest as % of float.
    High short interest + low float = squeeze fuel.

    ORKT: Short interest rose 353% -- heavy squeeze setup -> score ~90
    Note: Short interest CHANGE is also valuable. We compute it from
    shares_short vs shares_short_prior (prior month) when available.
    """
    si_pct = info.get("short_percent_of_float", 0) or 0
    shares_short = info.get("shares_short", 0) or 0
    shares_short_prior = info.get("shares_short_prior", 0) or 0
    short_ratio = info.get("short_ratio", 0) or 0

    score = 40  # Default for low/unknown SI
    for threshold, pts in SHORT_INTEREST_THRESHOLDS:
        if si_pct >= threshold:
            score = pts
            break

    # Bonus for SHORT INTEREST INCREASING (squeeze building)
    if shares_short > 0 and shares_short_prior > 0:
        si_change = (shares_short - shares_short_prior) / shares_short_prior
        if si_change > 1.0:
            score = min(100, score + 15)  # SI more than doubled
        elif si_change > 0.5:
            score = min(100, score + 10)  # SI up 50%+
        elif si_change > 0.2:
            score = min(100, score + 5)   # SI up 20%+
        elif si_change < -0.3:
            # SI declining significantly -- shorts are covering,
            # could mean the squeeze already happened
            score = max(0, score - 10)

    # Bonus for high days-to-cover (short ratio)
    if short_ratio > 5:
        score = min(100, score + 10)  # Takes 5+ days to cover
    elif short_ratio > 3:
        score = min(100, score + 5)

    return {
        "score": max(0, min(100, score)),
        "short_pct_float": round(si_pct, 4) if si_pct else 0,
        "shares_short": shares_short,
        "shares_short_prior": shares_short_prior,
        "short_ratio": short_ratio,
    }


def _score_cash_position(info: dict) -> dict:
    """
    Score based on cash position relative to debt and burn rate.
    Healthy cash = company can survive and invest in growth.

    AGL: burning $20M/month with inadequate reserves -> near 0
    (actual kill happens in quality_gate.py, this just scores the spectrum)
    """
    total_cash = info.get("total_cash", 0) or 0
    total_debt = info.get("total_debt", 0) or 0
    operating_cf = info.get("operating_cashflow", 0) or 0
    current_ratio = info.get("current_ratio", 0) or 0

    score = 50  # Neutral baseline

    # Cash vs debt ratio
    if total_debt > 0:
        cash_debt_ratio = total_cash / total_debt
        if cash_debt_ratio > 3:
            score += 20
        elif cash_debt_ratio > 1.5:
            score += 12
        elif cash_debt_ratio > 0.5:
            score += 5
        elif cash_debt_ratio < 0.2:
            score -= 15
    elif total_cash > 0:
        score += 15  # Cash and no debt = great

    # Operating cash flow
    if operating_cf > 0:
        score += 15  # Cash flow positive is rare and valuable for penny stocks
    elif operating_cf < 0 and total_cash > 0:
        runway_years = total_cash / abs(operating_cf) if operating_cf != 0 else float("inf")
        if runway_years > 3:
            score += 5
        elif runway_years > 1:
            score += 0  # neutral
        elif runway_years > 0.5:
            score -= 10
        else:
            score -= 25  # Near death (kill filter should catch worst cases)

    # Current ratio
    if current_ratio > 2:
        score += 5
    elif current_ratio > 1:
        score += 2
    elif 0 < current_ratio < 0.5:
        score -= 10

    return {
        "score": round(max(0, min(100, score)), 1),
        "total_cash": total_cash,
        "total_debt": total_debt,
        "operating_cashflow": operating_cf,
        "current_ratio": current_ratio,
    }


def _score_squeeze_composite(info: dict) -> dict:
    """
    Composite short squeeze probability score.
    Combines: SI% + float size + days-to-cover + SI change + float ratio.

    The interaction effect matters: low float + high SI + rising SI +
    high DTC = much more than the sum of parts.

    Validated: SMX (1M float + squeeze dynamics = +1000%)
    BBGI (<1M float + 8 days-to-cover + social coordination = +312%)
    ORKT (353% SI increase + 7.8M float = squeeze setup)
    """
    si_pct = info.get("short_percent_of_float", 0) or 0
    float_shares = info.get("float_shares", 0) or 0
    short_ratio = info.get("short_ratio", 0) or 0
    shares_short = info.get("shares_short", 0) or 0
    shares_short_prior = info.get("shares_short_prior", 0) or 0

    score = 30  # Baseline

    # SI% of float (most important squeeze component)
    if si_pct >= 0.30:
        score += 25
    elif si_pct >= 0.20:
        score += 20
    elif si_pct >= 0.10:
        score += 12
    elif si_pct >= 0.05:
        score += 5

    # Float size (smaller = more violent squeeze)
    if 0 < float_shares < 5_000_000:
        score += 20  # Ultra-low float
    elif float_shares < 10_000_000:
        score += 15
    elif float_shares < 20_000_000:
        score += 8

    # Days to cover (higher = harder for shorts to exit)
    if short_ratio > 10:
        score += 15
    elif short_ratio > 5:
        score += 10
    elif short_ratio > 3:
        score += 5

    # SI change (rising = squeeze building)
    if shares_short > 0 and shares_short_prior > 0:
        si_change = (shares_short - shares_short_prior) / shares_short_prior
        if si_change > 1.0:
            score += 15  # Doubled
        elif si_change > 0.5:
            score += 10
        elif si_change > 0.2:
            score += 5
        elif si_change < -0.3:
            score -= 10  # Shorts covering = squeeze may be over

    return {
        "score": min(100, max(0, score)),
        "si_pct": round(si_pct, 4),
        "float_shares": float_shares,
        "short_ratio": short_ratio,
        "is_squeeze_setup": score >= 65,
    }


def _score_dilution_risk(ticker: str) -> dict:
    """
    Score dilution risk from SEC filings (S-1, S-3, 424B).
    S-3 shelf registrations on small companies are ticking time bombs.

    check_dilution_filings() already exists in sec_client but was
    previously unused. Now wired into fundamental scoring.
    """
    try:
        result = check_dilution_filings(ticker, days_back=180)
        count = result.get("dilution_filings", 0)
    except Exception:
        count = 0

    if count >= 3:
        penalty = 8  # Heavy dilution activity
    elif count >= 2:
        penalty = 5
    elif count >= 1:
        penalty = 3
    else:
        penalty = 0

    return {
        "dilution_filings_6m": count,
        "penalty": penalty,
        "has_dilution_risk": count > 0,
    }


# ════════════════════════════════════════════════════════════════════
# COMBINED ANALYSIS (backward-compatible with existing callers)
# ════════════════════════════════════════════════════════════════════

def analyze(ticker: str) -> dict:
    """
    Full fundamental analysis combining setup quality and fundamental quality.
    Backward-compatible interface.
    """
    info = get_stock_info(ticker)

    setup = score_setup(ticker, info)
    fundamentals = score_fundamentals(ticker, info)

    # Combined score: setup and fundamentals contribute to the overall
    # fundamental score reported here. The actual weighting against
    # technical/sentiment/catalyst happens in algorithm.py.
    combined = setup["score"] * 0.55 + fundamentals["score"] * 0.45

    return {
        "score": round(max(0, min(100, combined)), 1),
        "setup": setup,
        "fundamentals": fundamentals,
    }


def extract_features(ticker: str) -> dict:
    """
    Extract numerical fundamental features for algorithm learning.
    Returns flat dict of feature_name: value.
    Expanded to include the new dimensions (float, P/B, proximity-to-low).
    """
    info = get_stock_info(ticker)
    insider = get_insider_transactions(ticker, days_back=90)
    sec = get_recent_filings(ticker, days_back=90)

    # Compute derived values
    price = info.get("price", 0) or 0
    low_52w = info.get("52w_low", 0) or 0
    high_52w = info.get("52w_high", 0) or 0
    book_value = info.get("book_value", 0) or 0
    float_shares = info.get("float_shares", 0) or 0
    shares_out = info.get("shares_outstanding", 0) or 0
    shares_short = info.get("shares_short", 0) or 0
    shares_short_prior = info.get("shares_short_prior", 0) or 0

    # 52w position: 0 = at low, 1 = at high
    if high_52w > low_52w and price > 0:
        position_52w = (price - low_52w) / (high_52w - low_52w)
    else:
        position_52w = 0.5

    # P/B ratio
    ptb = info.get("price_to_book", 0) or 0
    if ptb == 0 and book_value > 0 and price > 0:
        ptb = price / book_value

    # Float ratio
    float_ratio = float_shares / shares_out if float_shares > 0 and shares_out > 0 else 0

    # Short interest change MoM
    si_change = 0
    if shares_short > 0 and shares_short_prior > 0:
        si_change = (shares_short - shares_short_prior) / shares_short_prior

    return {
        # Existing features
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
        "float_shares": float_shares,
        "insider_buys_90d": insider.get("recent_buys", 0),
        "insider_sells_90d": insider.get("recent_sells", 0),
        "insider_net_buys": insider.get("recent_buys", 0) - insider.get("recent_sells", 0),
        "sec_filings_90d": sec.get("total_filings", 0),
        "sec_8k_count": sec.get("filing_types", {}).get("8-K", 0),
        # NEW: Setup quality features
        "float_ratio": round(float_ratio, 4),
        "position_52w": round(position_52w, 4),
        "price_to_book": round(ptb, 4) if ptb else 0,
        "gross_margins": info.get("gross_margins", 0) or 0,
        "operating_cashflow": info.get("operating_cashflow", 0) or 0,
        "free_cashflow": info.get("free_cashflow", 0) or 0,
        "short_interest_change_mom": round(si_change, 4),
        # NEW: EPS growth and dilution risk
        "earnings_growth": info.get("earnings_growth", 0) or 0,
        "profit_margins": info.get("profit_margins", 0) or 0,
    }
