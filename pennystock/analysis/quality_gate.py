"""
Quality Gate: Kill filters + scoring penalties for stock screening.

LAYER 1 of the two-layer scoring system.

Two tiers of red flags:

HARD KILLS (instant disqualification, score 0):
  These catch truly broken situations that no technical setup can overcome:
  - Fraud / SEC / DOJ investigations (management is crooked)
  - Core product failures (Phase 3 trial failed, etc.)
  - Shell company indicators (no revenue, no employees)
  - Toxic gross margins (< 5% = business model doesn't work)
  - Cash runway exhaustion (< 6 months of cash left at current burn)
  - Pre-revenue burn (< $1M revenue + burning > $50M/year)
  - Already pumped (> 100% gain in 5 days = too late, chasing the pump)
  - Pump-and-dump aftermath (>300% spike in <30 days then >80% crash)
  - Negative shareholder equity (balance sheet is underwater)
  - Sub-dime price (< $0.10 = untradeable garbage)
  - Extreme profit margin losses (< -200% = hemorrhaging money)

SCORING PENALTIES (reduce score but don't kill):
  "Normal penny stock shadiness" -- bad signs but not fatal:
  - Going concern (auditor doubt -- common in penny land)
  - Delisting / compliance notices (often resolved, stock can recover)
  - Extreme price decay (85%+ from 52w high -- scaled, 95%+ hits harder)
  - Recent reverse splits (scaled -- 1-for-50+ gets heavier penalty)
  - Excessive float (> 100M shares -- harder to squeeze but not impossible)
  - Micro-employee count (< 10 employees = likely shell/zombie company)

The key insight: catch stocks BEFORE the pump, not after. The "Already Pumped"
filter is the most important -- a stock up >100% in 5 days has already moved.

Real-world examples:
  - QNCX (Quince): "Phase 3 failed" in news + zero revenue shell -> KILL
  - AZI (Autozi): 1.6% gross margin -> KILL (toxic margins)
  - AGL (Agilon): fraud lawsuit + $20M/month cash burn -> KILL
  - GUTS: $3K revenue, -$86M/year burn -> KILL (pre-revenue burn)
  - ZONE (CleanCore): going concern in 10-K -> PENALTY (not kill)
  - OPAD (Offerpad): delisting notice -> PENALTY (not kill)
  - DXST: 95% decay, reverse split -> PENALTIES (accumulated)
  - SOPA: going concern + reverse split + decay -> PENALTIES (stacked)
"""

from loguru import logger

import numpy as np

from pennystock.config import (
    KILL_FRAUD_KEYWORDS,
    KILL_FAILURE_KEYWORDS,
    KILL_SHELL_MAX_REVENUE,
    KILL_SHELL_MAX_MARKET_CAP,
    KILL_MIN_GROSS_MARGIN,
    KILL_MIN_CASH_RUNWAY_YEARS,
    KILL_PRE_REVENUE_MAX_REVENUE,
    KILL_PRE_REVENUE_MIN_BURN,
    KILL_ALREADY_PUMPED_PCT,
    KILL_ALREADY_PUMPED_DAYS,
    KILL_NEGATIVE_EQUITY,
    KILL_MIN_PRICE,
    KILL_EXTREME_LOSS_MARGIN,
    KILL_PUMP_DUMP_SPIKE_RATIO,
    KILL_PUMP_DUMP_SPIKE_WINDOW,
    KILL_PUMP_DUMP_DECLINE_PCT,
    KILL_GOING_CONCERN,
    KILL_DELISTING_KEYWORDS,
    PENALTY_GOING_CONCERN,
    PENALTY_DELISTING_NOTICE,
    PENALTY_PRICE_DECAY,
    PENALTY_PRICE_DECAY_THRESHOLD,
    PENALTY_PRICE_DECAY_EXTREME,
    PENALTY_PRICE_DECAY_EXTREME_THRESHOLD,
    PENALTY_REVERSE_SPLIT,
    PENALTY_REVERSE_SPLIT_EXTREME,
    PENALTY_REVERSE_SPLIT_EXTREME_RATIO,
    PENALTY_EXCESSIVE_FLOAT,
    PENALTY_MAX_FLOAT,
    PENALTY_MICRO_EMPLOYEES,
    PENALTY_MICRO_EMPLOYEES_THRESHOLD,
)
from pennystock.data.yahoo_client import (
    get_stock_info, get_news, has_recent_reverse_split, get_price_history,
)
from pennystock.data.sec_client import check_going_concern


def run_kill_filters(ticker: str, info: dict = None, news: list = None) -> dict:
    """
    Run all quality checks on a stock: hard kills + scoring penalties.

    Args:
        ticker: Stock ticker symbol.
        info: Pre-fetched Yahoo Finance info dict (optional, will fetch if None).
        news: Pre-fetched news articles list (optional, will fetch if None).

    Returns:
        {
            "passed": bool,           # True = stock survived hard kills
            "killed": bool,           # True = stock was disqualified
            "kill_reasons": list,     # Human-readable reasons for each kill
            "penalties": list,        # Penalty descriptions (not kills)
            "total_penalty": int,     # Total points to deduct from final score
            "filters_run": int,       # How many filters were checked
            "filters_failed": int,    # How many hard kills triggered
        }
    """
    if info is None:
        info = get_stock_info(ticker)
    if news is None:
        news = get_news(ticker)

    kill_reasons = []
    penalties = []
    total_penalty = 0

    news_titles = [a.get("title", "").lower() for a in (news or [])]
    total_revenue = info.get("total_revenue", 0) or 0
    market_cap = info.get("market_cap", 0) or 0
    employees = info.get("full_time_employees", 0) or 0
    price = info.get("price", 0) or 0
    high_52w = info.get("52w_high", 0) or 0
    gross_margins = info.get("gross_margins", 0) or 0
    total_cash = info.get("total_cash", 0) or 0
    operating_cf = info.get("operating_cashflow", 0) or 0
    float_shares = info.get("float_shares", 0) or 0

    # ═════════════════════════════════════════════════════════════════
    # HARD KILLS: Truly broken situations
    # ═════════════════════════════════════════════════════════════════

    # ── Kill 1: Already Pumped (THE most important filter) ─────────
    # Catch stocks BEFORE the pump. If it already ran >100% in 5 days,
    # you're buying someone else's exit liquidity.
    try:
        gain_5d = _check_recent_gain(ticker, KILL_ALREADY_PUMPED_DAYS)
        if gain_5d is not None and gain_5d >= KILL_ALREADY_PUMPED_PCT:
            kill_reasons.append(
                f"ALREADY PUMPED: Stock gained {gain_5d:.1f}% in the last "
                f"{KILL_ALREADY_PUMPED_DAYS} trading days "
                f"(threshold: {KILL_ALREADY_PUMPED_PCT:.0f}%). "
                f"You're late -- this already ran."
            )
    except Exception as e:
        logger.debug(f"Already-pumped check failed for {ticker}: {e}")

    # ── Kill 2: Pump-and-Dump Aftermath ──────────────────────────
    # Detect stocks that had a massive spike (>300% in <30 trading days)
    # followed by a crash (>80% from peak). The money has been extracted.
    # Would have killed: MSGY (IPO $4 -> $22.20 in weeks -> $0.66)
    # Safe for pre-pump winners: SMX, BBGI, BEAT etc. hadn't spiked yet
    try:
        pnd = _check_pump_dump_aftermath(ticker)
        if pnd["is_pump_dump"]:
            kill_reasons.append(
                f"PUMP-AND-DUMP AFTERMATH: Stock spiked {pnd['spike_magnitude']:.0f}x "
                f"in {pnd['spike_days']} trading days (peak ${pnd['peak_price']:.2f}), "
                f"then crashed {pnd['decline_pct']:.0f}% from peak. "
                f"The party is over -- this is post-manipulation dead money."
            )
    except Exception as e:
        logger.debug(f"Pump-and-dump check failed for {ticker}: {e}")

    # ── Kill 3: Fraud / SEC Investigation (news headlines) ─────────
    for title in news_titles:
        for keyword in KILL_FRAUD_KEYWORDS:
            if keyword in title:
                kill_reasons.append(
                    f"FRAUD/INVESTIGATION: News headline contains '{keyword}' "
                    f"-> \"{title}\""
                )
                break
        if any(r.startswith("FRAUD") for r in kill_reasons):
            break

    # ── Kill 4: Core Product Failure (news headlines) ──────────────
    for title in news_titles:
        for keyword in KILL_FAILURE_KEYWORDS:
            if keyword in title:
                kill_reasons.append(
                    f"PRODUCT FAILURE: News headline contains '{keyword}' "
                    f"-> \"{title}\""
                )
                break
        if any(r.startswith("PRODUCT") for r in kill_reasons):
            break

    # ── Kill 5: Shell Company Indicators ───────────────────────────
    if total_revenue < KILL_SHELL_MAX_REVENUE and market_cap > 0:
        if market_cap < KILL_SHELL_MAX_MARKET_CAP:
            kill_reasons.append(
                f"SHELL COMPANY: Revenue ${total_revenue:,.0f} "
                f"(< ${KILL_SHELL_MAX_REVENUE:,.0f}) and market cap "
                f"${market_cap:,.0f} (< ${KILL_SHELL_MAX_MARKET_CAP:,.0f})"
            )
        elif employees == 0 and total_revenue == 0:
            kill_reasons.append(
                "SHELL COMPANY: Zero revenue AND zero reported employees"
            )

    # ── Kill 6: Toxic Gross Margins ────────────────────────────────
    if total_revenue > KILL_SHELL_MAX_REVENUE and 0 < gross_margins < KILL_MIN_GROSS_MARGIN:
        kill_reasons.append(
            f"TOXIC MARGINS: Gross margin {gross_margins*100:.1f}% "
            f"(< {KILL_MIN_GROSS_MARGIN*100:.0f}% minimum). "
            f"Business model does not generate meaningful gross profit."
        )

    # ── Kill 7: Cash Runway Exhaustion ─────────────────────────────
    if operating_cf < 0 and total_cash > 0:
        annual_burn = abs(operating_cf)
        runway_years = total_cash / annual_burn if annual_burn > 0 else float("inf")

        if runway_years < KILL_MIN_CASH_RUNWAY_YEARS:
            runway_months = runway_years * 12
            monthly_burn = annual_burn / 12
            kill_reasons.append(
                f"CASH DEATH SPIRAL: Only {runway_months:.1f} months of cash "
                f"remaining (${total_cash:,.0f} cash / ${monthly_burn:,.0f}/month "
                f"burn). Threshold: {KILL_MIN_CASH_RUNWAY_YEARS * 12:.0f} months."
            )

    # ── Kill 8: Pre-Revenue Company with Massive Burn ────────────
    if total_revenue < KILL_PRE_REVENUE_MAX_REVENUE:
        if operating_cf < KILL_PRE_REVENUE_MIN_BURN:
            kill_reasons.append(
                f"PRE-REVENUE BURN: Revenue ${total_revenue:,.0f} "
                f"(< ${KILL_PRE_REVENUE_MAX_REVENUE:,.0f}) with "
                f"${abs(operating_cf)/1e6:.1f}M/year cash burn "
                f"(> ${abs(KILL_PRE_REVENUE_MIN_BURN)/1e6:.0f}M/year max). "
                f"Pre-revenue company burning through cash with no product revenue."
            )

    # ── Kill 9: Negative Shareholder Equity ──────────────────────
    # Balance sheet is underwater -- liabilities exceed assets.
    # Would have killed: SMXT (-$11.78M), XPON (-$65M)
    if KILL_NEGATIVE_EQUITY:
        stockholders_equity = info.get("stockholders_equity")
        book_value = info.get("book_value")
        if stockholders_equity is not None and stockholders_equity < 0:
            kill_reasons.append(
                f"NEGATIVE EQUITY: Stockholders' equity is "
                f"${stockholders_equity:,.0f}. Liabilities exceed assets -- "
                f"the company is technically insolvent."
            )
        elif book_value is not None and book_value < 0:
            kill_reasons.append(
                f"NEGATIVE EQUITY: Book value per share is "
                f"${book_value:.2f}. Balance sheet is underwater."
            )

    # ── Kill 10: Sub-Dime Stock Price ────────────────────────────
    # Stocks below $0.10 are untradeable penny garbage, usually
    # manipulated or dying. Would have killed: BYAH ($0.05)
    if price > 0 and price < KILL_MIN_PRICE:
        kill_reasons.append(
            f"SUB-DIME PRICE: Stock at ${price:.4f} "
            f"(< ${KILL_MIN_PRICE:.2f} minimum). "
            f"Sub-dime stocks are untradeable, illiquid, and manipulated."
        )

    # ── Kill 11: Extreme Profit Margin Losses ────────────────────
    # Companies with profit margins worse than -200% are hemorrhaging
    # money at an absurd rate. Would have killed: BYAH (-701%), XPON
    profit_margins = info.get("profit_margins")
    if profit_margins is not None and profit_margins < KILL_EXTREME_LOSS_MARGIN:
        kill_reasons.append(
            f"EXTREME LOSSES: Profit margin {profit_margins*100:.0f}% "
            f"(< {KILL_EXTREME_LOSS_MARGIN*100:.0f}% threshold). "
            f"Company is losing money at a catastrophic rate."
        )

    # ═════════════════════════════════════════════════════════════════
    # SCORING PENALTIES: Normal penny stock shadiness
    # These reduce the final score but don't instantly disqualify.
    # A stock with strong setup/technicals/fundamentals can overcome these.
    # ═════════════════════════════════════════════════════════════════

    # ── Penalty: Going Concern (SEC EDGAR) ─────────────────────────
    if KILL_GOING_CONCERN:
        try:
            if check_going_concern(ticker):
                total_penalty += PENALTY_GOING_CONCERN
                penalties.append(
                    f"GOING CONCERN (-{PENALTY_GOING_CONCERN}pts): Auditor doubt "
                    f"about company survival found in recent SEC filings. "
                    f"Common in penny stock land but still a red flag."
                )
        except Exception as e:
            logger.debug(f"Going concern check failed for {ticker}: {e}")

    # ── Penalty: Delisting / Compliance Notice (news headlines) ────
    for title in news_titles:
        for keyword in KILL_DELISTING_KEYWORDS:
            if keyword in title:
                total_penalty += PENALTY_DELISTING_NOTICE
                penalties.append(
                    f"DELISTING NOTICE (-{PENALTY_DELISTING_NOTICE}pts): "
                    f"News headline contains '{keyword}' -> \"{title}\". "
                    f"Often resolved, but adds risk."
                )
                break  # One delisting headline is enough for the penalty
        if any("DELISTING" in p for p in penalties):
            break

    # ── Penalty: Extreme Price Decay (SCALED) ────────────────────
    if high_52w > 0 and price > 0:
        decay_ratio = price / high_52w
        if decay_ratio < PENALTY_PRICE_DECAY_EXTREME_THRESHOLD:
            # 95%+ decline (e.g. BYAH at 99.9%) -- much heavier penalty
            pct_decline = (1 - decay_ratio) * 100
            total_penalty += PENALTY_PRICE_DECAY_EXTREME
            penalties.append(
                f"CATASTROPHIC DECAY (-{PENALTY_PRICE_DECAY_EXTREME}pts): "
                f"Price ${price:.4f} is {pct_decline:.1f}% below 52-week high "
                f"${high_52w:.2f} (ratio {decay_ratio:.4f}). "
                f"This stock has been destroyed."
            )
        elif decay_ratio < PENALTY_PRICE_DECAY_THRESHOLD:
            pct_decline = (1 - decay_ratio) * 100
            total_penalty += PENALTY_PRICE_DECAY
            penalties.append(
                f"EXTREME DECAY (-{PENALTY_PRICE_DECAY}pts): Price ${price:.2f} "
                f"is {pct_decline:.1f}% below 52-week high ${high_52w:.2f} "
                f"(ratio {decay_ratio:.4f}). Could be deep value or could be dead."
            )

    # ── Penalty: Recent Reverse Split (SCALED for extreme ratios) ──
    try:
        rs = has_recent_reverse_split(ticker, months=6)
        if rs["has_reverse_split"]:
            inv_ratio = int(1 / rs["split_ratio"]) if rs["split_ratio"] > 0 else 0
            ratio_str = f"1-for-{inv_ratio}" if inv_ratio > 0 else "unknown"
            if inv_ratio >= PENALTY_REVERSE_SPLIT_EXTREME_RATIO:
                # Extreme reverse split (1-for-50+) -- much heavier penalty
                total_penalty += PENALTY_REVERSE_SPLIT_EXTREME
                penalties.append(
                    f"EXTREME REVERSE SPLIT (-{PENALTY_REVERSE_SPLIT_EXTREME}pts): "
                    f"{ratio_str} reverse split on {rs['split_date']}. "
                    f"Ratios this extreme signal a dying company."
                )
            else:
                total_penalty += PENALTY_REVERSE_SPLIT
                penalties.append(
                    f"REVERSE SPLIT (-{PENALTY_REVERSE_SPLIT}pts): {ratio_str} "
                    f"reverse split on {rs['split_date']}. Desperation move to "
                    f"maintain listing, but stock may stabilize."
                )
    except Exception as e:
        logger.debug(f"Reverse split check failed for {ticker}: {e}")

    # ── Penalty: Excessive Float ───────────────────────────────────
    if float_shares > PENALTY_MAX_FLOAT:
        total_penalty += PENALTY_EXCESSIVE_FLOAT
        penalties.append(
            f"EXCESSIVE FLOAT (-{PENALTY_EXCESSIVE_FLOAT}pts): "
            f"{float_shares/1e6:.1f}M shares float "
            f"(> {PENALTY_MAX_FLOAT/1e6:.0f}M max). Harder to squeeze "
            f"but not impossible with enough catalyst."
        )

    # ── Penalty: Micro-Employee Count ────────────────────────────
    # Companies with fewer than 10 employees are often shells, zombies,
    # or SPACs masquerading as operating companies.
    # Would have penalized: ATON (4 employees)
    if 0 < employees < PENALTY_MICRO_EMPLOYEES_THRESHOLD:
        total_penalty += PENALTY_MICRO_EMPLOYEES
        penalties.append(
            f"MICRO EMPLOYEES (-{PENALTY_MICRO_EMPLOYEES}pts): "
            f"Only {employees} full-time employee(s) reported. "
            f"Companies with < {PENALTY_MICRO_EMPLOYEES_THRESHOLD} FTEs "
            f"are often shells, zombies, or SPACs."
        )

    # ── Compile result ─────────────────────────────────────────────
    total_hard_kills = 11  # 7 original + pump-dump + neg equity + sub-dime + extreme losses
    killed = len(kill_reasons) > 0

    if killed:
        logger.info(f"KILLED {ticker}: {len(kill_reasons)} kill filter(s) triggered")
        for reason in kill_reasons:
            logger.info(f"  -> {reason}")

    if penalties:
        logger.info(
            f"PENALTIES {ticker}: {len(penalties)} penalty(ies), "
            f"total deduction: -{total_penalty}pts"
        )
        for pen in penalties:
            logger.info(f"  -> {pen}")

    return {
        "passed": not killed,
        "killed": killed,
        "kill_reasons": kill_reasons,
        "penalties": penalties,
        "total_penalty": total_penalty,
        "filters_run": total_hard_kills,
        "filters_failed": len(kill_reasons),
    }


def _check_recent_gain(ticker: str, days: int) -> float | None:
    """
    Check how much a stock has gained in the last N trading days.

    Returns the percentage gain (e.g., 150.0 for +150%), or None if
    insufficient data.
    """
    hist = get_price_history(ticker, period="1mo")
    if hist is None or hist.empty:
        return None

    close = hist["Close"]
    if len(close) < days + 1:
        return None

    price_then = close.iloc[-(days + 1)]
    price_now = close.iloc[-1]

    if price_then <= 0:
        return None

    return ((price_now - price_then) / price_then) * 100


def _check_pump_dump_aftermath(ticker: str) -> dict:
    """
    Detect pump-and-dump aftermath pattern in 1-year price history.

    Looks for: massive spike (>300% in <30 trading days) followed by
    crash (>80% from peak). This catches the AFTERMATH of manipulation --
    the money has already been extracted, the stock is dead.

    Safe for pre-pump winners: if we evaluate a stock BEFORE it spikes,
    no spike exists yet, so this filter won't trigger.

    Returns:
        {"is_pump_dump": True/False, ...details...}
    """
    hist = get_price_history(ticker, period="1y")
    if hist is None or hist.empty or len(hist) < 60:
        return {"is_pump_dump": False}

    close = hist["Close"]
    if hasattr(close, 'values'):
        prices = close.values.flatten()
    else:
        prices = np.array(close)

    current_price = float(prices[-1])
    peak_price = float(np.nanmax(prices))
    peak_pos = int(np.nanargmax(prices))

    if peak_price <= 0 or current_price <= 0:
        return {"is_pump_dump": False}

    # Must be down >80% from peak (current < 20% of peak)
    decline_ratio = current_price / peak_price
    if decline_ratio > KILL_PUMP_DUMP_DECLINE_PCT:
        return {"is_pump_dump": False}

    # Check if the rise to peak was rapid (<30 trading days from base)
    # Find where the stock was at 1/3 of peak price, looking backwards
    target_price = peak_price / KILL_PUMP_DUMP_SPIKE_RATIO
    spike_start = None
    search_start = max(0, peak_pos - 60)  # Look up to 60 days before peak

    for i in range(peak_pos - 1, search_start - 1, -1):
        if float(prices[i]) <= target_price:
            spike_start = i
            break

    if spike_start is None:
        # Price was always above 1/3 of peak in the lookback -- not a spike
        return {"is_pump_dump": False}

    spike_days = peak_pos - spike_start
    if spike_days > KILL_PUMP_DUMP_SPIKE_WINDOW:
        # Spike took too long -- this is a gradual rise, not a pump
        return {"is_pump_dump": False}

    spike_magnitude = peak_price / float(prices[spike_start])
    decline_pct = (1 - decline_ratio) * 100

    return {
        "is_pump_dump": True,
        "spike_days": spike_days,
        "peak_price": peak_price,
        "spike_magnitude": spike_magnitude,
        "decline_pct": decline_pct,
        "base_price": float(prices[spike_start]),
        "current_price": current_price,
    }
