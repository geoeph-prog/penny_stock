"""
Quality Gate: Hard kill filters that disqualify fundamentally broken stocks.

LAYER 1 of the two-layer scoring system.
Any single kill filter triggering = instant disqualification (score 0).

These catch the categories of disasters that no technical setup can overcome:
  - Going concern (auditor doubts the company will survive, incl. 20-F foreign filers)
  - Delisting notices (exchange is kicking them out)
  - Fraud / SEC / DOJ investigations (management is crooked)
  - Core product failures (Phase 3 trial failed, etc.)
  - Shell company indicators (no revenue, no employees)
  - Extreme price decay (85%+ decline from 52w high = structural wreckage)
  - Toxic gross margins (< 5% = business model doesn't work)
  - Cash runway exhaustion (< 6 months of cash left at current burn)
  - Recent reverse splits (desperation move to maintain listing)
  - Excessive float (> 100M shares = declining mid-cap, not a setup)
  - Pre-revenue burn (< $1M revenue + burning > $50M/year)

Real-world examples this would have caught:
  - ZONE (CleanCore): going concern in 10-K
  - OPAD (Offerpad): delisting notice + negative revenue growth
  - QNCX (Quince): "Phase 3 failed" in news + zero revenue shell
  - AZI (Autozi): 1.6% gross margin + 99.97% price decay
  - AGL (Agilon): fraud lawsuit in news + $20M/month cash burn
  - DXST: 95% decay, reverse split vote, auditor change (foreign filer 20-F)
  - SOPA: 99.9% decay, going concern in 20-F, reverse split
  - SLQT: DOJ kickback lawsuit, 143.8M float (excessive)
  - GUTS: $3K revenue, -$86M/year burn (pre-revenue burn)
"""

from loguru import logger

from pennystock.config import (
    KILL_GOING_CONCERN,
    KILL_DELISTING_KEYWORDS,
    KILL_FRAUD_KEYWORDS,
    KILL_FAILURE_KEYWORDS,
    KILL_SHELL_MAX_REVENUE,
    KILL_SHELL_MAX_MARKET_CAP,
    KILL_PRICE_DECAY_THRESHOLD,
    KILL_MIN_GROSS_MARGIN,
    KILL_MIN_CASH_RUNWAY_YEARS,
    KILL_MAX_FLOAT,
    KILL_PRE_REVENUE_MAX_REVENUE,
    KILL_PRE_REVENUE_MIN_BURN,
)
from pennystock.data.yahoo_client import get_stock_info, get_news, has_recent_reverse_split
from pennystock.data.sec_client import check_going_concern


def run_kill_filters(ticker: str, info: dict = None, news: list = None) -> dict:
    """
    Run all hard kill filters on a stock.

    Args:
        ticker: Stock ticker symbol.
        info: Pre-fetched Yahoo Finance info dict (optional, will fetch if None).
        news: Pre-fetched news articles list (optional, will fetch if None).

    Returns:
        {
            "passed": bool,         # True = stock survived all filters
            "killed": bool,         # True = stock was disqualified
            "kill_reasons": list,   # Human-readable reasons for each kill
            "filters_run": int,     # How many filters were checked
            "filters_failed": int,  # How many filters triggered
        }
    """
    if info is None:
        info = get_stock_info(ticker)
    if news is None:
        news = get_news(ticker)

    kill_reasons = []

    # ── Filter 1: Going Concern (SEC EDGAR) ──────────────────────────
    if KILL_GOING_CONCERN:
        try:
            if check_going_concern(ticker):
                kill_reasons.append(
                    "GOING CONCERN: Auditor doubt about company survival "
                    "found in recent 10-K/10-Q SEC filings"
                )
        except Exception as e:
            logger.debug(f"Going concern check failed for {ticker}: {e}")

    # ── Filter 2: Delisting Notice (news headlines) ──────────────────
    news_titles = [a.get("title", "").lower() for a in (news or [])]
    for title in news_titles:
        for keyword in KILL_DELISTING_KEYWORDS:
            if keyword in title:
                kill_reasons.append(
                    f"DELISTING: News headline contains '{keyword}' -> \"{title}\""
                )
                break  # One delisting headline is enough
        if any(r.startswith("DELISTING") for r in kill_reasons):
            break

    # ── Filter 3: Fraud / SEC Investigation (news headlines) ─────────
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

    # ── Filter 4: Core Product Failure (news headlines) ──────────────
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

    # ── Filter 5: Shell Company Indicators ───────────────────────────
    total_revenue = info.get("total_revenue", 0) or 0
    market_cap = info.get("market_cap", 0) or 0
    employees = info.get("full_time_employees", 0) or 0

    is_shell = False
    if total_revenue < KILL_SHELL_MAX_REVENUE and market_cap > 0:
        if market_cap < KILL_SHELL_MAX_MARKET_CAP:
            is_shell = True
            kill_reasons.append(
                f"SHELL COMPANY: Revenue ${total_revenue:,.0f} "
                f"(< ${KILL_SHELL_MAX_REVENUE:,.0f}) and market cap "
                f"${market_cap:,.0f} (< ${KILL_SHELL_MAX_MARKET_CAP:,.0f})"
            )
        elif employees == 0 and total_revenue == 0:
            is_shell = True
            kill_reasons.append(
                "SHELL COMPANY: Zero revenue AND zero reported employees"
            )

    # ── Filter 6: Extreme Price Decay ────────────────────────────────
    price = info.get("price", 0) or 0
    high_52w = info.get("52w_high", 0) or 0

    if high_52w > 0 and price > 0:
        decay_ratio = price / high_52w
        if decay_ratio < KILL_PRICE_DECAY_THRESHOLD:
            pct_decline = (1 - decay_ratio) * 100
            kill_reasons.append(
                f"EXTREME DECAY: Price ${price:.2f} is {pct_decline:.1f}% below "
                f"52-week high ${high_52w:.2f} (ratio {decay_ratio:.4f} "
                f"< {KILL_PRICE_DECAY_THRESHOLD})"
            )

    # ── Filter 7: Toxic Gross Margins ────────────────────────────────
    gross_margins = info.get("gross_margins", 0) or 0
    # Only apply if the company has revenue (pre-revenue biotechs get a pass)
    if total_revenue > KILL_SHELL_MAX_REVENUE and 0 < gross_margins < KILL_MIN_GROSS_MARGIN:
        kill_reasons.append(
            f"TOXIC MARGINS: Gross margin {gross_margins*100:.1f}% "
            f"(< {KILL_MIN_GROSS_MARGIN*100:.0f}% minimum). "
            f"Business model does not generate meaningful gross profit."
        )

    # ── Filter 8: Cash Runway Exhaustion ─────────────────────────────
    total_cash = info.get("total_cash", 0) or 0
    operating_cf = info.get("operating_cashflow", 0) or 0

    if operating_cf < 0 and total_cash > 0:
        # Runway = how many years of cash left at current burn rate
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

    # ── Filter 9: Recent Reverse Split ────────────────────────────────
    try:
        rs = has_recent_reverse_split(ticker, months=6)
        if rs["has_reverse_split"]:
            ratio_str = f"1-for-{int(1/rs['split_ratio'])}" if rs["split_ratio"] > 0 else "unknown"
            kill_reasons.append(
                f"REVERSE SPLIT: {ratio_str} reverse split on {rs['split_date']}. "
                f"Penny stocks with recent reverse splits often continue declining."
            )
    except Exception as e:
        logger.debug(f"Reverse split check failed for {ticker}: {e}")

    # ── Filter 10: Max Float (too large for penny stock setup) ────────
    float_shares = info.get("float_shares", 0) or 0
    if float_shares > KILL_MAX_FLOAT:
        kill_reasons.append(
            f"EXCESSIVE FLOAT: {float_shares/1e6:.1f}M shares float "
            f"(> {KILL_MAX_FLOAT/1e6:.0f}M max). This is a declining mid-cap, "
            f"not a tight penny stock setup."
        )

    # ── Filter 11: Pre-Revenue Company with Massive Burn ────────────
    if total_revenue < KILL_PRE_REVENUE_MAX_REVENUE:
        if operating_cf < KILL_PRE_REVENUE_MIN_BURN:
            kill_reasons.append(
                f"PRE-REVENUE BURN: Revenue ${total_revenue:,.0f} "
                f"(< ${KILL_PRE_REVENUE_MAX_REVENUE:,.0f}) with "
                f"${abs(operating_cf)/1e6:.1f}M/year cash burn "
                f"(> ${abs(KILL_PRE_REVENUE_MIN_BURN)/1e6:.0f}M/year max). "
                f"Pre-revenue company burning through cash with no product revenue."
            )

    # ── Compile result ───────────────────────────────────────────────
    total_filters = 11
    killed = len(kill_reasons) > 0

    if killed:
        logger.info(f"KILLED {ticker}: {len(kill_reasons)} kill filter(s) triggered")
        for reason in kill_reasons:
            logger.info(f"  -> {reason}")

    return {
        "passed": not killed,
        "killed": killed,
        "kill_reasons": kill_reasons,
        "filters_run": total_filters,
        "filters_failed": len(kill_reasons),
    }
