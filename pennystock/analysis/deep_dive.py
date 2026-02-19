"""
Deep Dive: Comprehensive single-stock analysis.

Produces a Yahoo Finance-style report with every data point the algorithm
uses, plus raw social media comments, full technical breakdown, and
pre-pump signal analysis.  Designed for manual review of picks before
placing a trade.

Usage:
    from pennystock.analysis.deep_dive import run_deep_dive
    report = run_deep_dive("GETY")   # returns dict + prints formatted report
"""

import time
from datetime import datetime, timedelta

from loguru import logger

from pennystock.config import (
    ALGORITHM_VERSION, WEIGHTS, MIN_RECOMMENDATION_SCORE,
    PRE_PUMP_HIGH_CONVICTION_BONUS, PRE_PUMP_MEDIUM_CONVICTION_BONUS,
)
from pennystock.data.yahoo_client import (
    get_stock_info, get_news, get_price_history,
)
from pennystock.analysis import technical, fundamental, sentiment, catalyst
from pennystock.analysis.pre_pump import score_pre_pump
from pennystock.analysis.quality_gate import run_kill_filters


# ────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ────────────────────────────────────────────────────────────────────

def run_deep_dive(ticker: str, progress_callback=None) -> dict:
    """
    Run a comprehensive analysis on a single stock and return all data.
    Also prints a formatted report to stdout.

    Args:
        ticker: Stock ticker symbol.
        progress_callback: Optional callable for GUI progress updates.
    """
    ticker = ticker.upper().strip()
    start = time.time()
    lines = []

    def p(msg=""):
        """Print, capture, and notify GUI."""
        print(msg)
        lines.append(msg)
        if progress_callback:
            progress_callback(msg)

    p(f"\n{'═' * 72}")
    p(f"  DEEP DIVE: {ticker}  |  Penny Stock Analyzer v{ALGORITHM_VERSION}")
    p(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"{'═' * 72}")

    # ── 1. Fetch Core Data ───────────────────────────────────────────
    p("\n  Fetching data...")
    info = get_stock_info(ticker)
    news_list = get_news(ticker)
    hist_6m = get_price_history(ticker, period="6mo")
    hist_1y = get_price_history(ticker, period="1y")
    hist_3y = get_price_history(ticker, period="3y")

    price = info.get("price", 0) or 0
    company = info.get("company_name") or info.get("short_name") or ticker

    # ── 2. HEADER: Stock Overview ────────────────────────────────────
    p(f"\n{'─' * 72}")
    p(f"  {company} ({ticker})")
    p(f"{'─' * 72}")
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    country = info.get("country", "N/A")
    p(f"  Sector: {sector}  |  Industry: {industry}  |  Country: {country}")
    p(f"  Price: ${price:.4f}")

    # ── 3. PRICE MOVEMENTS ──────────────────────────────────────────
    p(f"\n{'─' * 72}")
    p(f"  PRICE MOVEMENTS")
    p(f"{'─' * 72}")

    movements = _compute_price_movements(hist_6m, hist_1y, hist_3y)
    _print_price_table(p, movements, price, info)

    # ── 4. COMPANY FUNDAMENTALS ─────────────────────────────────────
    p(f"\n{'─' * 72}")
    p(f"  COMPANY FUNDAMENTALS")
    p(f"{'─' * 72}")
    _print_fundamentals(p, info)

    # ── 5. QUALITY GATE (Kill Filters + Penalties) ──────────────────
    p(f"\n{'─' * 72}")
    p(f"  QUALITY GATE  (12 Kill Filters + 6 Penalty Checks)")
    p(f"{'─' * 72}")
    gate = run_kill_filters(ticker, info=info, news=news_list)
    _print_quality_gate(p, gate)

    # ── 6. PRE-PUMP SIGNALS ─────────────────────────────────────────
    p(f"\n{'─' * 72}")
    p(f"  PRE-PUMP SIGNAL DETECTION  (7 Signals)")
    p(f"{'─' * 72}")
    tech_analysis = technical.analyze(hist_6m) if hist_6m is not None and not hist_6m.empty else {"valid": False}
    pre_pump_features = {
        "multiday_unusual_vol_days": (
            tech_analysis.get("multiday_unusual_volume", {}).get("unusual_days", 0)
            if tech_analysis.get("valid") else 0
        ),
    }
    pp = score_pre_pump(ticker, info=info, tech_features=pre_pump_features)
    _print_pre_pump(p, pp)

    # ── 7. TECHNICAL ANALYSIS ───────────────────────────────────────
    p(f"\n{'─' * 72}")
    p(f"  TECHNICAL ANALYSIS")
    p(f"{'─' * 72}")
    _print_technical(p, tech_analysis)

    # ── 8. SETUP SCORE ──────────────────────────────────────────────
    p(f"\n{'─' * 72}")
    p(f"  SETUP SCORE  (Float, Insider, Proximity, P/B)")
    p(f"{'─' * 72}")
    setup_result = fundamental.score_setup(ticker, info=info)
    _print_setup(p, setup_result)

    # ── 9. FUNDAMENTAL SCORE ────────────────────────────────────────
    p(f"\n{'─' * 72}")
    p(f"  FUNDAMENTAL SCORE  (Revenue, Short Interest, Cash, Squeeze)")
    p(f"{'─' * 72}")
    fund_result = fundamental.score_fundamentals(ticker, info=info)
    _print_fundamental_score(p, fund_result)

    # ── 10. CATALYST ASSESSMENT ─────────────────────────────────────
    p(f"\n{'─' * 72}")
    p(f"  CATALYST ASSESSMENT  (News Analysis)")
    p(f"{'─' * 72}")
    cat_result = catalyst.analyze(ticker)
    _print_catalyst(p, cat_result, news_list)

    # ── 11. SENTIMENT ANALYSIS ──────────────────────────────────────
    p(f"\n{'─' * 72}")
    p(f"  SENTIMENT ANALYSIS  (Reddit + StockTwits + Twitter)")
    p(f"{'─' * 72}")
    sent_result = sentiment.analyze(ticker, company_name=company)
    _print_sentiment(p, sent_result)

    # ── 12. COMPOSITE SCORE ─────────────────────────────────────────
    p(f"\n{'─' * 72}")
    p(f"  COMPOSITE SCORE  (v{ALGORITHM_VERSION} MEGA-ALGORITHM)")
    p(f"{'─' * 72}")

    setup_score = setup_result.get("score", 50)
    tech_score = tech_analysis.get("score", 50) if tech_analysis.get("valid") else 50
    pre_pump_score = pp.get("score", 50)
    fund_score = fund_result.get("score", 50)
    cat_score = cat_result.get("score", 50)

    base_score = (
        setup_score * WEIGHTS["setup"] +
        tech_score * WEIGHTS["technical"] +
        pre_pump_score * WEIGHTS["pre_pump"] +
        fund_score * WEIGHTS["fundamental"] +
        cat_score * WEIGHTS["catalyst"]
    )

    pp_conf = pp.get("confidence", "LOW")
    if pp_conf == "HIGH":
        conviction_bonus = PRE_PUMP_HIGH_CONVICTION_BONUS
    elif pp_conf == "MEDIUM":
        conviction_bonus = PRE_PUMP_MEDIUM_CONVICTION_BONUS
    else:
        conviction_bonus = 0

    penalty = gate.get("total_penalty", 0)
    final_score = max(0, min(100, base_score + conviction_bonus - penalty))
    confidence = "LOW" if final_score < 50 else "MEDIUM" if final_score < 65 else "HIGH"

    p(f"  {'Component':<20} {'Score':>6}  {'Weight':>6}  {'Contribution':>12}")
    p(f"  {'─' * 52}")
    p(f"  {'Setup':<20} {setup_score:>6.1f}  {WEIGHTS['setup']:>5.0%}   {setup_score * WEIGHTS['setup']:>11.1f}")
    p(f"  {'Technical':<20} {tech_score:>6.1f}  {WEIGHTS['technical']:>5.0%}   {tech_score * WEIGHTS['technical']:>11.1f}")
    p(f"  {'Pre-Pump':<20} {pre_pump_score:>6.1f}  {WEIGHTS['pre_pump']:>5.0%}   {pre_pump_score * WEIGHTS['pre_pump']:>11.1f}")
    p(f"  {'Fundamental':<20} {fund_score:>6.1f}  {WEIGHTS['fundamental']:>5.0%}   {fund_score * WEIGHTS['fundamental']:>11.1f}")
    p(f"  {'Catalyst':<20} {cat_score:>6.1f}  {WEIGHTS['catalyst']:>5.0%}   {cat_score * WEIGHTS['catalyst']:>11.1f}")
    p(f"  {'─' * 52}")
    p(f"  {'Base Score':<20} {'':>6}  {'':>6}  {base_score:>11.1f}")
    if conviction_bonus > 0:
        p(f"  {'Conviction Bonus':<20} {'':>6}  {'':>6}  {'+' + str(conviction_bonus):>11}")
    if penalty > 0:
        p(f"  {'Penalties':<20} {'':>6}  {'':>6}  {'-' + str(penalty):>11}")
    p(f"  {'─' * 52}")
    p(f"  {'FINAL SCORE':<20} {'':>6}  {'':>6}  {final_score:>11.1f}")
    p(f"  {'CONFIDENCE':<20} {'':>6}  {'':>6}  {confidence:>11}")
    p()
    if gate.get("killed"):
        p(f"  *** KILLED BY QUALITY FILTER -- DO NOT BUY ***")
    elif final_score < MIN_RECOMMENDATION_SCORE:
        p(f"  *** BELOW MINIMUM THRESHOLD ({MIN_RECOMMENDATION_SCORE}pts) -- NOT RECOMMENDED ***")
    elif confidence == "HIGH":
        p(f"  *** STRONG BUY SIGNAL -- Multiple independent signals align ***")
    elif confidence == "MEDIUM":
        p(f"  *** MODERATE BUY SIGNAL -- Setup is interesting but not overwhelming ***")
    else:
        p(f"  *** WEAK SIGNAL -- Consider passing or waiting for better entry ***")

    # ── 13. RISK MANAGEMENT ─────────────────────────────────────────
    p(f"\n{'─' * 72}")
    p(f"  RISK MANAGEMENT")
    p(f"{'─' * 72}")
    atr = tech_analysis.get("atr", 0) if tech_analysis.get("valid") else 0
    support = tech_analysis.get("support_resistance", {}).get("support") if tech_analysis.get("valid") else None
    _print_risk(p, price, atr, support)

    # ── Footer ──────────────────────────────────────────────────────
    elapsed = time.time() - start
    p(f"\n{'═' * 72}")
    p(f"  Analysis completed in {elapsed:.1f}s")
    p(f"{'═' * 72}\n")

    # Build result dict
    result = {
        "ticker": ticker,
        "company": company,
        "price": price,
        "final_score": final_score,
        "confidence": confidence,
        "killed": gate.get("killed", False),
        "sub_scores": {
            "setup": setup_score,
            "technical": tech_score,
            "pre_pump": pre_pump_score,
            "fundamental": fund_score,
            "catalyst": cat_score,
        },
        "quality_gate": gate,
        "pre_pump": pp,
        "technical": tech_analysis,
        "setup": setup_result,
        "fundamentals": fund_result,
        "catalyst": cat_result,
        "sentiment": sent_result,
        "info": info,
        "price_movements": movements,
        "report_lines": lines,
    }

    # Save to runs/
    _save_report(ticker, lines)

    return result


# ────────────────────────────────────────────────────────────────────
# PRICE MOVEMENT HELPERS
# ────────────────────────────────────────────────────────────────────

def _compute_price_movements(hist_6m, hist_1y, hist_3y) -> dict:
    """Compute price changes over multiple timeframes."""
    movements = {}

    def _pct_change(hist, days_back):
        if hist is None or hist.empty:
            return None
        close = hist["Close"]
        if len(close) < days_back + 1:
            return None
        price_now = float(close.iloc[-1])
        price_then = float(close.iloc[-(days_back + 1)])
        if price_then <= 0:
            return None
        return ((price_now - price_then) / price_then) * 100

    def _high_low(hist, days_back):
        if hist is None or hist.empty:
            return None, None
        if len(hist) < days_back:
            recent = hist
        else:
            recent = hist.iloc[-days_back:]
        high = float(recent["High"].max()) if "High" in recent.columns else float(recent["Close"].max())
        low = float(recent["Low"].min()) if "Low" in recent.columns else float(recent["Close"].min())
        return high, low

    # 1-day
    movements["1d"] = _pct_change(hist_6m, 1)
    # 5-day (1 week)
    movements["1w"] = _pct_change(hist_6m, 5)
    # 1 month (~21 trading days)
    movements["1m"] = _pct_change(hist_6m, 21)
    # 3 months (~63 trading days)
    movements["3m"] = _pct_change(hist_6m, 63)
    # 6 months (~126 trading days)
    movements["6m"] = _pct_change(hist_6m, 126)
    # 1 year (~252 trading days)
    movements["1y"] = _pct_change(hist_1y, 252)
    # 3 year
    if hist_3y is not None and not hist_3y.empty and len(hist_3y) > 252:
        movements["3y"] = _pct_change(hist_3y, min(756, len(hist_3y) - 1))
    else:
        movements["3y"] = None

    # High/low ranges
    for label, hist, days in [
        ("1w", hist_6m, 5), ("1m", hist_6m, 21), ("3m", hist_6m, 63),
        ("6m", hist_6m, 126), ("1y", hist_1y, 252),
    ]:
        h, l = _high_low(hist, days)
        movements[f"{label}_high"] = h
        movements[f"{label}_low"] = l

    return movements


def _print_price_table(p, movements, price, info):
    """Print price movement table."""
    p(f"\n  {'Period':<10} {'Change':>10}  {'High':>10}  {'Low':>10}")
    p(f"  {'─' * 44}")

    for label, key in [("1 Day", "1d"), ("1 Week", "1w"), ("1 Month", "1m"),
                        ("3 Month", "3m"), ("6 Month", "6m"), ("1 Year", "1y"),
                        ("3 Year", "3y")]:
        change = movements.get(key)
        change_str = f"{change:+.1f}%" if change is not None else "N/A"
        high = movements.get(f"{key}_high")
        low = movements.get(f"{key}_low")
        high_str = f"${high:.4f}" if high else "N/A"
        low_str = f"${low:.4f}" if low else "N/A"
        p(f"  {label:<10} {change_str:>10}  {high_str:>10}  {low_str:>10}")

    # 52-week data from Yahoo
    h52 = info.get("52w_high", 0)
    l52 = info.get("52w_low", 0)
    if h52 and l52 and h52 > 0:
        pos = (price - l52) / (h52 - l52) if h52 != l52 else 0
        p(f"\n  52-Week Range: ${l52:.4f} - ${h52:.4f}")
        p(f"  52-Week Position: {pos:.1%} (0%=at low, 100%=at high)")

    # Volume
    avg_vol = info.get("avg_volume", 0) or 0
    avg_vol_10d = info.get("avg_volume_10d", 0) or 0
    p(f"\n  Avg Volume (50d): {avg_vol:,.0f}")
    p(f"  Avg Volume (10d): {avg_vol_10d:,.0f}")
    if avg_vol > 0 and avg_vol_10d > 0:
        vol_ratio = avg_vol_10d / avg_vol
        p(f"  Volume Trend: {vol_ratio:.2f}x {'(increasing)' if vol_ratio > 1.2 else '(decreasing)' if vol_ratio < 0.8 else '(stable)'}")


def _print_fundamentals(p, info):
    """Print company fundamental data."""
    def fmt(val, prefix="", suffix="", div=1):
        if val is None or val == 0:
            return "N/A"
        return f"{prefix}{val/div:,.2f}{suffix}"

    def fmt_int(val, prefix="", suffix=""):
        if val is None or val == 0:
            return "N/A"
        return f"{prefix}{val:,.0f}{suffix}"

    def fmt_pct(val):
        if val is None:
            return "N/A"
        return f"{val*100:.1f}%"

    mc = info.get("market_cap", 0) or 0
    p(f"\n  Market Cap:          {fmt(mc, '$', '', 1e6)}M" if mc > 1e6 else f"\n  Market Cap:          {fmt_int(mc, '$')}")
    p(f"  Shares Outstanding:  {fmt_int(info.get('shares_outstanding'))}")
    p(f"  Float Shares:        {fmt_int(info.get('float_shares'))}")
    so = info.get("shares_outstanding", 0) or 1
    fl = info.get("float_shares", 0) or 0
    p(f"  Float Ratio:         {fl/so:.1%}" if so > 0 and fl > 0 else "  Float Ratio:         N/A")
    p(f"  Insider Ownership:   {fmt_pct(info.get('insider_percent_held'))}")
    p(f"  Institutional Own:   {fmt_pct(info.get('institution_percent_held'))}")
    p(f"  Employees:           {fmt_int(info.get('full_time_employees'))}")

    p(f"\n  Revenue (TTM):       {fmt(info.get('total_revenue'), '$', '', 1e6)}M" if (info.get('total_revenue') or 0) > 1e6
      else f"\n  Revenue (TTM):       {fmt_int(info.get('total_revenue'), '$')}")
    p(f"  Revenue Growth:      {fmt_pct(info.get('revenue_growth'))}")
    p(f"  Earnings Growth:     {fmt_pct(info.get('earnings_growth'))}")
    p(f"  Gross Margins:       {fmt_pct(info.get('gross_margins'))}")
    p(f"  Profit Margins:      {fmt_pct(info.get('profit_margins'))}")
    p(f"  Operating Margins:   {fmt_pct(info.get('operating_margins'))}")

    p(f"\n  Total Cash:          {fmt(info.get('total_cash'), '$', '', 1e6)}M" if (info.get('total_cash') or 0) > 1e6
      else f"\n  Total Cash:          {fmt_int(info.get('total_cash'), '$')}")
    p(f"  Total Debt:          {fmt(info.get('total_debt'), '$', '', 1e6)}M" if (info.get('total_debt') or 0) > 1e6
      else f"\n  Total Debt:          {fmt_int(info.get('total_debt'), '$')}")
    p(f"  Operating Cashflow:  {fmt(info.get('operating_cashflow'), '$', '', 1e6)}M" if abs(info.get('operating_cashflow') or 0) > 1e6
      else f"  Operating Cashflow:  {fmt_int(info.get('operating_cashflow'), '$')}")
    p(f"  Free Cash Flow:      {fmt(info.get('free_cashflow'), '$', '', 1e6)}M" if abs(info.get('free_cashflow') or 0) > 1e6
      else f"  Free Cash Flow:      {fmt_int(info.get('free_cashflow'), '$')}")
    p(f"  Current Ratio:       {info.get('current_ratio', 'N/A')}")
    p(f"  Debt/Equity:         {info.get('debt_to_equity', 'N/A')}")
    p(f"  Book Value/Share:    {fmt(info.get('book_value'), '$')}")
    p(f"  Price/Book:          {info.get('price_to_book', 'N/A')}")

    p(f"\n  Short Interest:")
    p(f"    Shares Short:      {fmt_int(info.get('shares_short'))}")
    p(f"    Shares Short Prior:{fmt_int(info.get('shares_short_prior'))}")
    si_chg = None
    ss = info.get("shares_short", 0) or 0
    ssp = info.get("shares_short_prior", 0) or 0
    if ssp > 0:
        si_chg = ((ss - ssp) / ssp) * 100
        p(f"    SI Change:         {si_chg:+.1f}%")
    p(f"    Short % of Float:  {fmt_pct(info.get('short_percent_of_float'))}")
    p(f"    Short Ratio (DTC): {info.get('short_ratio', 'N/A')}")


def _print_quality_gate(p, gate):
    """Print quality gate results."""
    if gate["killed"]:
        p(f"\n  *** KILLED *** ({gate['filters_failed']} filter(s) triggered)")
        for reason in gate["kill_reasons"]:
            p(f"    X  {reason[:100]}")
    else:
        p(f"\n  PASSED all {gate['filters_run']} kill filters")

    if gate["penalties"]:
        p(f"\n  Penalties ({gate['total_penalty']}pts total deduction):")
        for pen in gate["penalties"]:
            p(f"    -  {pen[:100]}")
    else:
        p(f"  No penalties applied")


def _print_pre_pump(p, pp):
    """Print pre-pump signal analysis."""
    p(f"\n  Overall: {pp['score']:.1f}/100  |  "
      f"Confluence: {pp['confluence_count']}/{pp['total_signals']} bullish  |  "
      f"Confidence: {pp['confidence']}")
    p()
    p(f"  {'Signal':<25} {'Score':>5}  {'Status':<20}  Details")
    p(f"  {'─' * 68}")

    for name, sig in pp.get("signals", {}).items():
        score = sig.get("score", 0)
        signal = sig.get("signal", "")
        bullish = score >= 65
        status = "BULLISH" if bullish else "neutral" if score >= 50 else "bearish"
        marker = ">>>" if bullish else "   "

        # Build detail string based on signal type
        detail = ""
        if name == "short_interest_change":
            detail = f"SI change: {sig.get('change_pct', 0):+.1f}% ({sig.get('shares_short', 0):,} vs {sig.get('shares_short_prior', 0):,})"
        elif name == "float_rotation":
            detail = f"rotation={sig.get('rotation', 0):.3f} (vol={sig.get('volume', 0):,.0f} / float={sig.get('float', 0):,.0f})"
        elif name == "compliance_risk":
            detail = f"price=${sig.get('price', 0):.4f} below_$1={sig.get('below_dollar', False)}"
        elif name == "volume_acceleration":
            detail = f"10d/avg ratio={sig.get('ratio', 0):.2f}x unusual_days={sig.get('multiday_unusual', 0)}"
        elif name == "supply_lock":
            detail = f"insider={sig.get('insider_pct', 0):.1f}% float={sig.get('float_shares', 0):,.0f}"
        elif name == "squeeze_setup":
            detail = f"SI%={sig.get('short_pct_float', 0):.1f}% DTC={sig.get('days_to_cover', 0):.1f}"
        elif name == "beaten_down":
            detail = f"52w pos={sig.get('position_52w', 0):.1%} (${sig.get('52w_low', 0):.2f}-${sig.get('52w_high', 0):.2f})"

        p(f"  {marker}{name:<22} {score:>5}  {status:<20}  {detail}")


def _print_technical(p, tech):
    """Print technical analysis breakdown."""
    if not tech.get("valid"):
        p(f"\n  Insufficient data for technical analysis")
        p(f"  Reason: {tech.get('reason', 'unknown')}")
        return

    p(f"\n  Overall Technical Score: {tech.get('score', 0):.1f}/100")

    # Core indicators
    p(f"\n  Momentum Indicators:")
    p(f"    RSI (14):          {tech.get('rsi', 'N/A'):.1f}" if tech.get('rsi') else "    RSI (14):          N/A")
    p(f"    StochRSI:          {tech.get('stochrsi', 'N/A'):.1f}" if tech.get('stochrsi') is not None else "    StochRSI:          N/A")
    p(f"    MACD Histogram:    {tech.get('macd_histogram', 'N/A'):.4f}" if tech.get('macd_histogram') is not None else "    MACD Histogram:    N/A")
    p(f"    Stochastic %K:     {tech.get('stochastic_k', 'N/A'):.1f}" if tech.get('stochastic_k') is not None else "    Stochastic %K:     N/A")
    p(f"    MFI:               {tech.get('mfi', 'N/A'):.1f}" if tech.get('mfi') is not None else "    MFI:               N/A")

    p(f"\n  Trend Indicators:")
    p(f"    ADX:               {tech.get('adx', 'N/A'):.1f}" if tech.get('adx') is not None else "    ADX:               N/A")
    p(f"    +DI / -DI:         {tech.get('plus_di', 0):.1f} / {tech.get('minus_di', 0):.1f}")
    trend_dir = "BULLISH" if (tech.get('plus_di', 0) or 0) > (tech.get('minus_di', 0) or 0) else "BEARISH"
    adx = tech.get('adx', 0) or 0
    trend_strength = "strong" if adx >= 40 else "moderate" if adx >= 25 else "weak"
    p(f"    Trend:             {trend_dir} ({trend_strength}, ADX={adx:.0f})")
    p(f"    OBV Trend:         {tech.get('obv_trend_direction', 'N/A')}")

    p(f"\n  Price Trends:")
    for period in ["5d", "20d"]:
        key = f"price_trend_{period}"
        val = tech.get(key)
        p(f"    {period:>3} Change:        {val:+.1f}%" if val is not None else f"    {period:>3} Change:        N/A")

    p(f"\n  Volatility:")
    p(f"    ATR:               {tech.get('atr', 0):.4f}")
    p(f"    Volume Spike:      {tech.get('volume_spike', 0):.2f}x average")
    bb = tech.get("bb_squeeze", {})
    p(f"    BB Squeeze:        {'YES' if bb.get('is_squeeze') else 'No'} (intensity: {bb.get('squeeze_intensity', 0):.3f})")
    p(f"    BB Position:       {tech.get('bb_position', 0):.2f} (0=lower band, 1=upper)")

    # Complex signals
    consol = tech.get("consolidation", {})
    p(f"\n  Patterns:")
    p(f"    Consolidating:     {'YES' if consol.get('is_consolidating') else 'No'} (score: {consol.get('consolidation_score', 0):.0f})")
    mvol = tech.get("multiday_unusual_volume", {})
    p(f"    Unusual Volume:    {mvol.get('unusual_days', 0)} days (accumulating: {'YES' if mvol.get('is_accumulating') else 'No'})")
    vpd = tech.get("volume_price_divergence", {})
    p(f"    Vol/Price Diverg:  {vpd.get('divergence_type', 'N/A')}")
    gap = tech.get("gap", {})
    if gap.get("has_gap"):
        p(f"    Gap:               {gap.get('gap_direction', '')} {gap.get('gap_pct', 0):.1f}%")
    candles = tech.get("candlestick_patterns", {})
    patterns = candles.get("patterns", [])
    if patterns:
        p(f"    Candlestick:       {', '.join(patterns)}")

    # Support/resistance
    sr = tech.get("support_resistance", {})
    p(f"\n  Support/Resistance:")
    p(f"    Support:           ${sr.get('support', 0):.4f} ({sr.get('support_distance_pct', 0):.1f}% away)" if sr.get("support") else "    Support:           N/A")
    p(f"    Resistance:        ${sr.get('resistance', 0):.4f} ({sr.get('resistance_distance_pct', 0):.1f}% away)" if sr.get("resistance") else "    Resistance:        N/A")


def _print_setup(p, setup):
    """Print setup score breakdown."""
    p(f"\n  Overall Setup Score: {setup.get('score', 0):.1f}/100")

    ft = setup.get("float_tightness", {})
    p(f"\n  Float Tightness:     {ft.get('score', 0):.0f}/100")
    p(f"    Float Shares:      {ft.get('float_shares', 0):,.0f}")
    p(f"    Shares Outstanding:{ft.get('shares_outstanding', 0):,.0f}")
    p(f"    Float Ratio:       {ft.get('float_ratio', 0):.1%}")

    io = setup.get("insider_ownership", {})
    p(f"\n  Insider Ownership:   {io.get('score', 0):.0f}/100")
    p(f"    Insider %:         {(io.get('insider_pct', 0) or 0)*100:.1f}%")
    p(f"    Institutional %:   {(io.get('institution_pct', 0) or 0)*100:.1f}%")

    prox = setup.get("proximity_to_low", {})
    p(f"\n  Proximity to Low:    {prox.get('score', 0):.0f}/100")
    p(f"    Position:          {prox.get('position', 0):.1%} (0%=at low, 100%=at high)")
    p(f"    Range:             ${prox.get('52w_low', 0):.4f} - ${prox.get('52w_high', 0):.4f}")

    pb = setup.get("price_to_book", {})
    p(f"\n  Price/Book:          {pb.get('score', 0):.0f}/100")
    p(f"    P/B Ratio:         {pb.get('price_to_book', 'N/A')}")
    p(f"    Book Value:        ${pb.get('book_value', 0):.2f}" if pb.get("book_value") else "    Book Value:        N/A")


def _print_fundamental_score(p, fund):
    """Print fundamental score breakdown."""
    p(f"\n  Overall Fundamental Score: {fund.get('score', 0):.1f}/100")

    rg = fund.get("revenue_growth", {})
    p(f"\n  Revenue Growth:      {rg.get('score', 0):.0f}/100")
    p(f"    YoY Growth:        {(rg.get('revenue_growth') or 0)*100:+.1f}%")
    rev = rg.get("total_revenue", 0) or 0
    p(f"    TTM Revenue:       ${rev/1e6:,.1f}M" if rev > 1e6 else f"    TTM Revenue:       ${rev:,.0f}")

    si = fund.get("short_interest", {})
    p(f"\n  Short Interest:      {si.get('score', 0):.0f}/100")
    p(f"    SI % of Float:     {(si.get('short_pct_float') or 0)*100:.1f}%")
    p(f"    Days to Cover:     {si.get('short_ratio', 0):.1f}")
    p(f"    Shares Short:      {si.get('shares_short', 0):,.0f}")
    p(f"    Prior Period:      {si.get('shares_short_prior', 0):,.0f}")

    cash = fund.get("cash_position", {})
    p(f"\n  Cash Position:       {cash.get('score', 0):.0f}/100")
    tc = cash.get("total_cash", 0) or 0
    p(f"    Cash:              ${tc/1e6:,.1f}M" if tc > 1e6 else f"    Cash:              ${tc:,.0f}")
    td = cash.get("total_debt", 0) or 0
    p(f"    Debt:              ${td/1e6:,.1f}M" if td > 1e6 else f"    Debt:              ${td:,.0f}")
    p(f"    Current Ratio:     {cash.get('current_ratio', 'N/A')}")

    sq = fund.get("squeeze_composite", {})
    p(f"\n  Squeeze Composite:   {sq.get('score', 0):.0f}/100")
    p(f"    Is Squeeze Setup:  {'YES' if sq.get('is_squeeze_setup') else 'No'}")

    dr = fund.get("dilution_risk", {})
    p(f"\n  Dilution Risk:")
    p(f"    Dilution Filings:  {dr.get('dilution_filings_6m', 0)} in last 6 months")
    p(f"    Penalty:           -{dr.get('penalty', 0)}pts" if dr.get("penalty", 0) > 0 else "    Penalty:           None")


def _print_catalyst(p, cat, news_list):
    """Print catalyst assessment and news."""
    p(f"\n  Catalyst Score: {cat.get('score', 50):.0f}/100")
    p(f"  Articles Found: {cat.get('total_articles', 0)}")
    p(f"  Catalyst Density: {cat.get('catalyst_density', 0):.2f}")

    pos = cat.get("positive_catalysts", [])
    neg = cat.get("negative_catalysts", [])
    if pos:
        p(f"\n  Positive Catalysts:")
        for c in pos[:5]:
            p(f"    +  {c[:90]}")
    if neg:
        p(f"\n  Negative Catalysts:")
        for c in neg[:5]:
            p(f"    -  {c[:90]}")

    # Full news list
    if news_list:
        p(f"\n  Recent News ({len(news_list)} articles):")
        for article in news_list[:10]:
            title = article.get("title", "") or ""
            pub = article.get("publisher", "") or ""
            ts = article.get("published", 0)
            if not title:
                continue  # Skip articles with no title
            date_str = ""
            if ts:
                try:
                    date_str = datetime.fromtimestamp(ts).strftime("%m/%d %H:%M")
                except Exception:
                    pass
            p(f"    [{date_str}] {title[:70]}")
            if pub:
                p(f"             via {pub}")
    else:
        p(f"\n  No recent news found.")


def _print_sentiment(p, sent):
    """Print full sentiment analysis with social media data."""
    p(f"\n  Combined Sentiment Score: {sent.get('score', 0):.0f}/100")
    p(f"  Combined Sentiment: {sent.get('combined_sentiment', 0):+.3f} (-1 to +1)")
    p(f"  Total Mentions: {sent.get('total_mentions', 0)}")
    p(f"  Buzz Score: {sent.get('buzz_score', 0):.1f}/100")
    aliases = sent.get("search_aliases", [])
    if aliases:
        p(f"  Search Aliases: {', '.join(aliases)}  (company name matches included)")

    # Reddit
    reddit = sent.get("reddit", {})
    p(f"\n  Reddit ({reddit.get('mentions', 0)} mentions):")
    if reddit.get("mentions", 0) > 0:
        p(f"    Avg Sentiment:   {reddit.get('avg_sentiment', 0):+.3f}")
        p(f"    Positive:        {reddit.get('positive_pct', 0):.0%}")
        sub_counts = reddit.get("subreddit_counts", {})
        if sub_counts:
            p(f"    Subreddits:")
            for sub, count in sorted(sub_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                p(f"      r/{sub}: {count} mention(s)")
        # Print actual post titles/snippets if available
        posts = reddit.get("posts", [])
        if posts:
            p(f"\n    Top Reddit Posts:")
            for post in posts[:8]:
                title = post.get("title", "")[:70]
                score = post.get("sentiment", 0)
                sub = post.get("subreddit", "")
                p(f"      [{score:+.2f}] r/{sub}: {title}")
    else:
        p(f"    No Reddit mentions found")

    # StockTwits
    st = sent.get("stocktwits", {})
    p(f"\n  StockTwits ({st.get('total', 0)} messages):")
    if st.get("total", 0) > 0:
        total = st.get("total", 1)
        bull = st.get("bullish", 0)
        bear = st.get("bearish", 0)
        p(f"    Bullish:         {bull} ({bull/max(1,total):.0%})")
        p(f"    Bearish:         {bear} ({bear/max(1,total):.0%})")
        p(f"    Avg Sentiment:   {st.get('avg_sentiment', 0):+.3f}")
        # Print actual messages if available
        messages = st.get("messages", [])
        if messages:
            p(f"\n    Recent StockTwits:")
            for msg in messages[:8]:
                body = msg.get("body", "")[:70]
                label = msg.get("label", "")
                p(f"      [{label:>7}] {body}")
    else:
        p(f"    No StockTwits data found")

    # Twitter
    tw = sent.get("twitter", {})
    p(f"\n  Twitter ({tw.get('total', 0)} tweets):")
    if tw.get("total", 0) > 0:
        p(f"    Avg Sentiment:   {tw.get('avg_sentiment', 0):+.3f}")
        tweets = tw.get("tweets", [])
        if tweets:
            p(f"\n    Recent Tweets:")
            for tweet in tweets[:8]:
                text = tweet.get("text", "")[:70]
                score = tweet.get("sentiment", 0)
                p(f"      [{score:+.2f}] {text}")
    elif not tw.get("enabled", False):
        p(f"    Twitter analysis disabled (no account configured)")
    else:
        p(f"    No tweets found")


def _print_risk(p, price, atr, support):
    """Print risk management section."""
    if price <= 0:
        p(f"\n  No price data for risk calculations")
        return

    # ATR-based stop
    if atr > 0:
        stop_atr = max(0, price - (2.0 * atr))
        risk_pct = ((price - stop_atr) / price) * 100
        p(f"\n  ATR-Based Stop Loss:")
        p(f"    ATR (14):          ${atr:.4f}")
        p(f"    Stop (2x ATR):     ${stop_atr:.4f}")
        p(f"    Risk:              {risk_pct:.1f}%")

    # Support-based stop
    if support and support > 0:
        stop_support = support * 0.97  # 3% below support
        risk_support = ((price - stop_support) / price) * 100
        p(f"\n  Support-Based Stop:")
        p(f"    Support Level:     ${support:.4f}")
        p(f"    Stop (3% below):   ${stop_support:.4f}")
        p(f"    Risk:              {risk_support:.1f}%")

    # Position sizing guidance
    p(f"\n  Position Sizing (1% account risk):")
    if atr > 0:
        stop = max(0, price - (2.0 * atr))
        risk_per_share = price - stop
        if risk_per_share > 0:
            for acct_size in [1000, 5000, 10000]:
                max_risk = acct_size * 0.01
                shares = int(max_risk / risk_per_share)
                position = shares * price
                p(f"    ${acct_size:>6,} account: {shares:>5,} shares (${position:>8,.2f} position)")


def _save_report(ticker: str, lines: list):
    """Save report to runs/ directory."""
    import os
    os.makedirs("runs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("runs", f"analyze_{ticker}_{timestamp}_v{ALGORITHM_VERSION}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"\n  Report saved to {path}")
