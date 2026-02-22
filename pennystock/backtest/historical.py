"""
Historical backtesting engine.

Simulates running the pick algorithm on a past date, then checks what
actually happened to those picks over the following days/weeks.

Usage:
    from pennystock.backtest.historical import run_historical_backtest
    result = run_historical_backtest("2025-08-01")

LIMITATIONS (clearly noted in output):
- Survivorship bias: Only tests stocks in the CURRENT Finviz universe.
  Stocks delisted between the target date and now are invisible.
- Fundamental data (insider %, float, SI, revenue, P/B) uses CURRENT
  Yahoo values as a proxy.  These may have changed since the target date.
- News/sentiment: Not available historically.  Catalyst gets a neutral
  score (50); sentiment is skipped entirely.
- Penalties (going concern, delisting notice, etc.) use current data.

What IS historically accurate:
- Price & volume (OHLCV) -- all from Yahoo Finance history
- Technical indicators computed on historical data only
- Price-based kill filters (already pumped, recent spike, pump-and-dump)
- 52-week high/low position computed from historical price range
"""

import time
from datetime import datetime

import pandas as pd
from loguru import logger

from pennystock.config import (
    ALGORITHM_VERSION, WEIGHTS, MIN_PRICE, MAX_PRICE, MIN_VOLUME,
    MIN_RECOMMENDATION_SCORE,
    PRE_PUMP_HIGH_CONVICTION_BONUS, PRE_PUMP_MEDIUM_CONVICTION_BONUS,
    KILL_ALREADY_PUMPED_PCT, KILL_ALREADY_PUMPED_DAYS,
    KILL_RECENT_SPIKE_PCT, KILL_RECENT_SPIKE_DAYS,
    KILL_PUMP_DUMP_SPIKE_RATIO, KILL_PUMP_DUMP_SPIKE_WINDOW,
    KILL_PUMP_DUMP_DECLINE_PCT,
    BACKTEST_WINNER_THRESHOLD, BACKTEST_STOP_LOSS_PCT,
)
from pennystock.data.finviz_client import get_penny_stocks
from pennystock.data.yahoo_client import get_stock_info, get_batch_history
from pennystock.analysis.technical import (
    extract_features as extract_tech_features,
    analyze as analyze_technical,
)
from pennystock.analysis.fundamental import (
    score_setup, score_fundamentals,
)
from pennystock.analysis.pre_pump import score_pre_pump
from pennystock.algorithm import load_algorithm


# ────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ────────────────────────────────────────────────────────────────────

def run_historical_backtest(
    target_date_str: str,
    hold_days: list = None,
    top_n: int = 5,
    progress_callback=None,
) -> dict:
    """
    Simulate running the algorithm on a historical date and evaluate
    forward returns for the picks it would have made.

    Args:
        target_date_str: ISO date string, e.g. '2025-08-01'.
        hold_days: List of trading-day horizons to evaluate (default [5, 10, 14]).
        top_n: Number of top picks to return.
        progress_callback: Callable for GUI progress updates.

    Returns:
        Dict with picks, forward returns, and summary statistics.
    """
    from pennystock.config import BACKTEST_HOLD_DAYS
    hold_days = hold_days or BACKTEST_HOLD_DAYS
    target = pd.Timestamp(target_date_str)
    start_time = time.time()
    lines = []

    def _log(msg=""):
        logger.info(msg)
        lines.append(msg)
        if progress_callback:
            progress_callback(msg)

    _log(f"{'=' * 70}")
    _log(f"  HISTORICAL BACKTEST  |  v{ALGORITHM_VERSION}")
    _log(f"  Target Date: {target_date_str}  |  Hold: {hold_days} trading days")
    _log(f"{'=' * 70}")

    # ── 1. Load learned algorithm ──────────────────────────────────
    algorithm = load_algorithm()
    if not algorithm:
        _log("ERROR: No algorithm.json found. Run 'Build Algorithm' first.")
        return None
    tech_factors = [f for f in algorithm["factors"] if f["category"] == "technical"]
    _log(f"  Loaded algorithm ({len(algorithm['factors'])} factors)")

    # ── 2. Get current stock universe ───────────────────────────────
    _log("\n  Fetching stock universe from Finviz...")
    stocks = get_penny_stocks()
    _log(f"  Found {len(stocks)} stocks in universe")

    # ── 3. Download extended price history ─────────────────────────
    _log(f"  Downloading 2-year price history (this takes a minute)...")
    tickers = [s["ticker"] for s in stocks]
    all_hist = get_batch_history(tickers, period="2y")
    _log(f"  Got history for {len(all_hist)}/{len(stocks)} stocks")

    # ── 4. Filter to penny stocks AS OF target date ────────────────
    _log(f"\n  Filtering to penny stocks as of {target_date_str}...")
    candidates = _filter_historical_candidates(stocks, all_hist, target)
    _log(f"  Found {len(candidates)} penny stocks on {target_date_str}")

    if not candidates:
        _log("  No candidates found for this date. Is the date a trading day?")
        return _empty_result(target_date_str, hold_days, lines)

    # ── 5. Stage 1: Quick technical screen ─────────────────────────
    _log(f"\n  Stage 1: Technical screening {len(candidates)} candidates...")
    for c in candidates:
        features = extract_tech_features(c["hist_before"])
        c["tech_score"] = _score_features(features, tech_factors) if features else 0

    candidates.sort(key=lambda x: x["tech_score"], reverse=True)
    top_50 = candidates[:50]
    _log(f"  Top 50 tech scores: {top_50[0]['tech_score']:.1f} to {top_50[-1]['tech_score']:.1f}")

    # ── 6. Stage 2: Deep analysis ─────────────────────────────────
    _log(f"\n  Stage 2: Deep analysis on top 50...")
    scored_picks = []
    killed_count = 0

    for j, c in enumerate(top_50):
        ticker = c["ticker"]
        try:
            # A. Price-based kill filters (100% historical)
            killed, reason = _historical_kill_check(c["hist_before"])
            if killed:
                killed_count += 1
                continue

            # B. Technical analysis (100% historical)
            tech = analyze_technical(c["hist_before"])
            tech_score = tech.get("score", 50) if tech.get("valid") else 50

            # C. Setup score (current fundamentals -- limitation noted)
            info = get_stock_info(ticker)

            setup_result = score_setup(ticker, info=info)
            setup_score = setup_result.get("score", 50)

            # Override proximity-to-low with HISTORICAL 52w range
            hist_52w = c["hist_before"].iloc[-252:] if len(c["hist_before"]) >= 252 else c["hist_before"]
            h52 = float(hist_52w["High"].max()) if "High" in hist_52w.columns else float(hist_52w["Close"].max())
            l52 = float(hist_52w["Low"].min()) if "Low" in hist_52w.columns else float(hist_52w["Close"].min())
            if h52 > l52 > 0:
                hist_pos = (c["entry_price"] - l52) / (h52 - l52)
                # Re-score proximity: near low = high score
                if hist_pos <= 0.10:
                    prox_score = 95
                elif hist_pos <= 0.25:
                    prox_score = 80
                elif hist_pos <= 0.40:
                    prox_score = 60
                elif hist_pos <= 0.60:
                    prox_score = 40
                else:
                    prox_score = 15
                # Blend: replace proximity sub-score in setup
                # Proximity is 25% of setup, so adjust accordingly
                orig_prox = setup_result.get("proximity_to_low", {}).get("score", 50)
                setup_score = setup_score + (prox_score - orig_prox) * 0.25

            # D. Pre-pump signals (partial -- SI uses current data)
            pp_features = {
                "multiday_unusual_vol_days": (
                    tech.get("multiday_unusual_volume", {}).get("unusual_days", 0)
                    if tech.get("valid") else 0
                ),
            }
            pp = score_pre_pump(ticker, info=info, tech_features=pp_features)
            pre_pump_score = pp["score"]

            # E. Fundamental score (current data -- limitation)
            fund = score_fundamentals(ticker, info=info)
            fund_score = fund.get("score", 50)

            # F. Catalyst (neutral for backtest -- no historical news)
            cat_score = 50

            # G. Composite score
            base_score = (
                setup_score * WEIGHTS["setup"]
                + tech_score * WEIGHTS["technical"]
                + pre_pump_score * WEIGHTS["pre_pump"]
                + fund_score * WEIGHTS["fundamental"]
                + cat_score * WEIGHTS["catalyst"]
            )

            pp_conf = pp.get("confidence", "LOW")
            if pp_conf == "HIGH":
                bonus = PRE_PUMP_HIGH_CONVICTION_BONUS
            elif pp_conf == "MEDIUM":
                bonus = PRE_PUMP_MEDIUM_CONVICTION_BONUS
            else:
                bonus = 0

            final_score = max(0, min(100, base_score + bonus))
            confidence = "LOW" if final_score < 50 else "MEDIUM" if final_score < 65 else "HIGH"

            if final_score < MIN_RECOMMENDATION_SCORE:
                continue

            # H. Forward returns (the whole point!)
            forward = _compute_forward_returns(
                c["hist_full"], c["entry_date"], c["entry_price"], hold_days,
            )

            scored_picks.append({
                "ticker": ticker,
                "company": c.get("company", ""),
                "entry_price": round(c["entry_price"], 4),
                "entry_date": str(c["entry_date"].date()),
                "score": round(final_score, 1),
                "confidence": confidence,
                "sub_scores": {
                    "setup": round(setup_score, 1),
                    "technical": round(tech_score, 1),
                    "pre_pump": round(pre_pump_score, 1),
                    "fundamental": round(fund_score, 1),
                    "catalyst": round(cat_score, 1),
                },
                "pre_pump_confidence": pp_conf,
                "pre_pump_confluence": pp.get("confluence_count", 0),
                "forward_returns": forward,
            })

            _log(f"    {ticker}: {final_score:.1f} ({confidence})")

        except Exception as e:
            logger.debug(f"  Backtest error for {ticker}: {e}")
            continue

    _log(f"\n  Stage 2: {killed_count} killed, {len(scored_picks)} scored above {MIN_RECOMMENDATION_SCORE}pts")

    # ── 7. Rank and build results ──────────────────────────────────
    scored_picks.sort(key=lambda x: x["score"], reverse=True)
    top_picks = scored_picks[:top_n]

    summary = _compute_summary(top_picks, scored_picks, hold_days)

    # ── 8. Print results ───────────────────────────────────────────
    _print_results(_log, top_picks, summary, target_date_str, hold_days)

    elapsed = time.time() - start_time
    _log(f"\n{'=' * 70}")
    _log(f"  Backtest completed in {elapsed:.0f}s")
    _log(f"{'=' * 70}\n")

    result = {
        "target_date": target_date_str,
        "hold_days": hold_days,
        "universe_size": len(stocks),
        "candidates_on_date": len(candidates),
        "killed": killed_count,
        "total_scored": len(scored_picks),
        "picks": top_picks,
        "all_scored": scored_picks,
        "summary": summary,
        "report_lines": lines,
    }

    _save_report(target_date_str, lines)
    return result


# ────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────

def _filter_historical_candidates(stocks, all_hist, target):
    """Filter to stocks that were penny stocks on the target date."""
    candidates = []
    for stock in stocks:
        ticker = stock["ticker"]
        hist = all_hist.get(ticker)
        if hist is None or hist.empty:
            continue

        hist_before = hist[hist.index <= target]
        if len(hist_before) < 30:
            continue

        entry_price = float(hist_before["Close"].iloc[-1])
        entry_date = hist_before.index[-1]

        if entry_price < MIN_PRICE or entry_price > MAX_PRICE:
            continue

        # Average volume check
        vol_window = hist_before["Volume"].iloc[-20:] if len(hist_before) >= 20 else hist_before["Volume"]
        avg_vol = float(vol_window.mean())
        if avg_vol < MIN_VOLUME:
            continue

        candidates.append({
            "ticker": ticker,
            "company": stock.get("company", ""),
            "entry_price": entry_price,
            "entry_date": entry_date,
            "hist_before": hist_before,
            "hist_full": hist,
        })
    return candidates


def _historical_kill_check(hist_before):
    """Run price-based kill filters on historical data."""
    if hist_before is None or hist_before.empty:
        return True, "No data"

    close = hist_before["Close"]
    current_price = float(close.iloc[-1])

    # Kill 1: Already Pumped (>100% in last 5 days)
    days = int(KILL_ALREADY_PUMPED_DAYS)
    if len(close) > days + 1:
        price_ago = float(close.iloc[-(days + 1)])
        if price_ago > 0:
            gain = ((current_price - price_ago) / price_ago) * 100
            if gain > KILL_ALREADY_PUMPED_PCT:
                return True, f"Already pumped {gain:.0f}% in {days}d"

    # Kill 2: Recent Spike (high/low range > 60% in 40 days)
    spike_days = int(KILL_RECENT_SPIKE_DAYS)
    if len(hist_before) >= spike_days:
        recent = hist_before.iloc[-spike_days:]
    else:
        recent = hist_before

    if "High" in recent.columns and "Low" in recent.columns:
        max_high = float(recent["High"].max())
        min_low = float(recent["Low"].min())
    else:
        max_high = float(recent["Close"].max())
        min_low = float(recent["Close"].min())

    if min_low > 0:
        spike_pct = ((max_high - min_low) / min_low) * 100
        if spike_pct > KILL_RECENT_SPIKE_PCT:
            return True, f"Recent spike {spike_pct:.0f}% in {spike_days}d"

    # Kill 3: Pump-and-dump aftermath
    window = int(KILL_PUMP_DUMP_SPIKE_WINDOW)
    if len(hist_before) > window + 10:
        lookback = hist_before.iloc[-(window + 10):]
        lk_close = lookback["Close"]
        peak = float(lk_close.max())
        base_idx = lk_close.idxmin()
        base = float(lk_close[base_idx])

        if base > 0 and peak / base >= KILL_PUMP_DUMP_SPIKE_RATIO:
            if current_price < peak * KILL_PUMP_DUMP_DECLINE_PCT:
                return True, "Pump-and-dump aftermath"

    return False, ""


def _compute_forward_returns(hist_full, entry_date, entry_price, hold_days):
    """
    Compute returns at each hold-day horizon after entry.
    Also tracks:
    - peak_return_pct: best return seen up to that horizon
    - peak_day: trading day when peak occurred
    - stop_loss_triggered: whether stop-loss was hit before the horizon
    - stop_loss_day: which day it triggered (None if not triggered)
    - return_with_sl: return if using stop-loss (capped at stop-loss level)
    """
    hist_after = hist_full[hist_full.index > entry_date]
    forward = {}
    max_horizon = max(hold_days) if hold_days else 14

    # Pre-compute daily returns and track peak/stop-loss across full period
    stop_loss_day = None
    stop_loss_pct = BACKTEST_STOP_LOSS_PCT  # e.g. -15.0

    for days in hold_days:
        key = f"{days}d"
        if len(hist_after) >= days:
            exit_price = float(hist_after["Close"].iloc[days - 1])
            ret = ((exit_price - entry_price) / entry_price) * 100

            # Track peak (best close) within this horizon
            closes_in_window = hist_after["Close"].iloc[:days]
            highs_in_window = hist_after["High"].iloc[:days] if "High" in hist_after.columns else closes_in_window
            peak_price = float(highs_in_window.max())
            peak_return = ((peak_price - entry_price) / entry_price) * 100

            # Find which day the peak occurred
            peak_idx = highs_in_window.idxmax()
            peak_day_num = list(hist_after.index[:days]).index(peak_idx) + 1

            # Check if stop-loss was triggered within this window
            sl_triggered = False
            sl_day = None
            return_with_sl = ret  # default: same as actual

            if stop_loss_pct < 0:
                lows_in_window = hist_after["Low"].iloc[:days] if "Low" in hist_after.columns else closes_in_window
                for d_idx in range(days):
                    day_low = float(lows_in_window.iloc[d_idx])
                    day_ret = ((day_low - entry_price) / entry_price) * 100
                    if day_ret <= stop_loss_pct:
                        sl_triggered = True
                        sl_day = d_idx + 1
                        return_with_sl = stop_loss_pct  # sold at stop
                        break

            forward[key] = {
                "return_pct": round(ret, 2),
                "exit_price": round(exit_price, 4),
                "win": ret > 0,
                "peak_return_pct": round(peak_return, 2),
                "peak_day": peak_day_num,
                "stop_loss_triggered": sl_triggered,
                "stop_loss_day": sl_day,
                "return_with_sl": round(return_with_sl, 2),
            }
        else:
            forward[key] = {
                "return_pct": None, "exit_price": None, "win": None,
                "peak_return_pct": None, "peak_day": None,
                "stop_loss_triggered": None, "stop_loss_day": None,
                "return_with_sl": None,
            }

    return forward


def _compute_summary(top_picks, all_scored, hold_days):
    """Compute win rates and average returns across picks, including stop-loss stats."""
    summary = {}

    for group_name, picks in [("top_picks", top_picks), ("all_scored", all_scored)]:
        for days in hold_days:
            key = f"{days}d"
            returns = [
                p["forward_returns"][key]["return_pct"]
                for p in picks
                if p["forward_returns"].get(key, {}).get("return_pct") is not None
            ]
            sl_returns = [
                p["forward_returns"][key]["return_with_sl"]
                for p in picks
                if p["forward_returns"].get(key, {}).get("return_with_sl") is not None
            ]
            peaks = [
                p["forward_returns"][key]["peak_return_pct"]
                for p in picks
                if p["forward_returns"].get(key, {}).get("peak_return_pct") is not None
            ]
            sl_triggered = [
                p["forward_returns"][key]["stop_loss_triggered"]
                for p in picks
                if p["forward_returns"].get(key, {}).get("stop_loss_triggered") is not None
            ]

            wins = [r for r in returns if r > 0]
            big_wins = [r for r in returns if r >= BACKTEST_WINNER_THRESHOLD * 100]
            sl_wins = [r for r in sl_returns if r > 0]

            if returns:
                summary[f"{group_name}_{key}"] = {
                    "count": len(returns),
                    "win_rate": round(len(wins) / len(returns) * 100, 1),
                    "big_win_rate": round(len(big_wins) / len(returns) * 100, 1),
                    "avg_return": round(sum(returns) / len(returns), 2),
                    "median_return": round(sorted(returns)[len(returns) // 2], 2),
                    "best": round(max(returns), 2),
                    "worst": round(min(returns), 2),
                    # Stop-loss adjusted metrics
                    "sl_win_rate": round(len(sl_wins) / len(sl_returns) * 100, 1) if sl_returns else 0,
                    "sl_avg_return": round(sum(sl_returns) / len(sl_returns), 2) if sl_returns else 0,
                    "sl_worst": round(min(sl_returns), 2) if sl_returns else 0,
                    "sl_triggered_count": sum(1 for s in sl_triggered if s),
                    # Peak return (best possible exit)
                    "avg_peak": round(sum(peaks) / len(peaks), 2) if peaks else 0,
                    "best_peak": round(max(peaks), 2) if peaks else 0,
                }
            else:
                summary[f"{group_name}_{key}"] = {
                    "count": 0, "win_rate": 0, "big_win_rate": 0,
                    "avg_return": 0, "median_return": 0, "best": 0, "worst": 0,
                    "sl_win_rate": 0, "sl_avg_return": 0, "sl_worst": 0,
                    "sl_triggered_count": 0, "avg_peak": 0, "best_peak": 0,
                }

    return summary


def _score_features(stock_features, algorithm_factors):
    """Score features against learned factors (replicated from algorithm.py)."""
    if not algorithm_factors or not stock_features:
        return 50.0

    total_score = 0
    total_weight = 0

    for factor in algorithm_factors:
        name = factor["feature"]
        if name not in stock_features or stock_features[name] is None:
            continue

        value = float(stock_features[name])
        w_mean = factor["winner_mean"]
        l_mean = factor["loser_mean"]
        separation = factor["separation"]

        if separation < 0.1:
            continue

        range_ = abs(w_mean - l_mean)
        if range_ < 0.0001:
            continue

        if factor["direction"] == "higher":
            score = (value - l_mean) / range_
        else:
            score = (l_mean - value) / range_

        score = max(0, min(1, score))
        weight = separation

        total_score += score * weight
        total_weight += weight

    if total_weight > 0:
        return (total_score / total_weight) * 100
    return 50.0


def _print_results(_log, picks, summary, target_date, hold_days):
    """Print formatted backtest results."""
    hold_keys = [f"{d}d" for d in hold_days]

    _log(f"\n{'=' * 70}")
    _log(f"  TOP {len(picks)} PICKS FOR {target_date}")
    _log(f"{'=' * 70}")

    for i, p in enumerate(picks):
        _log(f"\n  #{i+1}. {p['ticker']} - ${p['entry_price']:.4f} "
             f"(Score: {p['score']:.1f}, {p['confidence']})")
        _log(f"      {p.get('company', '')}")
        ss = p["sub_scores"]
        _log(f"      Setup:{ss['setup']:.0f} Tech:{ss['technical']:.0f} "
             f"PrePump:{ss['pre_pump']:.0f} Fund:{ss['fundamental']:.0f}")

        # Forward returns with peak info
        ret_parts = []
        for key in hold_keys:
            fr = p["forward_returns"].get(key, {})
            r = fr.get("return_pct")
            if r is not None:
                marker = "W" if r > 0 else "L"
                peak = fr.get("peak_return_pct", 0)
                sl = fr.get("stop_loss_triggered", False)
                sl_marker = " SL!" if sl else ""
                ret_parts.append(f"{key}:{r:+.1f}%({marker}) pk:{peak:+.1f}%{sl_marker}")
            else:
                ret_parts.append(f"{key}:N/A")
        _log(f"      Forward: {' | '.join(ret_parts)}")

    # Summary stats
    _log(f"\n{'=' * 70}")
    _log(f"  BACKTEST SUMMARY  (Stop-loss: {BACKTEST_STOP_LOSS_PCT}%)")
    _log(f"{'=' * 70}")

    _log(f"\n  Top {len(picks)} Picks Performance:")
    _log(f"  {'Hold':>4}  {'WinRate':>7}  {'AvgRet':>7}  {'Best':>7}  {'Worst':>7}"
         f"  {'AvgPeak':>7}  {'SL WinR':>7}  {'SL Avg':>7}  {'SL#':>3}")
    _log(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*3}")
    for key in hold_keys:
        s = summary.get(f"top_picks_{key}", {})
        if s.get("count", 0) > 0:
            _log(f"  {key:>4}  {s['win_rate']:6.1f}%  {s['avg_return']:+6.1f}%  "
                 f"{s['best']:+6.1f}%  {s['worst']:+6.1f}%  "
                 f"{s['avg_peak']:+6.1f}%  {s['sl_win_rate']:6.1f}%  "
                 f"{s['sl_avg_return']:+6.1f}%  {s['sl_triggered_count']:>3}")

    all_count = summary.get(f"all_scored_{hold_keys[0]}", {}).get("count", 0)
    if all_count > len(picks):
        _log(f"\n  All {all_count} Scored Stocks Performance:")
        _log(f"  {'Hold':>4}  {'WinRate':>7}  {'AvgRet':>7}  {'Best':>7}  {'Worst':>7}"
             f"  {'AvgPeak':>7}  {'SL WinR':>7}  {'SL Avg':>7}  {'SL#':>3}")
        _log(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*3}")
        for key in hold_keys:
            s = summary.get(f"all_scored_{key}", {})
            if s.get("count", 0) > 0:
                _log(f"  {key:>4}  {s['win_rate']:6.1f}%  {s['avg_return']:+6.1f}%  "
                     f"{s['best']:+6.1f}%  {s['worst']:+6.1f}%  "
                     f"{s['avg_peak']:+6.1f}%  {s['sl_win_rate']:6.1f}%  "
                     f"{s['sl_avg_return']:+6.1f}%  {s['sl_triggered_count']:>3}")

    _log(f"\n  INTERPRETATION:")
    _log(f"    - WinRate = % of picks with positive return at that horizon")
    _log(f"    - AvgPeak = average best return seen DURING the hold period")
    _log(f"    - SL WinR/Avg = performance WITH {BACKTEST_STOP_LOSS_PCT}% stop-loss active")
    _log(f"    - SL# = how many picks hit the stop-loss")
    _log(f"    - Compare AvgRet vs SL Avg to see if stop-loss helps")

    _log(f"\n  LIMITATIONS:")
    _log(f"    - Survivorship bias: only tests stocks in current Finviz universe")
    _log(f"    - Fundamentals (insider%, float, SI) use CURRENT data, not historical")
    _log(f"    - Catalyst score set to neutral (50) -- no historical news data")
    _log(f"    - Going concern / delisting penalties not applied")


def _save_report(target_date_str, lines):
    """Save backtest report to runs/."""
    import os
    os.makedirs("runs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("runs", f"backtest_{target_date_str}_{timestamp}_v{ALGORITHM_VERSION}.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        print(f"\n  Backtest report saved to {path}")
    except Exception as e:
        logger.debug(f"Failed to save backtest report: {e}")


def _empty_result(target_date_str, hold_days, lines):
    return {
        "target_date": target_date_str,
        "hold_days": hold_days,
        "universe_size": 0,
        "candidates_on_date": 0,
        "killed": 0,
        "total_scored": 0,
        "picks": [],
        "all_scored": [],
        "summary": {},
        "report_lines": lines,
    }
