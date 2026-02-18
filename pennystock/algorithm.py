"""
The ONE algorithm for penny stock prediction.

Two-layer architecture:
  LAYER 1: Kill Filters  - Instantly disqualify broken stocks (quality_gate.py)
  LAYER 2: Positive Score - Rank survivors by setup + technical + fundamental + catalyst

Scoring formula (Layer 2):
  final_score = setup(40%) + technical(25%) + fundamental(25%) + catalyst(10%)

  Setup (40%):     float_tightness(35%) + insider_own(25%) + proximity_low(25%) + P/B(15%)
  Technical (25%): RSI(20%) + MACD(20%) + StochRSI(20%) + volume(15%) + OBV(10%) + BB(5%) + trend(10%)
  Fundamental(25%): revenue_growth(40%) + short_interest(30%) + cash_position(30%)
  Catalyst (10%):  news-based positive/negative catalyst detection

Functions:
  build_algorithm()  - Tab 1: Learn what predicts winners from 3 months of data
  pick_stocks()      - Tab 2: Apply the learned algorithm to find today's top 5
"""

import json
import math
import os
import time

import numpy as np
import pandas as pd
from loguru import logger

from pennystock.config import WEIGHTS
from pennystock.data.finviz_client import get_penny_stocks, get_high_gainers
from pennystock.data.yahoo_client import get_price_history, get_stock_info
from pennystock.analysis.technical import extract_features as extract_tech_features
from pennystock.analysis.technical import analyze as analyze_technical
from pennystock.analysis.sentiment import analyze as analyze_sentiment
from pennystock.analysis.sentiment import ensure_bulk_downloaded
from pennystock.analysis.fundamental import extract_features as extract_fund_features
from pennystock.analysis.fundamental import analyze as analyze_fundamental
from pennystock.analysis.fundamental import score_setup, score_fundamentals
from pennystock.analysis.catalyst import analyze as analyze_catalyst
from pennystock.analysis.market_context import analyze as analyze_market
from pennystock.analysis.quality_gate import run_kill_filters
from pennystock.storage.db import Database

ALGORITHM_FILE = "algorithm.json"


# ════════════════════════════════════════════════════════════════════
# TAB 1: BUILD THE ALGORITHM
# ════════════════════════════════════════════════════════════════════

def build_algorithm(progress_callback=None):
    """
    Learn what predicts penny stock winners by analyzing the last 3 months.

    Steps:
      1. Get ALL penny stocks from Finviz ($0.05-$1.00)
      2. Get 3-month price history for each
      3. Classify: WINNER = gained >15% over 2 weeks AND >20% over 4 weeks
      4. Extract technical features for ALL stocks (fast)
      5. Extract sentiment + fundamental features for winners + sample of losers
      6. Statistically compare winners vs losers on every feature
      7. Build one unified scoring algorithm from the most discriminative features
      8. Save algorithm permanently to algorithm.json
    """
    def _log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    start = time.time()
    _log("=" * 60)
    _log("BUILDING ALGORITHM FROM RECENT PENNY STOCK DATA")
    _log("=" * 60)

    # ── Step 1: Get all penny stocks ────────────────────────────────
    _log("Step 1: Discovering penny stocks via Finviz...")
    stocks = get_penny_stocks()
    if not stocks:
        _log("ERROR: No stocks found. Check network connection.")
        return None

    tickers = [s["ticker"] for s in stocks]
    stock_info = {s["ticker"]: s for s in stocks}
    _log(f"  Found {len(tickers)} penny stocks")

    # ── Step 2+3: Get history and classify winners vs losers ────────
    _log("Step 2: Downloading price history and classifying winners...")
    winners = []
    losers = []
    all_tech_features = {}

    for i, ticker in enumerate(tickers):
        try:
            hist = get_price_history(ticker, period="3mo")
            if hist is None or hist.empty or len(hist) < 30:
                continue

            close = hist["Close"]

            # Check 2-week and 4-week gains at MULTIPLE points
            # (not just end-of-period -- a stock could have won and come back)
            is_winner = False
            for offset in range(0, max(1, len(close) - 28), 7):
                end_2w = offset + 10
                end_4w = offset + 20
                if end_4w >= len(close):
                    break
                start_price = close.iloc[offset]
                if start_price <= 0:
                    continue
                gain_2w = (close.iloc[end_2w] - start_price) / start_price * 100
                gain_4w = (close.iloc[end_4w] - start_price) / start_price * 100
                if gain_2w >= 15 and gain_4w >= 20:
                    is_winner = True
                    break

            # Extract technical features
            tech = extract_tech_features(hist)
            if tech:
                tech["ticker"] = ticker
                all_tech_features[ticker] = tech

                if is_winner:
                    winners.append(ticker)
                else:
                    losers.append(ticker)

        except Exception as e:
            logger.debug(f"Failed for {ticker}: {e}")

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            _log(f"  Progress: {i+1}/{len(tickers)} "
                 f"({len(winners)} winners, {len(losers)} losers) "
                 f"~{(len(tickers)-i-1)/rate:.0f}s remaining")

        time.sleep(0.1)

    _log(f"  Classification done: {len(winners)} winners, {len(losers)} losers")

    if len(winners) < 3:
        _log("WARNING: Very few winners found. Algorithm may be unreliable.")
    if not winners:
        _log("ERROR: No winners found. Cannot build algorithm.")
        return None

    # ── Step 4: Compare technical features ──────────────────────────
    _log("Step 3: Analyzing technical patterns...")
    tech_comparison = _compare_groups(
        [all_tech_features[t] for t in winners if t in all_tech_features],
        [all_tech_features[t] for t in losers if t in all_tech_features],
    )
    _log(f"  Technical features analyzed: {len(tech_comparison)} factors")

    # ── Step 5: Sentiment + fundamentals (winners + sample of losers)
    _log("Step 4: Analyzing sentiment & fundamentals (winners + loser sample)...")

    # Pre-download Reddit posts ONCE to avoid per-ticker rate limiting
    _log("  Pre-downloading Reddit posts (bulk mode)...")
    ensure_bulk_downloaded()

    loser_sample = losers[:min(len(losers), len(winners) * 2)]  # 2:1 ratio

    sentiment_features = {"winners": [], "losers": []}
    fund_features = {"winners": [], "losers": []}

    analyze_tickers = [(t, "winners") for t in winners] + [(t, "losers") for t in loser_sample]

    for j, (ticker, group) in enumerate(analyze_tickers):
        try:
            # Sentiment (only include if there's actual data)
            sent = analyze_sentiment(ticker)
            if sent.get("has_data", False):
                sentiment_features[group].append({
                    "reddit_mentions": sent.get("reddit", {}).get("mentions", 0),
                    "reddit_sentiment": sent.get("reddit", {}).get("avg_sentiment", 0),
                    "stocktwits_bullish_pct": (
                        sent.get("stocktwits", {}).get("bullish", 0) /
                        max(1, sent.get("stocktwits", {}).get("total", 1))
                    ),
                    "combined_sentiment": sent.get("combined_sentiment", 0),
                    "buzz_score": sent.get("buzz_score", 0),
                })

            # Fundamentals
            fund = extract_fund_features(ticker)
            fund_features[group].append(fund)

        except Exception as e:
            logger.debug(f"Deep analysis failed for {ticker}: {e}")

        if (j + 1) % 5 == 0:
            _log(f"  Deep analysis: {j+1}/{len(analyze_tickers)}")

        time.sleep(0.3)

    sent_comparison = _compare_groups(
        sentiment_features["winners"], sentiment_features["losers"]
    )
    fund_comparison = _compare_groups(
        fund_features["winners"], fund_features["losers"]
    )

    _log(f"  Sentiment factors: {len(sent_comparison)}")
    _log(f"  Fundamental factors: {len(fund_comparison)}")

    # ── Step 6: Build the unified algorithm ─────────────────────────
    _log("Step 5: Building unified algorithm...")

    all_factors = []
    for factor in tech_comparison:
        factor["category"] = "technical"
        all_factors.append(factor)
    for factor in sent_comparison:
        factor["category"] = "sentiment"
        all_factors.append(factor)
    for factor in fund_comparison:
        factor["category"] = "fundamental"
        all_factors.append(factor)

    # Sort by discrimination power
    all_factors.sort(key=lambda f: f["separation"], reverse=True)

    # Assign category weights based on how discriminative each category is,
    # but with HARD CAPS to prevent pathological results.
    #
    # Bug fix: Previously, sentiment could get 50%+ weight with negative
    # correlation (winners had LOWER sentiment than losers). While partially
    # true (ORKT had zero buzz), letting sentiment dominate is nonsensical.
    #
    # Constraints:
    #   - technical: minimum 30%, maximum 60%
    #   - fundamental: minimum 20%, maximum 50%
    #   - sentiment: minimum 5%, maximum 20%
    WEIGHT_CAPS = {
        "technical":   {"min": 0.30, "max": 0.60},
        "fundamental": {"min": 0.20, "max": 0.50},
        "sentiment":   {"min": 0.05, "max": 0.20},
    }

    cat_separations = {}
    for f in all_factors:
        cat = f["category"]
        cat_separations.setdefault(cat, []).append(f["separation"])

    total_sep = sum(np.mean(v) for v in cat_separations.values()) or 1
    category_weights = {
        cat: np.mean(seps) / total_sep
        for cat, seps in cat_separations.items()
    }

    # Ensure all categories exist
    for cat in ["technical", "sentiment", "fundamental"]:
        if cat not in category_weights:
            category_weights[cat] = WEIGHT_CAPS[cat]["min"]

    # Apply hard caps
    for cat, caps in WEIGHT_CAPS.items():
        if cat in category_weights:
            category_weights[cat] = max(caps["min"], min(caps["max"], category_weights[cat]))

    # Re-normalize to sum to 1.0
    wt = sum(category_weights.values())
    category_weights = {k: round(v / wt, 3) for k, v in category_weights.items()}

    algorithm = {
        "version": "2.0",
        "built_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_summary": {
            "total_stocks": len(winners) + len(losers),
            "winners": len(winners),
            "losers": len(losers),
            "winner_tickers": winners[:20],
            "criteria": "2-week gain >= 15% AND 4-week gain >= 20%",
        },
        "category_weights": category_weights,
        "factors": all_factors,
    }

    # ── Step 7: Save permanently ────────────────────────────────────
    _save_algorithm(algorithm)

    elapsed = time.time() - start
    _log("=" * 60)
    _log(f"ALGORITHM BUILT in {elapsed:.0f}s")
    _log(f"  Winners analyzed: {len(winners)}")
    _log(f"  Losers analyzed: {len(losers)}")
    _log(f"  Total factors: {len(all_factors)}")
    _log(f"  Category weights: {category_weights}")
    _log(f"  Top 5 most predictive factors:")
    for f in all_factors[:5]:
        _log(f"    {f['feature']:25s} ({f['category']:12s}) "
             f"separation={f['separation']:.3f} "
             f"W={f['winner_mean']:.3f} L={f['loser_mean']:.3f}")
    _log(f"  Saved to: {ALGORITHM_FILE}")
    _log("=" * 60)

    # Save to database too
    db = Database()
    db.save_run({
        "type": "build_algorithm",
        "winners": len(winners),
        "losers": len(losers),
        "factors": len(all_factors),
        "elapsed_sec": round(elapsed),
    })

    return algorithm


# ════════════════════════════════════════════════════════════════════
# TAB 2: PICK STOCKS
# ════════════════════════════════════════════════════════════════════

def pick_stocks(top_n=5, progress_callback=None):
    """
    Apply the two-layer system to find today's best penny stock picks.

    LAYER 1: Quality Gate (quality_gate.py)
      Hard Kills (instant disqualification):
        - Already pumped (>100% in 5 days) -> KILL (most important!)
        - Fraud / SEC investigation in news -> KILL
        - Core product failure in news -> KILL
        - Shell company indicators -> KILL
        - Toxic gross margins (<5%) -> KILL
        - Cash runway exhaustion (<6 months) -> KILL
        - Pre-revenue massive burn -> KILL
        - Negative shareholder equity -> KILL (balance sheet underwater)
        - Sub-dime price (<$0.10) -> KILL (untradeable garbage)
        - Extreme profit margin losses (<-200%) -> KILL (hemorrhaging money)

      Scoring Penalties (reduce score, don't kill):
        - Going concern in SEC filings -> PENALTY (-25pts)
        - Delisting / compliance notice -> PENALTY (-30pts)
        - Extreme price decay (85%+ from 52w high) -> PENALTY (-15 to -30pts scaled)
        - Recent reverse split -> PENALTY (-20 to -35pts scaled for extreme ratios)
        - Excessive float (>100M shares) -> PENALTY (-15pts)
        - Micro-employees (<10 FTEs) -> PENALTY (-20pts)

    LAYER 2: Positive Scoring
      - Setup quality (40%): float, insider ownership, proximity-to-low, P/B
      - Technical (25%): RSI, MACD, StochRSI, volume, OBV, BB, trend
      - Fundamental (25%): revenue growth, short interest, cash position
      - Catalyst (10%): news-based catalyst detection

    Steps:
      1. Load saved algorithm
      2. Get ALL current penny stocks from Finviz
      3. Stage 1: Quick technical score (fast filter)
      4. Stage 2: On top 50:
         a. Run quality gate -- kill broken stocks, penalize sketchy ones
         b. Score survivors -- rank by composite score minus penalties
      5. Return top N picks with full breakdown
    """
    def _log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    algorithm = load_algorithm()
    if not algorithm:
        _log("ERROR: No algorithm found. Run 'Build Algorithm' first (Tab 1).")
        return []

    start = time.time()
    _log("=" * 60)
    _log("PICKING TOP PENNY STOCKS (v3: Kill Filters + Setup Scoring)")
    _log(f"Using algorithm from {algorithm.get('built_date', 'unknown')}")
    _log("=" * 60)

    # ── Get current penny stocks ────────────────────────────────────
    _log("Discovering current penny stocks...")
    stocks = get_penny_stocks()
    if not stocks:
        _log("ERROR: No stocks found.")
        return []

    _log(f"Found {len(stocks)} stocks. Running Stage 1 technical screen...")

    # ── Stage 1: Quick technical score ──────────────────────────────
    tech_factors = [f for f in algorithm["factors"] if f["category"] == "technical"]
    stage1_results = []

    for i, stock in enumerate(stocks):
        ticker = stock["ticker"]
        try:
            hist = get_price_history(ticker, period="3mo")
            if hist is None or hist.empty or len(hist) < 30:
                continue

            features = extract_tech_features(hist)
            if not features:
                continue

            score = _score_features(features, tech_factors)
            stage1_results.append({
                "ticker": ticker,
                "tech_score": score,
                "features": features,
                "price": stock.get("price", 0),
                "volume": stock.get("volume", 0),
                "company": stock.get("company", ""),
                "sector": stock.get("sector", ""),
            })
        except Exception as e:
            logger.debug(f"Stage 1 failed for {ticker}: {e}")

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            _log(f"  Stage 1: {i+1}/{len(stocks)} "
                 f"~{(len(stocks)-i-1)/rate:.0f}s remaining")
        time.sleep(0.1)

    stage1_results.sort(key=lambda x: x["tech_score"], reverse=True)
    _log(f"Stage 1 done: {len(stage1_results)} stocks scored")

    # ── Stage 2: Deep analysis on top 50 ────────────────────────────
    top_candidates = stage1_results[:50]
    sent_factors = [f for f in algorithm["factors"] if f["category"] == "sentiment"]
    fund_factors = [f for f in algorithm["factors"] if f["category"] == "fundamental"]

    # Pre-download Reddit posts ONCE to avoid per-ticker rate limiting
    _log("Pre-downloading Reddit posts (bulk mode)...")
    ensure_bulk_downloaded()

    _log(f"Stage 2: Deep analysis on top {len(top_candidates)} stocks...")
    _log("  Running LAYER 1 kill filters + LAYER 2 scoring...")
    final_results = []
    killed_count = 0

    for j, candidate in enumerate(top_candidates):
        ticker = candidate["ticker"]
        try:
            # ── LAYER 1: Kill Filters + Penalties ──────────────
            info = get_stock_info(ticker)
            gate = run_kill_filters(ticker, info=info)

            if gate["killed"]:
                killed_count += 1
                _log(f"  KILLED {ticker}: {gate['kill_reasons'][0][:80]}...")
                continue  # Skip to next stock -- this one is dead

            # Penalty deduction applied to final score later
            penalty_deduction = gate.get("total_penalty", 0)
            if penalty_deduction > 0:
                _log(f"  PENALTY {ticker}: -{penalty_deduction}pts "
                     f"({len(gate.get('penalties', []))} issue(s))")

            # ── LAYER 2: Positive Scoring ────────────────────────

            # A. Setup quality (40% of total)
            setup_result = score_setup(ticker, info)
            setup_score = setup_result["score"]

            # B. Technical score (25% of total)
            # Use the learned algorithm factors for the technical component
            tech_score = candidate["tech_score"]
            # Also compute the direct technical analysis score and blend
            tech_analysis = analyze_technical(
                get_price_history(ticker, period="3mo")
            )
            if tech_analysis.get("valid"):
                # Blend learned score with direct analysis (60/40)
                tech_score = tech_score * 0.6 + tech_analysis["score"] * 0.4

            # C. Fundamental quality (25% of total)
            fund_result = score_fundamentals(ticker, info)
            fund_score = fund_result["score"]
            # Also blend with learned fundamental factors if available
            if fund_factors:
                fund_feat = extract_fund_features(ticker)
                learned_fund = _score_features(fund_feat, fund_factors)
                fund_score = fund_score * 0.7 + learned_fund * 0.3

            # D. Catalyst score (10% of total)
            cat_result = analyze_catalyst(ticker)
            cat_score = cat_result.get("score", 50)

            # E. Sentiment (informational, not in primary weights,
            #    but used as a small adjustment)
            sent = analyze_sentiment(ticker)
            has_sentiment_data = sent.get("has_data", False)
            if has_sentiment_data and sent_factors:
                sent_features = {
                    "reddit_mentions": sent.get("reddit", {}).get("mentions", 0),
                    "reddit_sentiment": sent.get("reddit", {}).get("avg_sentiment", 0),
                    "stocktwits_bullish_pct": (
                        sent.get("stocktwits", {}).get("bullish", 0) /
                        max(1, sent.get("stocktwits", {}).get("total", 1))
                    ),
                    "combined_sentiment": sent.get("combined_sentiment", 0),
                    "buzz_score": sent.get("buzz_score", 0),
                }
                sent_score = _score_features(sent_features, sent_factors)
            else:
                sent_score = 50.0

            # Market context (small adjustment only)
            mkt_result = analyze_market(candidate.get("sector", ""))
            mkt_score = mkt_result.get("score", 50)

            # ── Composite Score ──────────────────────────────────
            # Primary: setup(40%) + technical(25%) + fundamental(25%) + catalyst(10%)
            base_score = (
                setup_score * WEIGHTS["setup"] +
                tech_score * WEIGHTS["technical"] +
                fund_score * WEIGHTS["fundamental"] +
                cat_score * WEIGHTS["catalyst"]
            )

            # Secondary adjustments: sentiment + market context
            # These are small nudges, not primary drivers.
            # ORKT had zero social buzz but still ran +300% -- sentiment
            # should NOT be a gate.
            sent_adj = (sent_score - 50) * 0.03  # +/- 1.5 points max
            mkt_adj = (mkt_score - 50) * 0.02    # +/- 1.0 points max

            # Apply quality gate penalty deductions (going concern,
            # delisting notices, price decay, reverse splits, excessive float).
            # These are "normal penny stock shadiness" -- bad signs that
            # reduce the score but don't kill outright.
            penalty_adj = -penalty_deduction

            final_score = max(0, min(100, base_score + sent_adj + mkt_adj + penalty_adj))

            final_results.append({
                "ticker": ticker,
                "final_score": round(final_score, 1),
                "company": candidate.get("company", ""),
                "price": candidate.get("price", 0),
                "volume": candidate.get("volume", 0),
                "sector": candidate.get("sector", ""),
                "quality_gate": gate,
                "penalty_deduction": penalty_deduction,
                "sub_scores": {
                    "setup": round(setup_score, 1),
                    "technical": round(tech_score, 1),
                    "fundamental": round(fund_score, 1),
                    "catalyst": round(cat_score, 1),
                    "sentiment": round(sent_score, 1),
                    "market": round(mkt_score, 1),
                },
                "setup_detail": setup_result,
                "fundamental_detail": fund_result,
                "key_indicators": {
                    "float_shares": info.get("float_shares", 0),
                    "insider_pct": info.get("insider_percent_held", 0),
                    "position_52w": setup_result["proximity_to_low"].get("position"),
                    "price_to_book": setup_result["price_to_book"].get("price_to_book"),
                    "rsi": candidate["features"].get("rsi"),
                    "stochrsi": candidate["features"].get("stochrsi"),
                    "volume_spike": candidate["features"].get("volume_spike"),
                    "revenue_growth": info.get("revenue_growth", 0),
                    "short_pct_float": info.get("short_percent_of_float", 0),
                    "sentiment": round(sent.get("combined_sentiment", 0), 3),
                    # New indicators
                    "adx": candidate["features"].get("adx"),
                    "mfi": candidate["features"].get("mfi"),
                    "bb_squeeze": tech_analysis.get("bb_squeeze", {}).get("is_squeeze", False) if tech_analysis.get("valid") else False,
                    "consolidating": tech_analysis.get("consolidation", {}).get("is_consolidating", False) if tech_analysis.get("valid") else False,
                    "multiday_unusual_vol": tech_analysis.get("multiday_unusual_volume", {}).get("unusual_days", 0) if tech_analysis.get("valid") else 0,
                    "squeeze_setup": fund_result.get("squeeze_composite", {}).get("is_squeeze_setup", False),
                    "dilution_filings": fund_result.get("dilution_risk", {}).get("dilution_filings_6m", 0),
                    "atr": tech_analysis.get("atr", 0) if tech_analysis.get("valid") else 0,
                },
                "risk_management": _compute_risk_management(
                    candidate.get("price", 0),
                    tech_analysis.get("atr", 0) if tech_analysis.get("valid") else 0,
                    tech_analysis.get("support_resistance", {}).get("support") if tech_analysis.get("valid") else None,
                ),
                "sentiment_detail": sent,
                "catalyst_detail": cat_result,
            })

        except Exception as e:
            logger.debug(f"Stage 2 failed for {ticker}: {e}")

        if (j + 1) % 5 == 0:
            _log(f"  Stage 2: {j+1}/{len(top_candidates)} "
                 f"({killed_count} killed, {len(final_results)} scored)")

    final_results.sort(key=lambda x: x["final_score"], reverse=True)
    picks = final_results[:top_n]

    elapsed = time.time() - start
    _log("=" * 60)
    _log(f"RESULTS: {killed_count} stocks KILLED by quality filters, "
         f"{len(final_results)} survived")
    _log(f"TOP {len(picks)} PICKS (found in {elapsed:.0f}s)")
    _log("=" * 60)
    for k, pick in enumerate(picks, 1):
        ss = pick["sub_scores"]
        ki = pick["key_indicators"]
        _log(f"  #{k}. {pick['ticker']} - ${pick['price']:.2f} "
             f"(Score: {pick['final_score']:.1f})")
        _log(f"      {pick['company']}")
        _log(f"      Setup:{ss['setup']:.0f} Tech:{ss['technical']:.0f} "
             f"Fund:{ss['fundamental']:.0f} Cat:{ss['catalyst']:.0f}")
        _log(f"      Float: {ki.get('float_shares', 'N/A'):,.0f} | "
             f"Insider: {(ki.get('insider_pct') or 0)*100:.0f}% | "
             f"52w Pos: {ki.get('position_52w', 'N/A')} | "
             f"P/B: {ki.get('price_to_book', 'N/A')} | "
             f"RSI: {ki.get('rsi', 'N/A')} | "
             f"StochRSI: {ki.get('stochrsi', 'N/A')}")
        # New indicators summary
        signals = []
        if ki.get("bb_squeeze"):
            signals.append("BB-SQUEEZE")
        if ki.get("consolidating"):
            signals.append("CONSOLIDATING")
        if ki.get("squeeze_setup"):
            signals.append("SQUEEZE-SETUP")
        if ki.get("multiday_unusual_vol", 0) >= 3:
            signals.append(f"UNUSUAL-VOL({ki['multiday_unusual_vol']}d)")
        if ki.get("dilution_filings", 0) > 0:
            signals.append(f"DILUTION-RISK({ki['dilution_filings']})")
        if signals:
            _log(f"      Signals: {' | '.join(signals)}")
        # ADX/MFI
        _log(f"      ADX: {ki.get('adx', 'N/A')} | "
             f"MFI: {ki.get('mfi', 'N/A')}")
        # Risk management
        rm = pick.get("risk_management", {})
        if rm.get("stop_loss"):
            _log(f"      Stop: ${rm['stop_loss']:.4f} "
                 f"({rm['risk_pct']:.1f}% risk)")
        if pick.get("penalty_deduction", 0) > 0:
            _log(f"      Penalties: -{pick['penalty_deduction']}pts "
                 f"({len(pick['quality_gate'].get('penalties', []))} issue(s))")
            for pen in pick["quality_gate"].get("penalties", []):
                _log(f"        {pen[:100]}")
    _log("=" * 60)

    # Save picks to database
    db = Database()
    for pick in picks:
        db.save_pick(pick)
    db.save_run({
        "type": "pick_stocks",
        "total_screened": len(stocks),
        "stage1_passed": len(stage1_results),
        "killed_by_filters": killed_count,
        "final_scored": len(final_results),
        "final_picks": len(picks),
        "elapsed_sec": round(elapsed),
    })

    return picks


# ════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════

def _compare_groups(winner_features: list, loser_features: list) -> list:
    """
    Compare feature distributions between winners and losers.
    Returns list of factors with separation scores (Cohen's d).
    """
    if not winner_features or not loser_features:
        return []

    w_df = pd.DataFrame(winner_features)
    l_df = pd.DataFrame(loser_features)

    factors = []
    numeric_cols = w_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col == "ticker":
            continue

        w_vals = w_df[col].dropna()
        l_vals = l_df[col].dropna()

        if len(w_vals) < 2 or len(l_vals) < 2:
            continue

        w_mean = float(w_vals.mean())
        l_mean = float(l_vals.mean())
        w_std = float(w_vals.std())
        l_std = float(l_vals.std())

        pooled_std = math.sqrt((w_std ** 2 + l_std ** 2) / 2) if (w_std + l_std) > 0 else 1
        separation = abs(w_mean - l_mean) / pooled_std if pooled_std > 0 else 0

        factors.append({
            "feature": col,
            "winner_mean": round(w_mean, 4),
            "loser_mean": round(l_mean, 4),
            "winner_std": round(w_std, 4),
            "loser_std": round(l_std, 4),
            "separation": round(separation, 4),
            "direction": "higher" if w_mean > l_mean else "lower",
        })

    factors.sort(key=lambda f: f["separation"], reverse=True)
    return factors


def _score_features(stock_features: dict, algorithm_factors: list) -> float:
    """
    Score a stock's features against the learned algorithm factors.

    For each factor: how much does this stock look like a winner?
    Returns score 0-100.
    """
    if not algorithm_factors:
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
            continue  # Not discriminative enough

        # How much does this value look like a winner vs loser?
        range_ = abs(w_mean - l_mean)
        if range_ < 0.0001:
            continue

        if factor["direction"] == "higher":
            # Higher value = more like winner
            score = (value - l_mean) / range_
        else:
            # Lower value = more like winner
            score = (l_mean - value) / range_

        score = max(0, min(1, score))  # Clamp to [0, 1]
        weight = separation  # More discriminative factors matter more

        total_score += score * weight
        total_weight += weight

    if total_weight > 0:
        return (total_score / total_weight) * 100
    return 50.0


def _compute_risk_management(price: float, atr: float, support: float = None) -> dict:
    """
    Compute risk management levels for a pick.
    Uses ATR-based stop-loss (2x ATR below entry) and support levels.
    """
    if not price or price <= 0:
        return {"stop_loss": None, "risk_pct": None, "position_note": ""}

    # ATR-based stop (2x ATR below entry)
    atr_stop = price - (2 * atr) if atr > 0 else None

    # Support-based stop (just below nearest support)
    support_stop = support * 0.97 if support and support > 0 else None

    # Use the tighter (higher) stop-loss
    if atr_stop and support_stop:
        stop = max(atr_stop, support_stop)
    elif atr_stop:
        stop = atr_stop
    elif support_stop:
        stop = support_stop
    else:
        stop = price * 0.85  # Default 15% stop

    stop = max(0.01, stop)  # Never below $0.01
    risk_pct = ((price - stop) / price) * 100

    return {
        "stop_loss": round(stop, 4),
        "risk_pct": round(risk_pct, 1),
        "atr_stop": round(atr_stop, 4) if atr_stop else None,
        "support_stop": round(support_stop, 4) if support_stop else None,
        "position_note": f"Stop ${stop:.4f} ({risk_pct:.1f}% risk)" if stop else "",
    }


def _save_algorithm(algorithm: dict):
    """Save algorithm to permanent JSON file."""
    with open(ALGORITHM_FILE, "w") as f:
        json.dump(algorithm, f, indent=2, default=str)
    logger.info(f"Algorithm saved to {ALGORITHM_FILE}")


def load_algorithm() -> dict:
    """Load the saved algorithm."""
    if not os.path.exists(ALGORITHM_FILE):
        return {}
    try:
        with open(ALGORITHM_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load algorithm: {e}")
        return {}
