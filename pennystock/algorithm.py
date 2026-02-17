"""
The ONE algorithm for penny stock prediction.

Two functions:
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

from pennystock.data.finviz_client import get_penny_stocks, get_high_gainers
from pennystock.data.yahoo_client import get_price_history, get_stock_info
from pennystock.analysis.technical import extract_features as extract_tech_features
from pennystock.analysis.technical import analyze as analyze_technical
from pennystock.analysis.sentiment import analyze as analyze_sentiment
from pennystock.analysis.fundamental import extract_features as extract_fund_features
from pennystock.analysis.fundamental import analyze as analyze_fundamental
from pennystock.analysis.catalyst import analyze as analyze_catalyst
from pennystock.analysis.market_context import analyze as analyze_market
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
    loser_sample = losers[:min(len(losers), len(winners) * 2)]  # 2:1 ratio

    sentiment_features = {"winners": [], "losers": []}
    fund_features = {"winners": [], "losers": []}

    analyze_tickers = [(t, "winners") for t in winners] + [(t, "losers") for t in loser_sample]

    for j, (ticker, group) in enumerate(analyze_tickers):
        try:
            # Sentiment
            sent = analyze_sentiment(ticker)
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

    # Assign category weights based on how discriminative each category is
    cat_separations = {}
    for f in all_factors:
        cat = f["category"]
        cat_separations.setdefault(cat, []).append(f["separation"])

    total_sep = sum(np.mean(v) for v in cat_separations.values()) or 1
    category_weights = {
        cat: round(np.mean(seps) / total_sep, 3)
        for cat, seps in cat_separations.items()
    }

    # Ensure minimum weights
    for cat in ["technical", "sentiment", "fundamental"]:
        if cat not in category_weights:
            category_weights[cat] = 0.1
    # Re-normalize
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
    Apply the learned algorithm to find today's best penny stock picks.

    Steps:
      1. Load saved algorithm
      2. Get ALL current penny stocks from Finviz
      3. Stage 1: Score each stock on technical features (fast)
      4. Stage 2: Deep analysis (sentiment, fundamentals) on top 50
      5. Return top N picks with full scoring breakdown
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
    _log("PICKING TOP PENNY STOCKS")
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
    cat_weights = algorithm.get("category_weights", {})

    _log(f"Stage 2: Deep analysis on top {len(top_candidates)} stocks...")
    final_results = []

    for j, candidate in enumerate(top_candidates):
        ticker = candidate["ticker"]
        try:
            # Sentiment
            sent = analyze_sentiment(ticker)
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

            # Fundamentals
            fund_feat = extract_fund_features(ticker)

            # Catalyst
            cat_result = analyze_catalyst(ticker)

            # Market context
            mkt_result = analyze_market(candidate.get("sector", ""))

            # Score each category
            tech_score = candidate["tech_score"]
            sent_score = _score_features(sent_features, sent_factors) if sent_factors else 50
            fund_score = _score_features(fund_feat, fund_factors) if fund_factors else 50
            cat_score = cat_result.get("score", 50)
            mkt_score = mkt_result.get("score", 50)

            # Weighted composite
            tw = cat_weights.get("technical", 0.4)
            sw = cat_weights.get("sentiment", 0.3)
            fw = cat_weights.get("fundamental", 0.3)

            # Add catalyst and market as bonus adjustments (not learned, fixed)
            base_score = tech_score * tw + sent_score * sw + fund_score * fw
            adjustment = (cat_score - 50) * 0.1 + (mkt_score - 50) * 0.05
            final_score = max(0, min(100, base_score + adjustment))

            final_results.append({
                "ticker": ticker,
                "final_score": round(final_score, 1),
                "company": candidate.get("company", ""),
                "price": candidate.get("price", 0),
                "volume": candidate.get("volume", 0),
                "sector": candidate.get("sector", ""),
                "sub_scores": {
                    "technical": round(tech_score, 1),
                    "sentiment": round(sent_score, 1),
                    "fundamental": round(fund_score, 1),
                    "catalyst": round(cat_score, 1),
                    "market": round(mkt_score, 1),
                },
                "key_indicators": {
                    "rsi": candidate["features"].get("rsi"),
                    "volume_spike": candidate["features"].get("volume_spike"),
                    "price_trend_20d": candidate["features"].get("price_trend_20d"),
                    "reddit_mentions": sent_features.get("reddit_mentions", 0),
                    "sentiment": round(sent_features.get("combined_sentiment", 0), 3),
                    "insider_direction": "N/A",
                },
                "sentiment_detail": sent,
                "catalyst_detail": cat_result,
            })

        except Exception as e:
            logger.debug(f"Stage 2 failed for {ticker}: {e}")

        if (j + 1) % 5 == 0:
            _log(f"  Stage 2: {j+1}/{len(top_candidates)}")

    final_results.sort(key=lambda x: x["final_score"], reverse=True)
    picks = final_results[:top_n]

    elapsed = time.time() - start
    _log("=" * 60)
    _log(f"TOP {len(picks)} PICKS (found in {elapsed:.0f}s)")
    _log("=" * 60)
    for k, pick in enumerate(picks, 1):
        ss = pick["sub_scores"]
        _log(f"  #{k}. {pick['ticker']} - ${pick['price']:.2f} "
             f"(Score: {pick['final_score']:.1f})")
        _log(f"      {pick['company']}")
        _log(f"      Tech:{ss['technical']:.0f} Sent:{ss['sentiment']:.0f} "
             f"Fund:{ss['fundamental']:.0f} Cat:{ss['catalyst']:.0f} "
             f"Mkt:{ss['market']:.0f}")
    _log("=" * 60)

    # Save picks to database
    db = Database()
    for pick in picks:
        db.save_pick(pick)
    db.save_run({
        "type": "pick_stocks",
        "total_screened": len(stocks),
        "stage1_passed": len(stage1_results),
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
