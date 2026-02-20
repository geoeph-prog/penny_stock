"""
Algorithm Optimizer — runs backtests on 1st & 15th of every month over ~3 years,
aggregates all results, and finds optimal sell strategy + scoring weights.

This is the "Backtest Algorithm" engine. It:
1. Downloads universe + 3-year price history ONCE
2. Pre-caches stock info for all tickers
3. Runs lightweight backtests on ~70 target dates
4. Collects all scored picks with full daily price paths
5. Grid-searches sell parameters (hold period, stop-loss, take-profit, trailing stop)
6. Tests weight configurations and correlates sub-scores with returns
7. Generates recommended config changes
8. Optionally applies them to optimized_config.json

Usage:
    from pennystock.backtest.optimizer import run_algorithm_optimization
    result = run_algorithm_optimization()
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from pennystock.config import (
    ALGORITHM_VERSION,
    WEIGHTS,
    MIN_PRICE,
    MAX_PRICE,
    MIN_VOLUME,
    MIN_RECOMMENDATION_SCORE,
    PRE_PUMP_HIGH_CONVICTION_BONUS,
    PRE_PUMP_MEDIUM_CONVICTION_BONUS,
)
from pennystock.data.finviz_client import get_penny_stocks
from pennystock.data.yahoo_client import get_stock_info, get_batch_history
from pennystock.analysis.technical import (
    extract_features as extract_tech_features,
    analyze as analyze_technical,
)
from pennystock.analysis.fundamental import score_setup, score_fundamentals
from pennystock.analysis.pre_pump import score_pre_pump
from pennystock.algorithm import load_algorithm
from pennystock.backtest.historical import (
    _filter_historical_candidates,
    _historical_kill_check,
    _score_features,
)


# ─── Grid search parameter space ─────────────────────────────────
HOLD_PERIODS = [3, 5, 7, 10, 14, 20]

STOP_LOSS_OPTIONS = [0, -8, -10, -12, -15, -20, -25]  # 0 = disabled

TAKE_PROFIT_OPTIONS = [0, 10, 15, 20, 25, 30, 50]  # 0 = disabled

TRAILING_STOP_OPTIONS = [
    (0, 0),       # disabled
    (10, 5),      # activate at +10%, trail 5% from peak
    (10, 8),
    (15, 8),
    (15, 10),
    (20, 10),
]

TOP_N_CANDIDATES = [3, 5, 7, 10, 15]

WEIGHT_CANDIDATES = [
    {"setup": 0.25, "technical": 0.20, "pre_pump": 0.35, "fundamental": 0.10, "catalyst": 0.10},
    {"setup": 0.20, "technical": 0.20, "pre_pump": 0.40, "fundamental": 0.10, "catalyst": 0.10},
    {"setup": 0.30, "technical": 0.20, "pre_pump": 0.30, "fundamental": 0.10, "catalyst": 0.10},
    {"setup": 0.25, "technical": 0.25, "pre_pump": 0.30, "fundamental": 0.10, "catalyst": 0.10},
    {"setup": 0.20, "technical": 0.15, "pre_pump": 0.45, "fundamental": 0.10, "catalyst": 0.10},
    {"setup": 0.25, "technical": 0.20, "pre_pump": 0.35, "fundamental": 0.15, "catalyst": 0.05},
    {"setup": 0.20, "technical": 0.25, "pre_pump": 0.35, "fundamental": 0.10, "catalyst": 0.10},
    {"setup": 0.15, "technical": 0.15, "pre_pump": 0.50, "fundamental": 0.10, "catalyst": 0.10},
]

MIN_SCORE_CANDIDATES = [35, 40, 45, 50, 55, 60]

MAX_DAILY_PATH = 20  # Store up to 20 days of forward OHLC per pick


# ─── Main entry point ────────────────────────────────────────────

def run_algorithm_optimization(progress_callback=None) -> dict:
    """Top-level entry point for the full optimization."""
    optimizer = AlgorithmOptimizer(progress_callback)
    return optimizer.run()


# ─── Core optimizer class ────────────────────────────────────────

class AlgorithmOptimizer:

    def __init__(self, progress_callback=None):
        self._log_lines = []
        self._progress = progress_callback
        self.stocks = []
        self.all_hist = {}
        self.stock_info_cache = {}
        self.algorithm = None

    def _log(self, msg=""):
        logger.info(msg)
        self._log_lines.append(msg)
        if self._progress:
            self._progress(msg)

    # ─────────────────────────────────────────────────────────
    # MAIN RUN
    # ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        start_time = time.time()
        self._log(f"{'=' * 70}")
        self._log(f"  ALGORITHM OPTIMIZER  v{ALGORITHM_VERSION}")
        self._log(f"  Full Backtest + Parameter Optimization")
        self._log(f"{'=' * 70}")

        # Phase 1: Load learned algorithm
        self._log("\n  Phase 1: Loading algorithm...")
        self.algorithm = load_algorithm()
        if not self.algorithm:
            self._log("  ERROR: No algorithm.json found. Run 'Build Algorithm' first.")
            return None
        self._log(f"  Loaded algorithm ({len(self.algorithm.get('factors', []))} factors)")

        # Phase 2: Fetch data (one-time, slow)
        self._log("\n  Phase 2: Fetching market data (one-time download)...")
        self._fetch_data()

        # Phase 3: Generate target dates
        target_dates = self._generate_target_dates()
        self._log(f"\n  Phase 3: Generated {len(target_dates)} backtest dates")
        self._log(f"  Range: {target_dates[0]} to {target_dates[-1]}")

        # Phase 4: Run all backtests
        self._log(f"\n  Phase 4: Running {len(target_dates)} backtests...")
        all_picks = self._run_all_backtests(target_dates)
        self._log(f"  Collected {len(all_picks)} total scored picks")

        if len(all_picks) < 30:
            self._log("  ERROR: Not enough picks to optimize. Need at least 30.")
            return {"error": "Insufficient data", "total_picks": len(all_picks),
                    "report_lines": self._log_lines}

        # Phase 5: Optimize sell parameters
        self._log(f"\n  Phase 5: Optimizing sell parameters (grid search)...")
        sell_results = self._optimize_sell_params(all_picks)

        # Phase 6: Optimize scoring weights
        self._log(f"\n  Phase 6: Optimizing scoring weights...")
        weight_results = self._optimize_weights(all_picks)

        # Phase 7: Generate recommendations
        self._log(f"\n  Phase 7: Generating final recommendations...")
        recommendations = self._generate_recommendations(sell_results, weight_results)

        elapsed = time.time() - start_time
        self._log(f"\n{'=' * 70}")
        self._log(f"  Optimization completed in {elapsed / 60:.1f} minutes")
        self._log(f"{'=' * 70}")

        result = {
            "sell_optimization": sell_results,
            "weight_optimization": weight_results,
            "recommendations": recommendations,
            "total_picks": len(all_picks),
            "dates_tested": len(target_dates),
            "report_lines": self._log_lines,
        }

        self._save_results(result)
        return result

    # ─────────────────────────────────────────────────────────
    # PHASE 2: DATA FETCHING
    # ─────────────────────────────────────────────────────────

    def _fetch_data(self):
        self._log("  Fetching penny stock universe from Finviz...")
        self.stocks = get_penny_stocks()
        self._log(f"  Found {len(self.stocks)} stocks in universe")

        self._log("  Downloading 3-year price history (this takes several minutes)...")
        tickers = [s["ticker"] for s in self.stocks]
        self.all_hist = get_batch_history(tickers, period="3y")
        self._log(f"  Got history for {len(self.all_hist)}/{len(self.stocks)} stocks")

        self._log("  Pre-fetching stock info for all tickers...")
        fetched = 0
        for i, ticker in enumerate(tickers):
            if i % 100 == 0 and i > 0:
                self._log(f"    Stock info: {i}/{len(tickers)} fetched...")
            try:
                info = get_stock_info(ticker)
                if info:
                    self.stock_info_cache[ticker] = info
                    fetched += 1
            except Exception:
                pass
        self._log(f"  Cached info for {fetched} stocks")

    # ─────────────────────────────────────────────────────────
    # PHASE 3: TARGET DATE GENERATION
    # ─────────────────────────────────────────────────────────

    def _generate_target_dates(self) -> list:
        """1st and 15th of each month for ~3 years, weekday-adjusted."""
        dates = []
        today = datetime.now()
        latest = today - timedelta(days=30)  # need 20+ trading days forward
        start = today - timedelta(days=365 * 3)

        current = datetime(start.year, start.month, 1)
        while current <= latest:
            for day in [1, 15]:
                try:
                    dt = datetime(current.year, current.month, day)
                except ValueError:
                    continue
                if dt > latest or dt < start:
                    continue
                # Adjust weekends to Monday
                while dt.weekday() >= 5:
                    dt += timedelta(days=1)
                dates.append(dt.strftime("%Y-%m-%d"))
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        return dates

    # ─────────────────────────────────────────────────────────
    # PHASE 4: RUN ALL BACKTESTS
    # ─────────────────────────────────────────────────────────

    def _run_all_backtests(self, target_dates) -> list:
        tech_factors = [f for f in self.algorithm["factors"]
                        if f["category"] == "technical"]
        all_picks = []
        for idx, date_str in enumerate(target_dates):
            if idx % 5 == 0:
                self._log(f"    [{idx + 1}/{len(target_dates)}] {date_str}...")
            picks = self._run_single_backtest(date_str, tech_factors)
            all_picks.extend(picks)
        return all_picks

    def _run_single_backtest(self, target_date_str, tech_factors) -> list:
        """Lightweight backtest for one date using cached data."""
        target = pd.Timestamp(target_date_str)

        candidates = _filter_historical_candidates(
            self.stocks, self.all_hist, target
        )
        if not candidates:
            return []

        # Stage 1: Tech screen
        for c in candidates:
            features = extract_tech_features(c["hist_before"])
            c["tech_score"] = _score_features(features, tech_factors) if features else 0
        candidates.sort(key=lambda x: x["tech_score"], reverse=True)
        top_50 = candidates[:50]

        # Stage 2: Full scoring
        scored = []
        for c in top_50:
            ticker = c["ticker"]
            try:
                killed, _ = _historical_kill_check(c["hist_before"])
                if killed:
                    continue

                tech = analyze_technical(c["hist_before"])
                tech_score = tech.get("score", 50) if tech.get("valid") else 50

                info = self.stock_info_cache.get(ticker, {})

                setup_result = score_setup(ticker, info=info)
                setup_score = setup_result.get("score", 50)

                # Historical 52w proximity override
                hist_52w = (c["hist_before"].iloc[-252:]
                            if len(c["hist_before"]) >= 252
                            else c["hist_before"])
                h52 = float(hist_52w["High"].max()
                            if "High" in hist_52w.columns
                            else hist_52w["Close"].max())
                l52 = float(hist_52w["Low"].min()
                            if "Low" in hist_52w.columns
                            else hist_52w["Close"].min())
                if h52 > l52 > 0:
                    hist_pos = (c["entry_price"] - l52) / (h52 - l52)
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
                    orig_prox = setup_result.get(
                        "proximity_to_low", {}
                    ).get("score", 50)
                    setup_score += (prox_score - orig_prox) * 0.25

                pp_features = {
                    "multiday_unusual_vol_days": (
                        tech.get("multiday_unusual_volume", {}).get("unusual_days", 0)
                        if tech.get("valid") else 0
                    ),
                }
                pp = score_pre_pump(ticker, info=info, tech_features=pp_features)
                pre_pump_score = pp["score"]

                fund = score_fundamentals(ticker, info=info)
                fund_score = fund.get("score", 50)

                cat_score = 50  # neutral

                sub_scores = {
                    "setup": setup_score,
                    "technical": tech_score,
                    "pre_pump": pre_pump_score,
                    "fundamental": fund_score,
                    "catalyst": cat_score,
                }

                pp_conf = pp.get("confidence", "LOW")
                pp_count = pp.get("confluence_count", 0)

                if pp_conf == "HIGH":
                    bonus = PRE_PUMP_HIGH_CONVICTION_BONUS
                elif pp_conf == "MEDIUM":
                    bonus = PRE_PUMP_MEDIUM_CONVICTION_BONUS
                else:
                    bonus = 0

                base_score = (
                    setup_score * WEIGHTS["setup"]
                    + tech_score * WEIGHTS["technical"]
                    + pre_pump_score * WEIGHTS["pre_pump"]
                    + fund_score * WEIGHTS["fundamental"]
                    + cat_score * WEIGHTS["catalyst"]
                )
                final_score = max(0, min(100, base_score + bonus))

                # Keep ALL scored picks (even below threshold) so we can
                # test different MIN_RECOMMENDATION_SCORE values later.
                # Filter at analysis time, not collection time.
                if final_score < 25:  # hard floor to avoid junk
                    continue

                daily_path = self._build_daily_path(
                    c["hist_full"], c["entry_date"], c["entry_price"]
                )
                if not daily_path:
                    continue

                scored.append({
                    "ticker": ticker,
                    "entry_date": str(c["entry_date"].date()),
                    "entry_price": c["entry_price"],
                    "score": final_score,
                    "sub_scores": sub_scores,
                    "pre_pump_confidence": pp_conf,
                    "pre_pump_confluence": pp_count,
                    "bonus": bonus,
                    "daily_path": daily_path,
                })
            except Exception as e:
                logger.debug(f"Optimizer error for {ticker}: {e}")
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def _build_daily_path(self, hist_full, entry_date, entry_price):
        """Build daily OHLC for up to MAX_DAILY_PATH days after entry."""
        hist_after = hist_full[hist_full.index > entry_date]
        path = []
        for i in range(min(MAX_DAILY_PATH, len(hist_after))):
            row = hist_after.iloc[i]
            path.append({
                "day": i + 1,
                "open": float(row.get("Open", row["Close"])),
                "high": float(row.get("High", row["Close"])),
                "low": float(row.get("Low", row["Close"])),
                "close": float(row["Close"]),
            })
        return path

    # ─────────────────────────────────────────────────────────
    # PHASE 5: SELL PARAMETER OPTIMIZATION
    # ─────────────────────────────────────────────────────────

    def _optimize_sell_params(self, all_picks) -> dict:
        """Grid search over sell parameters to maximize expected return."""

        # Group picks by entry_date, sorted by score within each date
        picks_by_date = defaultdict(list)
        for p in all_picks:
            picks_by_date[p["entry_date"]].append(p)
        for dk in picks_by_date:
            picks_by_date[dk].sort(key=lambda x: x["score"], reverse=True)

        total_combos = (len(TOP_N_CANDIDATES) * len(HOLD_PERIODS)
                        * len(STOP_LOSS_OPTIONS) * len(TAKE_PROFIT_OPTIONS)
                        * len(TRAILING_STOP_OPTIONS))
        self._log(f"  Testing {total_combos:,} parameter combinations...")

        results_grid = []
        combo_count = 0

        for top_n in TOP_N_CANDIDATES:
            for hold in HOLD_PERIODS:
                for sl in STOP_LOSS_OPTIONS:
                    for tp in TAKE_PROFIT_OPTIONS:
                        for trail_act, trail_dist in TRAILING_STOP_OPTIONS:
                            combo_count += 1
                            if combo_count % 2000 == 0:
                                self._log(
                                    f"    {combo_count:,}/{total_combos:,} "
                                    f"({combo_count / total_combos * 100:.0f}%)"
                                )

                            returns = []
                            for date_picks in picks_by_date.values():
                                for p in date_picks[:top_n]:
                                    if len(p["daily_path"]) < min(3, hold):
                                        continue
                                    ret, _ = _simulate_strategy(
                                        p, hold, sl, tp,
                                        trail_act, trail_dist,
                                    )
                                    returns.append(ret)

                            if len(returns) < 30:
                                continue

                            wins = sum(1 for r in returns if r > 0)
                            win_rate = wins / len(returns) * 100
                            avg_ret = sum(returns) / len(returns)
                            sorted_ret = sorted(returns)
                            median_ret = sorted_ret[len(sorted_ret) // 2]

                            results_grid.append({
                                "top_n": top_n,
                                "hold_days": hold,
                                "stop_loss": sl,
                                "take_profit": tp,
                                "trail_activate": trail_act,
                                "trail_distance": trail_dist,
                                "win_rate": round(win_rate, 1),
                                "avg_return": round(avg_ret, 2),
                                "median_return": round(median_ret, 2),
                                "total_trades": len(returns),
                                "worst": round(min(returns), 2),
                                "best": round(max(returns), 2),
                            })

        results_grid.sort(key=lambda x: x["avg_return"], reverse=True)
        top_20 = results_grid[:20]
        best = top_20[0] if top_20 else None

        self._log(f"\n  Tested {len(results_grid):,} valid combinations")
        self._log(f"\n  {'=' * 70}")
        self._log(f"  TOP 20 SELL STRATEGIES BY EXPECTED RETURN")
        self._log(f"  {'=' * 70}")
        self._log(
            f"  {'#':>3} {'TopN':>4} {'Hold':>4} {'SL':>5} {'TP':>5} "
            f"{'Trail':>10} {'WinR':>5} {'AvgR':>6} {'MedR':>6} {'n':>5}"
        )
        self._log(f"  {'---':>3} {'----':>4} {'----':>4} {'-----':>5} {'-----':>5} "
                   f"{'----------':>10} {'-----':>5} {'------':>6} {'------':>6} {'-----':>5}")

        for i, r in enumerate(top_20):
            trail_str = (f"{r['trail_activate']}/{r['trail_distance']}"
                         if r['trail_activate'] > 0 else "off")
            sl_str = f"{r['stop_loss']}%" if r['stop_loss'] != 0 else "off"
            tp_str = f"+{r['take_profit']}%" if r['take_profit'] != 0 else "off"
            self._log(
                f"  {i + 1:>3} {r['top_n']:>4} {r['hold_days']:>3}d "
                f"{sl_str:>5} {tp_str:>5} {trail_str:>10} "
                f"{r['win_rate']:>4.0f}% {r['avg_return']:>+5.1f}% "
                f"{r['median_return']:>+5.1f}% {r['total_trades']:>5}"
            )

        return {"best": best, "top_20": top_20, "total_tested": len(results_grid)}

    # ─────────────────────────────────────────────────────────
    # PHASE 6: WEIGHT OPTIMIZATION
    # ─────────────────────────────────────────────────────────

    def _optimize_weights(self, all_picks) -> dict:
        """Test weight configs and compute sub-score correlations."""

        picks_by_date = defaultdict(list)
        for p in all_picks:
            picks_by_date[p["entry_date"]].append(p)

        # First compute correlations to generate a data-driven weight config
        correlations = self._compute_score_correlations(all_picks, eval_hold=5)

        # Add correlation-based weight config
        weight_candidates = list(WEIGHT_CANDIDATES)
        if correlations:
            pos_corrs = {k: max(0.01, v) for k, v in correlations.items()}
            total = sum(pos_corrs.values())
            if total > 0:
                corr_weights = {k: round(v / total, 2) for k, v in pos_corrs.items()}
                # Fix rounding to sum to 1.0
                diff = 1.0 - sum(corr_weights.values())
                max_key = max(corr_weights, key=corr_weights.get)
                corr_weights[max_key] = round(corr_weights[max_key] + diff, 2)
                weight_candidates.append(corr_weights)
                self._log(f"  Added correlation-based weight config: {corr_weights}")

        # Test all weight + min_score combinations using 5d hold
        eval_hold = 5
        weight_results = []

        for min_score in MIN_SCORE_CANDIDATES:
            for weights in weight_candidates:
                returns = []
                for date_picks in picks_by_date.values():
                    rescored = []
                    for p in date_picks:
                        ss = p["sub_scores"]
                        new_base = (
                            ss["setup"] * weights["setup"]
                            + ss["technical"] * weights["technical"]
                            + ss["pre_pump"] * weights["pre_pump"]
                            + ss["fundamental"] * weights["fundamental"]
                            + ss["catalyst"] * weights["catalyst"]
                        )
                        new_score = max(0, min(100, new_base + p["bonus"]))
                        if new_score >= min_score:
                            rescored.append((new_score, p))

                    rescored.sort(key=lambda x: x[0], reverse=True)
                    for _, p in rescored[:5]:
                        if len(p["daily_path"]) >= eval_hold:
                            exit_price = p["daily_path"][eval_hold - 1]["close"]
                            ret = ((exit_price - p["entry_price"])
                                   / p["entry_price"]) * 100
                            returns.append(ret)

                if len(returns) < 20:
                    continue

                avg_ret = sum(returns) / len(returns)
                win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100

                weight_results.append({
                    "weights": weights,
                    "min_score": min_score,
                    "win_rate": round(win_rate, 1),
                    "avg_return": round(avg_ret, 2),
                    "total_picks": len(returns),
                })

        weight_results.sort(key=lambda x: x["avg_return"], reverse=True)
        best = weight_results[0] if weight_results else None

        self._log(f"\n  {'=' * 70}")
        self._log(f"  TOP 10 WEIGHT CONFIGURATIONS (5d hold, top 5)")
        self._log(f"  {'=' * 70}")

        for i, r in enumerate(weight_results[:10]):
            w = r["weights"]
            self._log(
                f"  #{i + 1}: MinScore={r['min_score']} "
                f"S:{w['setup']:.0%} T:{w['technical']:.0%} PP:{w['pre_pump']:.0%} "
                f"F:{w['fundamental']:.0%} C:{w['catalyst']:.0%} "
                f"-> WinR:{r['win_rate']:.0f}% Avg:{r['avg_return']:+.1f}% "
                f"n={r['total_picks']}"
            )

        return {
            "best": best,
            "top_10": weight_results[:10],
            "correlations": correlations,
        }

    def _compute_score_correlations(self, all_picks, eval_hold=5):
        """Pearson correlation between each sub-score and forward return."""
        scores = {
            "setup": [], "technical": [], "pre_pump": [],
            "fundamental": [], "catalyst": [],
        }
        returns = []

        for p in all_picks:
            if len(p["daily_path"]) < eval_hold:
                continue
            exit_price = p["daily_path"][eval_hold - 1]["close"]
            ret = ((exit_price - p["entry_price"]) / p["entry_price"]) * 100
            returns.append(ret)
            for key in scores:
                scores[key].append(p["sub_scores"][key])

        if len(returns) < 30:
            return {}

        correlations = {}
        for key in scores:
            vals = scores[key]
            n = len(returns)
            sum_x = sum(vals)
            sum_y = sum(returns)
            sum_xy = sum(x * y for x, y in zip(vals, returns))
            sum_x2 = sum(x * x for x in vals)
            sum_y2 = sum(y * y for y in returns)

            num = n * sum_xy - sum_x * sum_y
            den_sq = (n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)
            if den_sq > 0:
                correlations[key] = round(num / den_sq ** 0.5, 4)
            else:
                correlations[key] = 0.0

        self._log(f"\n  Sub-score correlations with {eval_hold}d forward return:")
        for key, corr in sorted(correlations.items(),
                                key=lambda x: abs(x[1]), reverse=True):
            bar_len = int(abs(corr) * 40)
            bar = ("+" if corr >= 0 else "-") * bar_len
            self._log(f"    {key:>15}: r={corr:+.4f} {bar}")

        return correlations

    # ─────────────────────────────────────────────────────────
    # PHASE 7: RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────

    def _generate_recommendations(self, sell_results, weight_results):
        """Combine sell + weight optimization into final recommendations."""
        recs = {}

        best_sell = sell_results.get("best")
        if best_sell:
            recs["sell_strategy"] = {
                "SELL_MAX_HOLD_DAYS": best_sell["hold_days"],
                "SELL_STOP_LOSS_PCT": float(best_sell["stop_loss"]),
                "SELL_TAKE_PROFIT_PCT": float(best_sell["take_profit"]),
                "SELL_TRAILING_STOP_ACTIVATE": float(best_sell["trail_activate"]),
                "SELL_TRAILING_STOP_DISTANCE": float(best_sell["trail_distance"]),
                "STAGE2_RETURN_TOP_N": best_sell["top_n"],
                "expected_win_rate": best_sell["win_rate"],
                "expected_avg_return": best_sell["avg_return"],
            }

        best_weights = weight_results.get("best")
        if best_weights:
            recs["scoring"] = {
                "weights": best_weights["weights"],
                "MIN_RECOMMENDATION_SCORE": best_weights["min_score"],
                "expected_win_rate": best_weights["win_rate"],
                "expected_avg_return": best_weights["avg_return"],
            }

        recs["correlations"] = weight_results.get("correlations", {})

        # Print comparison: current vs recommended
        self._log(f"\n{'=' * 70}")
        self._log(f"  RECOMMENDED CONFIGURATION")
        self._log(f"{'=' * 70}")

        self._log(f"\n  === Sell Strategy ===")
        if "sell_strategy" in recs:
            s = recs["sell_strategy"]
            self._log(f"    Hold Period:      {s['SELL_MAX_HOLD_DAYS']} trading days")
            if s["SELL_STOP_LOSS_PCT"] != 0:
                self._log(f"    Stop-Loss:        {s['SELL_STOP_LOSS_PCT']}%")
            else:
                self._log(f"    Stop-Loss:        disabled")
            if s["SELL_TAKE_PROFIT_PCT"] != 0:
                self._log(f"    Take-Profit:      +{s['SELL_TAKE_PROFIT_PCT']}%")
            else:
                self._log(f"    Take-Profit:      disabled")
            if s["SELL_TRAILING_STOP_ACTIVATE"] > 0:
                self._log(
                    f"    Trailing Stop:    activate at +{s['SELL_TRAILING_STOP_ACTIVATE']}%, "
                    f"trail {s['SELL_TRAILING_STOP_DISTANCE']}% from peak"
                )
            else:
                self._log(f"    Trailing Stop:    disabled")
            self._log(f"    Top N Picks:      {s['STAGE2_RETURN_TOP_N']}")
            self._log(f"    Expected Return:  {s['expected_win_rate']:.0f}% win, "
                       f"{s['expected_avg_return']:+.1f}% avg")

        self._log(f"\n  === Scoring Weights ===")
        if "scoring" in recs:
            w = recs["scoring"]
            self._log(f"    {'Dimension':>15}  {'Current':>8}  {'Recommended':>11}")
            self._log(f"    {'-' * 15}  {'-' * 8}  {'-' * 11}")
            for k in ["setup", "technical", "pre_pump", "fundamental", "catalyst"]:
                cur = WEIGHTS.get(k, 0)
                rec = w["weights"].get(k, 0)
                changed = " *" if abs(cur - rec) > 0.01 else ""
                self._log(f"    {k:>15}  {cur:>7.0%}  {rec:>10.0%}{changed}")
            self._log(f"    Min Score:        {MIN_RECOMMENDATION_SCORE} -> "
                       f"{w['MIN_RECOMMENDATION_SCORE']}")
            self._log(f"    Expected Return:  {w['expected_win_rate']:.0f}% win, "
                       f"{w['expected_avg_return']:+.1f}% avg")

        corrs = recs.get("correlations", {})
        if corrs:
            self._log(f"\n  === Factor Predictiveness ===")
            for k, v in sorted(corrs.items(),
                                key=lambda x: abs(x[1]), reverse=True):
                bar_len = int(abs(v) * 40)
                bar = ("+" if v >= 0 else "-") * bar_len
                self._log(f"    {k:>15}: {v:+.4f} {bar}")

        return recs

    # ─────────────────────────────────────────────────────────
    # SAVE RESULTS
    # ─────────────────────────────────────────────────────────

    def _save_results(self, result):
        os.makedirs("runs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Text report
        txt_path = os.path.join(
            "runs", f"optimize_{timestamp}_v{ALGORITHM_VERSION}.txt"
        )
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                for line in result.get("report_lines", []):
                    f.write(line + "\n")
            self._log(f"\n  Report saved to {txt_path}")
        except Exception as e:
            logger.debug(f"Failed to save report: {e}")

        # JSON results (for programmatic use)
        json_path = os.path.join(
            "runs", f"optimize_{timestamp}_v{ALGORITHM_VERSION}.json"
        )
        try:
            serializable = {
                "recommendations": result.get("recommendations", {}),
                "sell_top_20": result.get("sell_optimization", {}).get("top_20", []),
                "weight_top_10": result.get("weight_optimization", {}).get("top_10", []),
                "correlations": result.get("weight_optimization", {}).get("correlations", {}),
                "total_picks": result.get("total_picks", 0),
                "dates_tested": result.get("dates_tested", 0),
                "version": ALGORITHM_VERSION,
                "timestamp": timestamp,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            self._log(f"  JSON results saved to {json_path}")
        except Exception as e:
            logger.debug(f"Failed to save JSON: {e}")


# ─── Sell strategy simulation ────────────────────────────────

def _simulate_strategy(pick, hold_days, stop_loss_pct, take_profit_pct,
                       trail_activate_pct, trail_distance_pct):
    """
    Simulate a sell strategy on one pick's daily price path.
    Returns (return_pct, exit_day).
    """
    entry_price = pick["entry_price"]
    path = pick["daily_path"]

    if not path or hold_days < 1:
        return 0.0, 0

    actual_days = min(hold_days, len(path))
    peak_price = entry_price
    trailing_active = False

    for i in range(actual_days):
        day = path[i]

        # Update peak for trailing stop
        if day["high"] > peak_price:
            peak_price = day["high"]

        # Check stop-loss (on intraday low)
        if stop_loss_pct != 0:
            sl_price = entry_price * (1 + stop_loss_pct / 100)
            if day["low"] <= sl_price:
                return stop_loss_pct, i + 1

        # Check take-profit (on intraday high)
        if take_profit_pct != 0:
            tp_price = entry_price * (1 + take_profit_pct / 100)
            if day["high"] >= tp_price:
                return take_profit_pct, i + 1

        # Check trailing stop
        if trail_activate_pct > 0 and trail_distance_pct > 0:
            gain = ((peak_price - entry_price) / entry_price) * 100
            if gain >= trail_activate_pct:
                trailing_active = True
            if trailing_active:
                trail_price = peak_price * (1 - trail_distance_pct / 100)
                if day["low"] <= trail_price:
                    ret = ((trail_price - entry_price) / entry_price) * 100
                    return ret, i + 1

    # Held to end, sell at close on final day
    exit_price = path[actual_days - 1]["close"]
    ret = ((exit_price - entry_price) / entry_price) * 100
    return ret, actual_days


# ─── Apply optimized config ──────────────────────────────────

def apply_optimized_config(recommendations: dict) -> str:
    """
    Save optimized parameters to optimized_config.json.
    config.py will load and apply these at import time.
    Returns the path to the saved file.
    """
    config = {}

    sell = recommendations.get("sell_strategy", {})
    for key in ["SELL_MAX_HOLD_DAYS", "SELL_STOP_LOSS_PCT", "SELL_TAKE_PROFIT_PCT",
                "SELL_TRAILING_STOP_ACTIVATE", "SELL_TRAILING_STOP_DISTANCE",
                "STAGE2_RETURN_TOP_N"]:
        if key in sell:
            config[key] = sell[key]

    scoring = recommendations.get("scoring", {})
    if "weights" in scoring:
        config["weights"] = scoring["weights"]
    if "MIN_RECOMMENDATION_SCORE" in scoring:
        config["MIN_RECOMMENDATION_SCORE"] = scoring["MIN_RECOMMENDATION_SCORE"]

    path = os.path.join(os.path.dirname(__file__), "..", "..", "optimized_config.json")
    path = os.path.normpath(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return path
