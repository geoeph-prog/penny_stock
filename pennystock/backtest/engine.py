"""
Backtesting engine.

Simulates the scoring algorithm over historical data to validate
whether higher scores actually predict better returns.
"""

import time

import pandas as pd
from loguru import logger

from pennystock.backtest.data_collector import build_backtest_dataset
from pennystock.backtest.metrics import compute_metrics, compare_strategies, format_report
from pennystock.analysis.technical import analyze as tech_analyze
from pennystock.data.finviz_client import get_penny_stocks, get_high_gainers
from pennystock.data.yahoo_client import get_price_history
from pennystock.config import BACKTEST_HOLD_DAYS, BACKTEST_WINNER_THRESHOLD, BACKTEST_LOSER_THRESHOLD


def run_backtest(
    tickers: list = None,
    lookback: str = "1y",
    hold_days: list = None,
    top_n: int = 10,
    progress_callback=None,
) -> dict:
    """
    Run a full backtest of the technical scoring algorithm.

    1. Collects historical data for tickers
    2. Computes technical scores at each evaluation point
    3. Simulates picking top-scored stocks
    4. Measures actual forward returns

    Args:
        tickers: List of tickers to backtest. If None, discovers via Finviz.
        lookback: How far back to look (e.g., "1y", "2y").
        hold_days: List of holding periods to evaluate.
        top_n: Number of top picks to evaluate.
        progress_callback: Optional progress callback.

    Returns:
        {
            "dataset_size": int,
            "metrics_by_period": dict[int, dict],
            "strategy_comparison": list,
            "report": str,
        }
    """
    hold_days = hold_days or BACKTEST_HOLD_DAYS

    def _log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    start = time.time()

    # ── Discover tickers if needed ──────────────────────────────────
    if tickers is None:
        _log("Discovering tickers for backtest...")
        stocks = get_penny_stocks()
        tickers = [s["ticker"] for s in stocks[:100]]  # Cap at 100 for speed

        # Also include known winners and losers for balanced dataset
        winners = get_high_gainers(months=6, min_gain_pct=100)
        tickers.extend(winners[:30])
        tickers = list(set(tickers))  # Deduplicate

    _log(f"Backtesting {len(tickers)} tickers over {lookback}...")

    # ── Build dataset ───────────────────────────────────────────────
    dataset = build_backtest_dataset(
        tickers=tickers,
        lookback_period=lookback,
        evaluation_points=5,
        progress_callback=progress_callback,
    )

    if dataset.empty:
        _log("Backtest failed: no data collected")
        return {"error": "no data"}

    _log(f"Dataset: {len(dataset)} samples")

    # ── Score each sample using technical analysis ──────────────────
    # The features are already computed; we need a composite score
    # Use a simple weighted average of available features
    _log("Computing technical scores for all samples...")

    feature_cols = ["rsi", "volume_spike", "price_trend_5d", "price_trend_20d", "volatility_20d"]
    available_cols = [c for c in feature_cols if c in dataset.columns]

    if not available_cols:
        _log("Backtest failed: no features available")
        return {"error": "no features"}

    # Normalize features to 0-1 range for scoring
    normalized = dataset[available_cols].copy()
    for col in available_cols:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if col_max > col_min:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 0.5

    # For RSI, ideal is ~50 (distance from 50 should be minimized)
    if "rsi" in normalized.columns:
        normalized["rsi"] = 1 - abs(dataset["rsi"].fillna(50) - 50) / 50

    # For volatility, moderate is better than extreme
    if "volatility_20d" in normalized.columns:
        vol = dataset["volatility_20d"].fillna(0)
        # Sweet spot: 3-8% daily volatility
        normalized["volatility_20d"] = 1 - abs(vol - 5) / 10
        normalized["volatility_20d"] = normalized["volatility_20d"].clip(0, 1)

    dataset["technical_score"] = normalized.mean(axis=1) * 100

    # ── Compute metrics for each holding period ─────────────────────
    metrics_by_period = {}
    for days in hold_days:
        return_col = f"return_{days}d"
        if return_col in dataset.columns:
            valid = dataset.dropna(subset=[return_col])
            top_picks = valid.nlargest(top_n, "technical_score")
            metrics = compute_metrics(
                top_picks, days,
                winner_threshold=BACKTEST_WINNER_THRESHOLD * 100,
                loser_threshold=BACKTEST_LOSER_THRESHOLD * 100,
            )
            metrics_by_period[days] = metrics
            _log(f"{days}d hold: avg return {metrics.get('avg_return_pct', 0):+.1f}%, "
                 f"win rate {metrics.get('win_rate', 0):.0f}%")

    # ── Strategy comparison ─────────────────────────────────────────
    _log("Comparing top picks vs random selection...")
    comparison = compare_strategies(
        dataset, "technical_score",
        hold_days=20,
        top_n_values=[5, 10, 20],
    )

    for comp in comparison:
        _log(f"  Top {comp['top_n']}: {comp['top_picks_avg_return']:+.1f}% avg return "
             f"vs random {comp['random_avg_return']:+.1f}% "
             f"(edge: {comp['edge_vs_random']:+.1f}%)")

    elapsed = time.time() - start
    _log(f"Backtest complete in {elapsed:.0f}s")

    # ── Generate report ─────────────────────────────────────────────
    report_lines = [
        "=" * 60,
        "BACKTEST REPORT",
        "=" * 60,
        f"Tickers evaluated: {len(tickers)}",
        f"Data samples: {len(dataset)}",
        f"Lookback period: {lookback}",
        f"Time elapsed: {elapsed:.0f}s",
        "",
    ]

    for days, metrics in metrics_by_period.items():
        report_lines.append(format_report(metrics))
        report_lines.append("")

    if comparison:
        report_lines.append("STRATEGY COMPARISON (20d hold)")
        report_lines.append("-" * 40)
        for comp in comparison:
            report_lines.append(
                f"  Top {comp['top_n']}: {comp['top_picks_avg_return']:+.1f}% "
                f"(win rate: {comp['top_picks_win_rate']:.0f}%) | "
                f"Random: {comp['random_avg_return']:+.1f}% "
                f"(win rate: {comp['random_win_rate']:.0f}%) | "
                f"Edge: {comp['edge_vs_random']:+.1f}%"
            )

    report = "\n".join(report_lines)

    return {
        "dataset_size": len(dataset),
        "metrics_by_period": metrics_by_period,
        "strategy_comparison": comparison,
        "report": report,
    }
