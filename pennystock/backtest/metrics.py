"""
Backtesting performance metrics.

Computes win rate, average returns, Sharpe ratio, max drawdown, and
other statistics to evaluate the scoring algorithm.
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_metrics(
    picks: pd.DataFrame,
    hold_days: int = 20,
    winner_threshold: float = 20.0,
    loser_threshold: float = -15.0,
) -> dict:
    """
    Compute performance metrics for a set of backtested picks.

    Args:
        picks: DataFrame with at least 'return_{hold_days}d' column.
        hold_days: Holding period to evaluate.
        winner_threshold: % gain to count as a "win".
        loser_threshold: % loss to count as a "loss".

    Returns:
        {
            "total_picks": int,
            "win_rate": float,
            "loss_rate": float,
            "avg_return": float,
            "median_return": float,
            "best_return": float,
            "worst_return": float,
            "sharpe_ratio": float,
            "profit_factor": float,
            "max_consecutive_losses": int,
        }
    """
    return_col = f"return_{hold_days}d"

    if return_col not in picks.columns:
        logger.warning(f"Column {return_col} not found in picks DataFrame")
        return {"error": f"missing {return_col}"}

    returns = picks[return_col].dropna()

    if len(returns) == 0:
        return {"error": "no return data"}

    winners = returns[returns >= winner_threshold]
    losers = returns[returns <= loser_threshold]
    neutrals = returns[(returns > loser_threshold) & (returns < winner_threshold)]

    avg_return = returns.mean()
    std_return = returns.std()

    # Sharpe ratio (simplified, using 0 as risk-free rate)
    sharpe = avg_return / std_return if std_return > 0 else 0

    # Profit factor (gross profits / gross losses)
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max consecutive losses
    is_loss = (returns <= loser_threshold).values
    max_consec_losses = _max_consecutive(is_loss)

    return {
        "hold_days": hold_days,
        "total_picks": len(returns),
        "winners": len(winners),
        "losers": len(losers),
        "neutral": len(neutrals),
        "win_rate": round(len(winners) / len(returns) * 100, 1),
        "loss_rate": round(len(losers) / len(returns) * 100, 1),
        "avg_return_pct": round(avg_return, 2),
        "median_return_pct": round(returns.median(), 2),
        "std_return_pct": round(std_return, 2),
        "best_return_pct": round(returns.max(), 2),
        "worst_return_pct": round(returns.min(), 2),
        "sharpe_ratio": round(sharpe, 3),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "max_consecutive_losses": max_consec_losses,
        "winner_threshold_pct": winner_threshold,
        "loser_threshold_pct": loser_threshold,
    }


def _max_consecutive(bool_array) -> int:
    """Find maximum consecutive True values in a boolean array."""
    max_count = 0
    current = 0
    for val in bool_array:
        if val:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


def compare_strategies(
    dataset: pd.DataFrame,
    score_column: str,
    hold_days: int = 20,
    top_n_values: list = None,
) -> list:
    """
    Compare the algorithm's top picks against random selection.

    Tests: "Does picking higher-scored stocks actually produce better returns?"

    Args:
        dataset: Full backtest dataset with scores and returns.
        score_column: Column name containing the composite score.
        hold_days: Holding period to evaluate.
        top_n_values: List of top-N cutoffs to test (e.g., [5, 10, 20, 50]).

    Returns:
        List of comparison dicts for each top_n cutoff.
    """
    top_n_values = top_n_values or [5, 10, 20, 50]
    return_col = f"return_{hold_days}d"

    if return_col not in dataset.columns or score_column not in dataset.columns:
        return [{"error": "missing required columns"}]

    # Sort by score
    sorted_data = dataset.dropna(subset=[return_col, score_column]).sort_values(
        score_column, ascending=False
    )

    comparisons = []

    for top_n in top_n_values:
        if top_n > len(sorted_data):
            continue

        top_picks = sorted_data.head(top_n)
        bottom_picks = sorted_data.tail(top_n)
        random_sample = sorted_data.sample(n=min(top_n, len(sorted_data)))

        top_metrics = compute_metrics(top_picks, hold_days)
        bottom_metrics = compute_metrics(bottom_picks, hold_days)
        random_metrics = compute_metrics(random_sample, hold_days)

        comparisons.append({
            "top_n": top_n,
            "top_picks_avg_return": top_metrics.get("avg_return_pct", 0),
            "bottom_picks_avg_return": bottom_metrics.get("avg_return_pct", 0),
            "random_avg_return": random_metrics.get("avg_return_pct", 0),
            "top_picks_win_rate": top_metrics.get("win_rate", 0),
            "bottom_picks_win_rate": bottom_metrics.get("win_rate", 0),
            "random_win_rate": random_metrics.get("win_rate", 0),
            "edge_vs_random": round(
                top_metrics.get("avg_return_pct", 0) - random_metrics.get("avg_return_pct", 0), 2
            ),
        })

    return comparisons


def format_report(metrics: dict) -> str:
    """Format metrics dict into a readable report string."""
    if "error" in metrics:
        return f"Error: {metrics['error']}"

    lines = [
        f"Backtest Results ({metrics.get('hold_days', '?')}-day hold period)",
        "=" * 50,
        f"Total picks evaluated:   {metrics['total_picks']}",
        f"Winners (>{metrics['winner_threshold_pct']}%):  {metrics['winners']} ({metrics['win_rate']}%)",
        f"Losers (<{metrics['loser_threshold_pct']}%):   {metrics['losers']} ({metrics['loss_rate']}%)",
        f"Neutral:                 {metrics['neutral']}",
        "",
        f"Average return:          {metrics['avg_return_pct']:+.2f}%",
        f"Median return:           {metrics['median_return_pct']:+.2f}%",
        f"Std deviation:           {metrics['std_return_pct']:.2f}%",
        f"Best single pick:        {metrics['best_return_pct']:+.2f}%",
        f"Worst single pick:       {metrics['worst_return_pct']:+.2f}%",
        "",
        f"Sharpe ratio:            {metrics['sharpe_ratio']:.3f}",
        f"Profit factor:           {metrics['profit_factor']}",
        f"Max consecutive losses:  {metrics['max_consecutive_losses']}",
    ]
    return "\n".join(lines)
