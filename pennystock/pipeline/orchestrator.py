"""
Main orchestrator: coordinates screening, pattern learning, and backtesting.
"""

import json
import time

from loguru import logger

from pennystock.pipeline.screener import run_screen
from pennystock.scoring.pattern_learner import learn_patterns, load_patterns
from pennystock.storage.db import Database


def run_daily_picks(
    min_price: float = None,
    max_price: float = None,
    min_volume: int = None,
    top_n: int = 5,
    fast_mode: bool = False,
    progress_callback=None,
) -> dict:
    """
    Run the full daily stock picking pipeline.

    Args:
        min_price: Minimum price filter.
        max_price: Maximum price filter.
        min_volume: Minimum volume filter.
        top_n: Number of picks to return.
        fast_mode: Skip Stage 2 for faster results (technical only).
        progress_callback: Optional progress update callable.

    Returns:
        Full results dict from screener with picks saved to database.
    """
    db = Database()

    results = run_screen(
        min_price=min_price,
        max_price=max_price,
        min_volume=min_volume,
        stage2_top_n=top_n,
        skip_stage2=fast_mode,
        progress_callback=progress_callback,
    )

    # Save results to database
    for pick in results.get("final_picks", []):
        db.save_pick(pick)

    db.save_run(results.get("stats", {}))

    return results


def run_pattern_learning(progress_callback=None) -> dict:
    """
    Run pattern learning: analyze current winners AND losers.

    Returns learned patterns dict.
    """
    def _log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    _log("Starting pattern learning...")
    _log("This analyzes winning AND losing stocks to find discriminative patterns.")

    start = time.time()
    patterns = learn_patterns()
    elapsed = time.time() - start

    if "error" in patterns:
        _log(f"Pattern learning failed: {patterns['error']}")
        return patterns

    _log(f"Pattern learning complete in {elapsed:.0f}s")
    _log(f"  Winners analyzed: {patterns['winners_analyzed']}")
    _log(f"  Losers analyzed: {patterns['losers_analyzed']}")

    disc = patterns.get("discriminative_features", [])
    if disc:
        _log("Most discriminative features (winners vs losers):")
        for feat in disc[:5]:
            direction = "higher" if feat["direction"] == "higher" else "lower"
            _log(f"  {feat['feature']}: winners have {direction} values "
                 f"(W:{feat['winner_mean']:.2f} vs L:{feat['loser_mean']:.2f}, "
                 f"separation:{feat['separation']:.2f})")

    return patterns


def show_last_results(n: int = 5) -> list:
    """Retrieve the most recent picks from the database."""
    db = Database()
    return db.get_recent_picks(n)
