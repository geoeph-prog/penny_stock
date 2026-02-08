"""
Pattern learner: analyzes historical winners AND losers to find
discriminative technical features.

Unlike the original OPTT-only calibration, this collects features from
many stocks and computes statistical differences between winners and losers.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from loguru import logger

from pennystock.data.finviz_client import get_high_gainers
from pennystock.data.yahoo_client import get_price_history, get_batch_history
from pennystock.analysis.technical import extract_features


PATTERN_FILE = "learned_patterns.json"


def learn_patterns(
    winner_tickers: list = None,
    loser_tickers: list = None,
    max_tickers: int = 50,
) -> dict:
    """
    Learn discriminative patterns from winners vs losers.

    If tickers aren't provided, discovers them via Finviz.

    Returns:
        {
            "winners": {feature: {"mean": x, "std": y}},
            "losers": {feature: {"mean": x, "std": y}},
            "discriminative_features": [{feature, winner_mean, loser_mean, separation}],
            "winners_analyzed": int,
            "losers_analyzed": int,
            "date": str,
        }
    """
    if winner_tickers is None:
        logger.info("Discovering recent winners via Finviz...")
        winner_tickers = get_high_gainers(months=6, min_gain_pct=100)

    if loser_tickers is None:
        logger.info("Discovering recent losers via Finviz...")
        loser_tickers = _get_recent_losers()

    winner_tickers = winner_tickers[:max_tickers]
    loser_tickers = loser_tickers[:max_tickers]

    logger.info(f"Analyzing {len(winner_tickers)} winners and {len(loser_tickers)} losers")

    # Extract features for both groups
    winner_features = _extract_group_features(winner_tickers, "winners")
    loser_features = _extract_group_features(loser_tickers, "losers")

    if not winner_features or not loser_features:
        logger.warning("Insufficient data for pattern learning")
        return {"error": "insufficient data"}

    # Compute statistics for each group
    winner_stats = _compute_stats(winner_features)
    loser_stats = _compute_stats(loser_features)

    # Find most discriminative features
    discriminative = _find_discriminative_features(winner_stats, loser_stats)

    result = {
        "winners": winner_stats,
        "losers": loser_stats,
        "discriminative_features": discriminative,
        "winners_analyzed": len(winner_features),
        "losers_analyzed": len(loser_features),
        "date": time.strftime("%Y-%m-%d"),
    }

    # Save to file
    _save_patterns(result)

    return result


def _get_recent_losers() -> list:
    """Find stocks that lost significantly (negative examples)."""
    try:
        from finvizfinance.screener.overview import Overview

        foverview = Overview()
        filters = {
            "Price": "Under $5",
            "Average Volume": "Over 50K",
            "Performance": "Half -50%",
        }
        foverview.set_filter(filters_dict=filters)
        df = foverview.screener_view()

        if df is None or df.empty:
            return []

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df["ticker"].tolist() if "ticker" in df.columns else []

    except Exception as e:
        logger.warning(f"Failed to find losers via Finviz: {e}")
        return []


def _extract_group_features(tickers: list, label: str) -> list:
    """Extract technical features for a group of tickers."""
    features_list = []

    for i, ticker in enumerate(tickers):
        try:
            hist = get_price_history(ticker, period="6mo")
            if hist is None or hist.empty:
                continue

            features = extract_features(hist)
            if features:
                features["ticker"] = ticker
                features_list.append(features)

        except Exception as e:
            logger.debug(f"Failed to extract features for {ticker}: {e}")

        if (i + 1) % 10 == 0:
            logger.info(f"  {label}: {i+1}/{len(tickers)} processed")

        time.sleep(0.3)

    logger.info(f"  {label}: extracted features for {len(features_list)}/{len(tickers)} tickers")
    return features_list


def _compute_stats(features_list: list) -> dict:
    """Compute mean and std for each feature across a group."""
    df = pd.DataFrame(features_list)
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        values = df[col].dropna()
        if len(values) > 0:
            stats[col] = {
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std()), 4),
                "median": round(float(values.median()), 4),
                "count": int(len(values)),
            }
    return stats


def _find_discriminative_features(winner_stats: dict, loser_stats: dict) -> list:
    """
    Find features that best separate winners from losers.
    Uses normalized mean difference (Cohen's d-like measure).
    """
    discriminative = []

    all_features = set(winner_stats.keys()) & set(loser_stats.keys())
    for feature in all_features:
        w = winner_stats[feature]
        l = loser_stats[feature]

        # Pooled std
        pooled_std = np.sqrt((w["std"] ** 2 + l["std"] ** 2) / 2) if (w["std"] + l["std"]) > 0 else 1

        separation = abs(w["mean"] - l["mean"]) / pooled_std if pooled_std > 0 else 0

        discriminative.append({
            "feature": feature,
            "winner_mean": w["mean"],
            "loser_mean": l["mean"],
            "separation": round(separation, 3),
            "direction": "higher" if w["mean"] > l["mean"] else "lower",
        })

    # Sort by separation (most discriminative first)
    discriminative.sort(key=lambda x: x["separation"], reverse=True)
    return discriminative


def _save_patterns(patterns: dict):
    """Save learned patterns to JSON file."""
    try:
        with open(PATTERN_FILE, "w") as f:
            json.dump(patterns, f, indent=2)
        logger.info(f"Patterns saved to {PATTERN_FILE}")
    except Exception as e:
        logger.error(f"Failed to save patterns: {e}")


def load_patterns() -> dict:
    """Load previously learned patterns."""
    if not os.path.exists(PATTERN_FILE):
        return {}
    try:
        with open(PATTERN_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}
