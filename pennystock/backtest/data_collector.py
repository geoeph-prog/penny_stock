"""
Historical data collector for backtesting.

Builds a dataset of past penny stocks with their features at a point in time,
and their subsequent returns. This lets us test whether the scoring algorithm
would have picked winners.
"""

import time

import numpy as np
import pandas as pd
from loguru import logger

from pennystock.data.yahoo_client import get_price_history
from pennystock.analysis.technical import extract_features
from pennystock.config import BACKTEST_HOLD_DAYS


def build_backtest_dataset(
    tickers: list,
    lookback_period: str = "1y",
    evaluation_points: int = 5,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Build a dataset for backtesting by:
    1. Getting extended price history for each ticker
    2. Picking several evaluation points within that history
    3. Computing features AT each evaluation point
    4. Computing forward returns FROM each evaluation point

    Args:
        tickers: List of tickers to include.
        lookback_period: How far back to get price data.
        evaluation_points: How many historical snapshots per ticker.
        progress_callback: Optional progress callback.

    Returns:
        DataFrame with columns:
            ticker, eval_date, features..., return_5d, return_10d, return_20d, return_30d
    """
    rows = []

    def _log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    _log(f"Building backtest dataset for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        try:
            hist = get_price_history(ticker, period=lookback_period)
            if hist is None or hist.empty or len(hist) < 90:
                continue

            # Pick evaluation points spread across the history
            # Leave enough room for forward returns
            max_forward = max(BACKTEST_HOLD_DAYS)
            usable_length = len(hist) - max_forward - 30  # Need 30 days of history for features

            if usable_length < 1:
                continue

            eval_indices = np.linspace(30, 30 + usable_length - 1, evaluation_points).astype(int)

            for idx in eval_indices:
                # Slice history up to this point (what we would have known)
                hist_slice = hist.iloc[:idx + 1]

                # Extract features at this point
                features = extract_features(hist_slice)
                if not features:
                    continue

                # Compute forward returns
                eval_price = hist["Close"].iloc[idx]
                if eval_price <= 0:
                    continue

                returns = {}
                for days in BACKTEST_HOLD_DAYS:
                    future_idx = idx + days
                    if future_idx < len(hist):
                        future_price = hist["Close"].iloc[future_idx]
                        returns[f"return_{days}d"] = round(
                            (future_price - eval_price) / eval_price * 100, 2
                        )
                    else:
                        returns[f"return_{days}d"] = None

                row = {
                    "ticker": ticker,
                    "eval_date": str(hist.index[idx].date()),
                    **features,
                    **returns,
                }
                rows.append(row)

        except Exception as e:
            logger.debug(f"Backtest data collection failed for {ticker}: {e}")

        if (i + 1) % 10 == 0:
            _log(f"  Collected data for {i+1}/{len(tickers)} tickers ({len(rows)} samples)")

        time.sleep(0.2)

    _log(f"Backtest dataset: {len(rows)} samples from {len(tickers)} tickers")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
