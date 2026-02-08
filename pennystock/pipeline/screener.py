"""
Two-stage stock screening pipeline.

Stage 1: Fast technical filter on all stocks (eliminates ~90%)
Stage 2: Deep multi-factor analysis on top candidates
"""

import time

from loguru import logger

from pennystock.config import STAGE1_KEEP_TOP_N, STAGE2_RETURN_TOP_N
from pennystock.data.finviz_client import get_penny_stocks
from pennystock.data.yahoo_client import get_price_history
from pennystock.scoring.scorer import score_stock_technical_only, score_stock


def run_screen(
    min_price: float = None,
    max_price: float = None,
    min_volume: int = None,
    stage1_top_n: int = None,
    stage2_top_n: int = None,
    skip_stage2: bool = False,
    progress_callback=None,
) -> dict:
    """
    Run the full two-stage screening pipeline.

    Args:
        min_price: Minimum stock price filter.
        max_price: Maximum stock price filter.
        min_volume: Minimum average volume filter.
        stage1_top_n: How many stocks to pass to Stage 2.
        stage2_top_n: How many final picks to return.
        skip_stage2: If True, return Stage 1 results only (fast mode).
        progress_callback: Optional callable(message: str) for progress updates.

    Returns:
        {
            "stage1_results": list[dict],
            "stage2_results": list[dict],  # Only if skip_stage2=False
            "final_picks": list[dict],
            "stats": dict,
        }
    """
    stage1_top_n = stage1_top_n or STAGE1_KEEP_TOP_N
    stage2_top_n = stage2_top_n or STAGE2_RETURN_TOP_N

    def _log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    start_time = time.time()

    # ── Discover stocks ─────────────────────────────────────────────
    _log("Discovering penny stocks via Finviz...")
    stocks = get_penny_stocks(min_price, max_price, min_volume)

    if not stocks:
        _log("No stocks found. Check filters or network connection.")
        return {"stage1_results": [], "stage2_results": [], "final_picks": [], "stats": {}}

    _log(f"Found {len(stocks)} penny stocks. Starting Stage 1 technical filter...")

    # ── Stage 1: Fast Technical Screen ──────────────────────────────
    stage1_results = []
    stage1_start = time.time()

    for i, stock in enumerate(stocks):
        ticker = stock["ticker"]

        try:
            result = score_stock_technical_only(ticker)
            if result["valid"]:
                result["price"] = stock.get("price", 0)
                result["volume"] = stock.get("volume", 0)
                result["sector"] = stock.get("sector", "")
                result["company"] = stock.get("company", "")
                stage1_results.append(result)
        except Exception as e:
            logger.debug(f"Stage 1 failed for {ticker}: {e}")

        if (i + 1) % 25 == 0:
            elapsed = time.time() - stage1_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(stocks) - i - 1) / rate if rate > 0 else 0
            _log(f"  Stage 1: {i+1}/{len(stocks)} ({rate:.1f}/sec, ~{remaining:.0f}s remaining)")

        time.sleep(0.1)  # Small delay to be nice to Yahoo

    # Sort by technical score
    stage1_results.sort(key=lambda x: x["technical_score"], reverse=True)
    stage1_elapsed = time.time() - stage1_start

    _log(f"Stage 1 complete: {len(stage1_results)} valid stocks scored in {stage1_elapsed:.0f}s")
    if stage1_results:
        top5 = [f"{r['ticker']}:{r['technical_score']:.0f}" for r in stage1_results[:5]]
        _log(f"  Top 5 technical: {', '.join(top5)}")

    if skip_stage2:
        picks = stage1_results[:stage2_top_n]
        return {
            "stage1_results": stage1_results,
            "stage2_results": [],
            "final_picks": picks,
            "stats": {
                "total_screened": len(stocks),
                "stage1_passed": len(stage1_results),
                "stage1_time_sec": round(stage1_elapsed),
                "total_time_sec": round(time.time() - start_time),
            },
        }

    # ── Stage 2: Deep Multi-Factor Analysis ─────────────────────────
    top_candidates = stage1_results[:stage1_top_n]
    _log(f"Stage 2: Deep analysis on top {len(top_candidates)} stocks...")

    stage2_results = []
    stage2_start = time.time()

    for i, candidate in enumerate(top_candidates):
        ticker = candidate["ticker"]
        sector = candidate.get("sector", "")

        try:
            result = score_stock(ticker, sector=sector)
            result["price"] = candidate.get("price", 0)
            result["volume"] = candidate.get("volume", 0)
            result["company"] = candidate.get("company", "")
            stage2_results.append(result)
        except Exception as e:
            logger.warning(f"Stage 2 failed for {ticker}: {e}")

        if (i + 1) % 5 == 0:
            elapsed = time.time() - stage2_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(top_candidates) - i - 1) / rate if rate > 0 else 0
            _log(f"  Stage 2: {i+1}/{len(top_candidates)} (~{remaining:.0f}s remaining)")

    # Sort by final composite score
    stage2_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    stage2_elapsed = time.time() - stage2_start

    final_picks = stage2_results[:stage2_top_n]
    total_elapsed = time.time() - start_time

    _log(f"Stage 2 complete in {stage2_elapsed:.0f}s")
    _log(f"Total pipeline time: {total_elapsed:.0f}s")

    if final_picks:
        _log("=" * 60)
        _log(f"TOP {len(final_picks)} PICKS:")
        _log("=" * 60)
        for j, pick in enumerate(final_picks, 1):
            _log(f"  #{j}. {pick['ticker']} - ${pick.get('price', '?')} "
                 f"(Score: {pick['final_score']:.1f})")
            ss = pick.get("sub_scores", {})
            _log(f"      Tech:{ss.get('technical', 0):.0f} "
                 f"Sent:{ss.get('sentiment', 0):.0f} "
                 f"Fund:{ss.get('fundamental', 0):.0f} "
                 f"Cat:{ss.get('catalyst', 0):.0f} "
                 f"Mkt:{ss.get('market_ctx', 0):.0f} "
                 f"Sqz:{ss.get('short_squeeze', 0):.0f}")

    return {
        "stage1_results": stage1_results,
        "stage2_results": stage2_results,
        "final_picks": final_picks,
        "stats": {
            "total_screened": len(stocks),
            "stage1_passed": len(stage1_results),
            "stage2_analyzed": len(stage2_results),
            "stage1_time_sec": round(stage1_elapsed),
            "stage2_time_sec": round(stage2_elapsed),
            "total_time_sec": round(total_elapsed),
        },
    }
