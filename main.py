#!/usr/bin/env python3
"""
Penny Stock Analyzer - CLI Entry Point

Usage:
    python main.py pick           # Find top penny stock picks (full analysis)
    python main.py pick --fast    # Quick technical-only picks
    python main.py learn          # Learn patterns from winners vs losers
    python main.py backtest       # Backtest the scoring algorithm
    python main.py history        # Show recent picks from database
    python main.py history --backtests  # View saved backtest results
"""

import argparse
import sys

from loguru import logger


def setup_logging(verbose: bool = False):
    """Configure loguru logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <level>{message}</level>",
    )
    logger.add(
        "pennystock.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
    )


def cmd_pick(args):
    """Run the stock picking pipeline."""
    from pennystock.pipeline.orchestrator import run_daily_picks

    results = run_daily_picks(
        min_price=args.min_price,
        max_price=args.max_price,
        min_volume=args.min_volume,
        top_n=args.top_n,
        fast_mode=args.fast,
    )

    picks = results.get("final_picks", [])
    stats = results.get("stats", {})

    if not picks:
        print("\nNo picks found. Try adjusting filters or check your network connection.")
        return

    print("\n" + "=" * 70)
    print(f"  TOP {len(picks)} PENNY STOCK PICKS")
    print("=" * 70)

    for i, pick in enumerate(picks, 1):
        ticker = pick.get("ticker", "?")
        score = pick.get("final_score", pick.get("technical_score", 0))
        price = pick.get("price", 0)
        company = pick.get("company", "")

        print(f"\n  #{i}. {ticker} - ${price:.2f}  (Score: {score:.1f})")
        if company:
            print(f"      {company}")

        ss = pick.get("sub_scores", {})
        if ss:
            parts = [f"{k.title()}: {v:.0f}" for k, v in ss.items()]
            print(f"      {' | '.join(parts)}")

        # Show key indicators if available
        tech = pick.get("technical", pick.get("indicators", {}))
        if tech and isinstance(tech, dict):
            indicators = []
            if tech.get("rsi"):
                indicators.append(f"RSI: {tech['rsi']}")
            if tech.get("volume_spike"):
                indicators.append(f"Vol Spike: {tech['volume_spike']}x")
            if tech.get("price_trend_20d"):
                indicators.append(f"20d Trend: {tech['price_trend_20d']:+.1f}%")
            if indicators:
                print(f"      {' | '.join(indicators)}")

    print(f"\n  Screened: {stats.get('total_screened', '?')} stocks")
    print(f"  Total time: {stats.get('total_time_sec', '?')}s")
    print("=" * 70)


def cmd_learn(args):
    """Run pattern learning."""
    from pennystock.pipeline.orchestrator import run_pattern_learning

    patterns = run_pattern_learning()

    if "error" in patterns:
        print(f"\nPattern learning failed: {patterns['error']}")
        return

    disc = patterns.get("discriminative_features", [])
    if disc:
        print("\n" + "=" * 60)
        print("  LEARNED PATTERNS: What separates winners from losers")
        print("=" * 60)
        for feat in disc[:10]:
            direction = "HIGHER" if feat["direction"] == "higher" else "LOWER"
            print(f"  {feat['feature']:20s}: Winners have {direction} values")
            print(f"    {'':20s}  Winner avg: {feat['winner_mean']:8.2f}")
            print(f"    {'':20s}  Loser avg:  {feat['loser_mean']:8.2f}")
            print(f"    {'':20s}  Separation: {feat['separation']:8.3f}")
        print("=" * 60)


def cmd_backtest(args):
    """Run backtesting."""
    from pennystock.backtest.engine import run_backtest

    results = run_backtest(
        lookback=args.lookback,
        top_n=args.top_n,
    )

    if "error" in results:
        print(f"\nBacktest failed: {results['error']}")
        return

    print("\n" + results.get("report", "No report generated"))

    # Save to database
    from pennystock.storage.db import Database
    db = Database()
    db.save_backtest(
        results.get("report", ""),
        results.get("metrics_by_period", {}),
    )


def cmd_history(args):
    """Show recent picks or backtest results."""
    from pennystock.storage.db import Database

    db = Database()

    if args.backtests:
        _show_backtest_history(db, args)
        return

    from pennystock.pipeline.orchestrator import show_last_results

    picks = show_last_results(args.n)

    if not picks:
        print("\nNo picks in history. Run 'python main.py pick' first.")
        return

    print(f"\n  Last {len(picks)} picks:")
    print(f"  {'Date':<20} {'Ticker':<8} {'Score':>6} {'Price':>8}")
    print("  " + "-" * 46)
    for pick in picks:
        print(f"  {pick['timestamp'][:19]:<20} {pick['ticker']:<8} "
              f"{pick['final_score']:>6.1f} ${pick['price']:>7.2f}")


def _show_backtest_history(db, args):
    """Show saved backtest results."""
    if args.backtest_id:
        # Show a specific backtest in full
        result = db.get_backtest_by_id(args.backtest_id)
        if not result:
            print(f"\nNo backtest found with ID {args.backtest_id}")
            return
        print(f"\n  Backtest #{result['id']} - {result['timestamp'][:19]}")
        print(result["report"])
        return

    # List recent backtests
    backtests = db.get_backtest_history(args.n)

    if not backtests:
        print("\nNo backtest results in history. Run 'python main.py backtest' first.")
        return

    print(f"\n  Last {len(backtests)} backtest runs:")
    print(f"  {'ID':>4}  {'Date':<20} {'Summary'}")
    print("  " + "-" * 60)
    for bt in backtests:
        # Extract a one-line summary from the report
        metrics = bt.get("metrics", {})
        summary_parts = []
        for period, m in metrics.items():
            if isinstance(m, dict) and "avg_return_pct" in m:
                summary_parts.append(
                    f"{period}d: {m['avg_return_pct']:+.1f}% avg, "
                    f"{m.get('win_rate', 0):.0f}% win"
                )
        summary = " | ".join(summary_parts) if summary_parts else "(no metrics)"
        print(f"  {bt['id']:>4}  {bt['timestamp'][:19]:<20} {summary}")

    print(f"\n  View full report: python main.py history --backtests --id <ID>")


def main():
    parser = argparse.ArgumentParser(
        description="Penny Stock Analyzer - Find winning penny stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py pick                 # Full analysis, top 5 picks
  python main.py pick --fast -n 10    # Quick technical screen, top 10
  python main.py pick --min-price 0.10 --max-price 0.50
  python main.py learn                # Learn from winners vs losers
  python main.py backtest             # Validate algorithm historically
  python main.py history              # View recent picks
  python main.py history --backtests  # List saved backtest results
  python main.py history --backtests --id 1  # View full backtest report
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose debug output")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── pick command ────────────────────────────────────────────────
    pick_parser = subparsers.add_parser("pick", help="Find top penny stock picks")
    pick_parser.add_argument("--fast", action="store_true", help="Technical-only (skip deep analysis)")
    pick_parser.add_argument("-n", "--top-n", type=int, default=5, help="Number of picks (default: 5)")
    pick_parser.add_argument("--min-price", type=float, default=0.05, help="Min price (default: $0.05)")
    pick_parser.add_argument("--max-price", type=float, default=1.00, help="Max price (default: $1.00)")
    pick_parser.add_argument("--min-volume", type=int, default=50000, help="Min avg volume (default: 50000)")

    # ── learn command ───────────────────────────────────────────────
    subparsers.add_parser("learn", help="Learn patterns from winners vs losers")

    # ── backtest command ────────────────────────────────────────────
    bt_parser = subparsers.add_parser("backtest", help="Backtest the scoring algorithm")
    bt_parser.add_argument("--lookback", default="1y", help="Lookback period (default: 1y)")
    bt_parser.add_argument("-n", "--top-n", type=int, default=10, help="Top N picks to evaluate")

    # ── history command ─────────────────────────────────────────────
    hist_parser = subparsers.add_parser("history", help="Show recent picks or backtest results")
    hist_parser.add_argument("-n", type=int, default=10, help="Number of recent entries to show")
    hist_parser.add_argument("--backtests", action="store_true", help="Show backtest results instead of picks")
    hist_parser.add_argument("--id", dest="backtest_id", type=int, default=None, help="Show full report for a specific backtest ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)

    commands = {
        "pick": cmd_pick,
        "learn": cmd_learn,
        "backtest": cmd_backtest,
        "history": cmd_history,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
