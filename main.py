#!/usr/bin/env python3
"""
Penny Stock Analyzer

Usage:
    python main.py              # Launch GUI (default)
    python main.py --cli build  # Build algorithm from CLI
    python main.py --cli pick   # Pick stocks from CLI
    python main.py analyze GETY # Deep dive analysis on a single stock
    python main.py --cli history  # View past picks
"""

import argparse
import sys

from loguru import logger


def setup_logging(verbose=False):
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr, level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <level>{message}</level>",
    )
    logger.add("pennystock.log", level="DEBUG", rotation="10 MB", retention="7 days")


def cmd_build(args):
    from pennystock.algorithm import build_algorithm
    result = build_algorithm()
    if not result:
        print("\nAlgorithm build failed. Check log for details.")


def cmd_pick(args):
    from pennystock.algorithm import pick_stocks
    from pennystock.config import ALGORITHM_VERSION
    picks = pick_stocks(top_n=args.top_n)
    if not picks:
        print("\nNo picks found. Make sure you've built the algorithm first.")
        return

    print("\n" + "=" * 70)
    print(f"  TOP {len(picks)} PENNY STOCK PICKS (v{ALGORITHM_VERSION} MEGA-ALGORITHM)")
    print("=" * 70)
    for i, pick in enumerate(picks, 1):
        ss = pick.get("sub_scores", {})
        ki = pick.get("key_indicators", {})
        confidence = "LOW" if pick["final_score"] < 50 else "MEDIUM" if pick["final_score"] < 65 else "HIGH"
        print(f"\n  #{i}. {pick['ticker']} - ${pick['price']:.2f}  "
              f"(Score: {pick['final_score']:.1f}, Confidence: {confidence})")
        if pick.get("company"):
            print(f"      {pick['company']}")
        print(f"      Setup:{ss.get('setup',0):.0f} | Tech:{ss.get('technical',0):.0f} | "
              f"PrePump:{ss.get('pre_pump',0):.0f} | Fund:{ss.get('fundamental',0):.0f} | "
              f"Cat:{ss.get('catalyst',0):.0f}")
        print(f"      Pre-Pump: confluence={ki.get('pre_pump_confluence', 0)}/7 "
              f"confidence={ki.get('pre_pump_confidence', 'N/A')}")
        if pick.get("penalty_deduction", 0) > 0:
            print(f"      Penalties: -{pick['penalty_deduction']}pts")
    print("=" * 70)


def cmd_analyze(args):
    from pennystock.analysis.deep_dive import run_deep_dive
    if not args.ticker:
        print("\nUsage: python main.py analyze TICKER")
        print("Example: python main.py analyze GETY")
        return
    run_deep_dive(args.ticker)


def cmd_history(args):
    from pennystock.storage.db import Database
    db = Database()
    picks = db.get_recent_picks(args.n)
    if not picks:
        print("\nNo picks in history yet.")
        return
    print(f"\n  Last {len(picks)} picks:")
    print(f"  {'Date':<20} {'Ticker':<8} {'Score':>6} {'Price':>8}")
    print("  " + "-" * 46)
    for pick in picks:
        print(f"  {pick['timestamp'][:19]:<20} {pick['ticker']:<8} "
              f"{pick['final_score']:>6.1f} ${pick['price']:>7.2f}")


def main():
    parser = argparse.ArgumentParser(description="Penny Stock Analyzer")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of GUI")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("build", help="Build algorithm from recent winners vs losers")

    pick_p = subparsers.add_parser("pick", help="Pick top penny stocks")
    pick_p.add_argument("-n", "--top-n", type=int, default=5)

    analyze_p = subparsers.add_parser("analyze", help="Deep dive analysis on a single stock")
    analyze_p.add_argument("ticker", type=str, nargs="?", help="Stock ticker to analyze (e.g. GETY)")

    hist_p = subparsers.add_parser("history", help="View past picks")
    hist_p.add_argument("-n", type=int, default=10)

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.cli or args.command:
        # CLI mode
        commands = {"build": cmd_build, "pick": cmd_pick, "analyze": cmd_analyze, "history": cmd_history}
        func = commands.get(args.command)
        if func:
            func(args)
        else:
            parser.print_help()
    else:
        # GUI mode (default)
        try:
            from pennystock.gui import launch_gui
            launch_gui()
        except ImportError as e:
            print(f"PyQt6 not installed. Install with: pip install PyQt6")
            print(f"Or use CLI mode: python main.py --cli build")
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
