#!/usr/bin/env python3
"""
Stock Analyzer ($0.50-$5.00)

Usage:
    python main.py              # Launch GUI (default)
    python main.py --cli build  # Build algorithm from CLI
    python main.py --cli pick   # Pick stocks from CLI
    python main.py analyze GETY   # Deep dive analysis on a single stock
    python main.py backtest 2025-08-01  # Backtest algorithm on a past date
    python main.py optimize     # Run full 3-year backtest optimization
    python main.py simulate       # Paper trading simulation status
    python main.py simulate trade # Run auto-trade cycle
    python main.py --cli history  # View past picks
"""

import argparse
import sys
import time

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
    print(f"  TOP {len(picks)} STOCK PICKS (v{ALGORITHM_VERSION} MEGA-ALGORITHM)")
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


def cmd_backtest(args):
    from pennystock.backtest.historical import run_historical_backtest
    if not args.target_date:
        print("\nUsage: python main.py backtest 2025-08-01")
        return
    run_historical_backtest(args.target_date, top_n=args.top_n)


def cmd_optimize(args):
    from pennystock.backtest.optimizer import run_algorithm_optimization, apply_optimized_config
    result = run_algorithm_optimization()
    if not result or "error" in result:
        print("\nOptimization failed. Check log for details.")
        return

    recs = result.get("recommendations", {})
    if not recs:
        print("\nNo recommendations generated.")
        return

    # Ask user whether to apply
    print("\nApply optimized config? (y/n): ", end="")
    try:
        answer = input().strip().lower()
    except EOFError:
        answer = "n"

    if answer == "y":
        path = apply_optimized_config(recs)
        print(f"\nOptimized config saved to {path}")
        print("The algorithm will use these settings on next run.")
    else:
        print("\nNot applied. You can apply later from the GUI (Tab 5).")


def cmd_simulate(args):
    from pennystock.simulation.engine import SimulationEngine
    engine = SimulationEngine()

    if args.sim_action == "status":
        summary = engine.get_portfolio_summary()
        print(f"\n  Portfolio Value: ${summary['total_value']:,.2f} ({summary['total_return_pct']:+.1f}%)")
        print(f"  Cash: ${summary['cash']:,.2f}")
        print(f"  Positions: {summary['num_positions']}/{5}")
        print(f"  Trades: {summary['total_trades']} ({summary['win_rate']:.0f}% win rate)")
        print(f"  Realized P&L: ${summary['realized_pnl']:+,.2f}")
        print(f"  Unrealized P&L: ${summary['unrealized_pnl']:+,.2f}")
        for pos in engine.state.get("positions", []):
            entry = pos["entry_price"]
            current = pos.get("current_price", entry)
            pnl_pct = ((current - entry) / entry) * 100 if entry > 0 else 0
            print(f"    {pos['ticker']}: {pos['shares']} shares @ ${entry:.4f} "
                  f"-> ${current:.4f} ({pnl_pct:+.1f}%)")

    elif args.sim_action == "trade":
        engine.run_auto_cycle(progress_callback=print)

    elif args.sim_action == "refresh":
        engine.refresh_prices(progress_callback=print)
        engine._save_state()
        print("Prices refreshed.")

    elif args.sim_action == "reset":
        engine.reset()
        print("Simulation reset to $5,000.")

    elif args.sim_action == "learn":
        learning = engine.get_learning_summary()
        if not learning:
            print("\nNo learning data yet. Run some trades first.")
            return
        print("\n  === Learning Summary ===")
        for key in sorted(learning.keys()):
            d = learning[key]
            if "trades" in d:
                print(f"  {key}: {d['win_rate']:.0f}%W, avg {d['avg_return']:+.1f}%, n={d['trades']}")
            elif "count" in d:
                print(f"  {key}: count={d['count']}")
    else:
        print("Usage: python main.py simulate {status|trade|refresh|reset|learn}")


def cmd_alerts(args):
    from pennystock.alerts.monitor import AlertMonitor

    if args.alert_action == "status":
        monitor = AlertMonitor()
        status = monitor.get_status()
        print(f"\n  Alert Monitor: {'RUNNING' if status['running'] else 'STOPPED'}")
        print(f"  Alerts Sent: {status['alerts_sent']} "
              f"(buy:{status['buy_alerts']} sell:{status['sell_alerts']} summary:{status['summary_alerts']})")
        if status['last_scan']:
            print(f"  Last Scan: {status['last_scan'][:19]}")
        if status['last_price_check']:
            print(f"  Last Price Check: {status['last_price_check'][:19]}")
        if status['last_daily_summary']:
            print(f"  Last Summary: {status['last_daily_summary'][:19]}")
        history = status.get("history", [])
        if history:
            print(f"\n  Recent alerts:")
            for h in history[-10:]:
                print(f"    [{h['time'][:16]}] {h['type']}: {h['detail']}")

    elif args.alert_action == "start":
        from pennystock.config import ALERT_ENABLED
        if not ALERT_ENABLED:
            print("\nAlerts not enabled. Set ALERT_ENABLED=True in config.py")
            print("and configure ALERT_EMAIL_SENDER, ALERT_EMAIL_PASSWORD, ALERT_EMAIL_RECIPIENT")
            return
        monitor = AlertMonitor()
        print("Starting alert monitor (Ctrl+C to stop)...")
        print(f"  Price check: every {monitor.state.get('price_interval', 15)} min")
        monitor.start(log_callback=print)
        try:
            while monitor.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop()
            print("\nMonitor stopped.")

    elif args.alert_action == "check":
        monitor = AlertMonitor()
        monitor._log_callback = print
        monitor.run_once()

    elif args.alert_action == "test":
        from pennystock.alerts.email_sender import send_test_email
        from pennystock.config import ALERT_EMAIL_SENDER, ALERT_EMAIL_RECIPIENT
        if not ALERT_EMAIL_SENDER or not ALERT_EMAIL_RECIPIENT:
            print("\nEmail not configured. Set these in config.py:")
            print("  ALERT_EMAIL_SENDER = 'you@gmail.com'")
            print("  ALERT_EMAIL_PASSWORD = 'your-app-password'")
            print("  ALERT_EMAIL_RECIPIENT = 'you@gmail.com'")
            return
        print(f"Sending test email to {ALERT_EMAIL_RECIPIENT}...")
        if send_test_email():
            print("Test email sent successfully!")
        else:
            print("Failed to send test email. Check config and logs.")

    else:
        print("Usage: python main.py alerts {status|start|check|test}")


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
    parser = argparse.ArgumentParser(description="Stock Analyzer ($0.50-$5.00)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of GUI")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("build", help="Build algorithm from recent winners vs losers")

    pick_p = subparsers.add_parser("pick", help="Pick top stocks ($0.50-$5.00)")
    pick_p.add_argument("-n", "--top-n", type=int, default=5)

    analyze_p = subparsers.add_parser("analyze", help="Deep dive analysis on a single stock")
    analyze_p.add_argument("ticker", type=str, nargs="?", help="Stock ticker to analyze (e.g. GETY)")

    bt_p = subparsers.add_parser("backtest", help="Backtest algorithm on a historical date")
    bt_p.add_argument("target_date", type=str, nargs="?", help="Date to backtest (e.g. 2025-08-01)")
    bt_p.add_argument("-n", "--top-n", type=int, default=5)

    subparsers.add_parser("optimize", help="Run full 3-year backtest optimization")

    sim_p = subparsers.add_parser("simulate", help="Paper trading simulation")
    sim_p.add_argument("sim_action", type=str, nargs="?", default="status",
                       choices=["status", "trade", "refresh", "reset", "learn"],
                       help="Simulation action (default: status)")

    alert_p = subparsers.add_parser("alerts", help="Email alert monitor")
    alert_p.add_argument("alert_action", type=str, nargs="?", default="status",
                         choices=["status", "start", "check", "test"],
                         help="Alert action (default: status)")

    hist_p = subparsers.add_parser("history", help="View past picks")
    hist_p.add_argument("-n", type=int, default=10)

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.cli or args.command:
        # CLI mode
        commands = {"build": cmd_build, "pick": cmd_pick, "analyze": cmd_analyze,
                    "backtest": cmd_backtest, "optimize": cmd_optimize,
                    "simulate": cmd_simulate, "alerts": cmd_alerts,
                    "history": cmd_history}
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
