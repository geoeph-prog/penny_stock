"""
Real-time price monitor and alert scheduler.

Runs during market hours:
  - Polls prices every N minutes for held positions
  - Checks sell triggers (SL, TP, trailing stop, max hold)
  - Runs the pick algorithm on a configurable schedule
  - Sends email alerts for buy/sell signals
  - Sends daily portfolio summary at market close

Can run as:
  1. CLI foreground process: python main.py alerts start
  2. Background thread from the GUI Alerts tab
"""

import json
import os
import time
import threading
from datetime import datetime, timedelta

import yfinance as yf
from loguru import logger

from pennystock.config import (
    ALERT_ENABLED,
    ALERT_PRICE_CHECK_MINUTES,
    ALERT_SCAN_HOURS,
    ALERT_DAILY_SUMMARY,
    ALERT_ON_BUY,
    ALERT_ON_SELL,
    SELL_MAX_HOLD_DAYS,
    SELL_STOP_LOSS_PCT,
    SELL_TAKE_PROFIT_PCT,
    SELL_TRAILING_STOP_ACTIVATE,
    SELL_TRAILING_STOP_DISTANCE,
)


def _is_market_hours() -> bool:
    """Check if US stock market is currently open (9:30 AM - 4:00 PM ET, Mon-Fri)."""
    now = datetime.now()
    # Simple weekday check (0=Mon, 6=Sun)
    if now.weekday() >= 5:
        return False
    # Approximate ET check (adjust if running in different timezone)
    # We use a generous window to account for timezone differences
    hour = now.hour
    return 6 <= hour <= 20  # Wide window — user can tighten via config


def _market_close_today() -> bool:
    """Return True if we're near market close (3:50-4:10 PM ET window)."""
    now = datetime.now()
    return now.weekday() < 5 and 15 <= now.hour <= 16


class AlertMonitor:
    """Monitors positions and sends email alerts."""

    STATE_FILE = os.path.join(
        os.path.dirname(__file__), "..", "..", "alert_state.json"
    )
    STATE_FILE = os.path.normpath(STATE_FILE)

    def __init__(self):
        self._running = False
        self._thread = None
        self._log_callback = None
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "last_scan": None,
            "last_price_check": None,
            "last_daily_summary": None,
            "alerts_sent": 0,
            "buy_alerts": 0,
            "sell_alerts": 0,
            "summary_alerts": 0,
            "history": [],
        }

    def _save_state(self):
        try:
            with open(self.STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alert state: {e}")

    def _log(self, msg: str):
        logger.info(msg)
        if self._log_callback:
            try:
                self._log_callback(msg)
            except Exception:
                pass

    def _add_history(self, alert_type: str, detail: str):
        self.state.setdefault("history", []).append({
            "time": datetime.now().isoformat(),
            "type": alert_type,
            "detail": detail,
        })
        # Keep last 200 entries
        if len(self.state["history"]) > 200:
            self.state["history"] = self.state["history"][-200:]

    # ─── Price check + sell triggers ────────────────────────

    def check_positions(self) -> list:
        """Fetch live prices for held positions and check sell triggers.

        Returns list of (position, reason, description) tuples.
        """
        from pennystock.simulation.engine import SimulationEngine

        engine = SimulationEngine()
        positions = engine.state.get("positions", [])
        if not positions:
            return []

        # Refresh prices
        tickers = [p["ticker"] for p in positions]
        self._log(f"Checking prices for {len(tickers)} positions: {', '.join(tickers)}")

        try:
            data = yf.download(tickers, period="5d", progress=False, threads=True)
        except Exception as e:
            self._log(f"Price fetch failed: {e}")
            return []

        now = datetime.now()
        for pos in positions:
            ticker = pos["ticker"]
            try:
                if len(tickers) == 1:
                    close_s = data["Close"]
                    high_s = data.get("High", data["Close"])
                else:
                    close_s = data["Close"][ticker]
                    high_s = data.get("High", data["Close"])
                    if hasattr(high_s, "columns"):
                        high_s = high_s[ticker]

                close_s = close_s.dropna()
                high_s = high_s.dropna()

                if not close_s.empty:
                    pos["current_price"] = round(float(close_s.iloc[-1]), 4)
                    recent_high = float(high_s.max())
                    if recent_high > pos.get("high_since_entry", 0):
                        pos["high_since_entry"] = round(recent_high, 4)
                    pos["last_updated"] = now.isoformat()
            except Exception as e:
                logger.debug(f"Price update failed for {ticker}: {e}")

        # Check sell triggers
        sells = []
        for pos in positions:
            entry_date = datetime.fromisoformat(pos["entry_date"])
            days_held = (now - entry_date).days
            entry_price = pos["entry_price"]
            current_price = pos.get("current_price", entry_price)
            high_since = pos.get("high_since_entry", entry_price)
            return_pct = ((current_price - entry_price) / entry_price) * 100

            if days_held >= SELL_MAX_HOLD_DAYS:
                sells.append((pos, "max_hold", f"Held {days_held}d (max {SELL_MAX_HOLD_DAYS}d)"))
                continue
            if SELL_STOP_LOSS_PCT != 0 and return_pct <= SELL_STOP_LOSS_PCT:
                sells.append((pos, "stop_loss", f"Hit stop-loss {SELL_STOP_LOSS_PCT}%"))
                continue
            if SELL_TAKE_PROFIT_PCT != 0 and return_pct >= SELL_TAKE_PROFIT_PCT:
                sells.append((pos, "take_profit", f"Hit take-profit +{SELL_TAKE_PROFIT_PCT}%"))
                continue
            if SELL_TRAILING_STOP_ACTIVATE > 0 and SELL_TRAILING_STOP_DISTANCE > 0:
                peak_gain = ((high_since - entry_price) / entry_price) * 100
                if peak_gain >= SELL_TRAILING_STOP_ACTIVATE:
                    pos["trailing_stop_active"] = True
                    trail_price = high_since * (1 - SELL_TRAILING_STOP_DISTANCE / 100)
                    if current_price <= trail_price:
                        sells.append((
                            pos, "trailing_stop",
                            f"Trailing stop: peaked +{peak_gain:.1f}%, dropped to ${current_price:.4f}",
                        ))

        # Save updated prices back
        engine.state["positions"] = positions
        engine.state["last_updated"] = now.isoformat()
        engine._save_state()

        self.state["last_price_check"] = now.isoformat()
        self._save_state()

        return sells

    # ─── Scan for new picks ─────────────────────────────────

    def scan_for_picks(self) -> list:
        """Run the pick algorithm and return results."""
        self._log("Running stock scan...")
        try:
            from pennystock.algorithm import pick_stocks
            picks = pick_stocks(top_n=10)
            self.state["last_scan"] = datetime.now().isoformat()
            self._save_state()
            if picks:
                self._log(f"Scan found {len(picks)} picks")
            else:
                self._log("Scan found no picks above threshold")
            return picks or []
        except Exception as e:
            self._log(f"Scan failed: {e}")
            return []

    # ─── Alert dispatch ─────────────────────────────────────

    def send_sell_alerts(self, sells: list):
        """Send email for sell triggers."""
        if not sells or not ALERT_ON_SELL:
            return
        from pennystock.alerts.email_sender import send_sell_alert
        if send_sell_alert(sells):
            self.state["alerts_sent"] += 1
            self.state["sell_alerts"] += 1
            tickers = [pos["ticker"] for pos, _, _ in sells]
            self._add_history("SELL", f"{', '.join(tickers)}")
            self._save_state()
            self._log(f"SELL alert sent for {', '.join(tickers)}")

    def send_buy_alerts(self, picks: list):
        """Send email for new buy signals."""
        if not picks or not ALERT_ON_BUY:
            return
        from pennystock.alerts.email_sender import send_buy_alert
        if send_buy_alert(picks):
            self.state["alerts_sent"] += 1
            self.state["buy_alerts"] += 1
            tickers = [p["ticker"] for p in picks]
            self._add_history("BUY", f"{', '.join(tickers)}")
            self._save_state()
            self._log(f"BUY alert sent for {', '.join(tickers)}")

    def send_daily_summary(self):
        """Send end-of-day portfolio summary."""
        if not ALERT_DAILY_SUMMARY:
            return
        from pennystock.simulation.engine import SimulationEngine
        from pennystock.alerts.email_sender import send_portfolio_summary
        engine = SimulationEngine()
        summary = engine.get_portfolio_summary()
        positions = engine.state.get("positions", [])
        if send_portfolio_summary(summary, positions):
            self.state["alerts_sent"] += 1
            self.state["summary_alerts"] += 1
            self.state["last_daily_summary"] = datetime.now().isoformat()
            self._add_history("SUMMARY", f"${summary['total_value']:,.2f}")
            self._save_state()
            self._log(f"Daily summary sent: ${summary['total_value']:,.2f}")

    # ─── Main loop ──────────────────────────────────────────

    def run_once(self):
        """Single monitoring cycle: check prices, check sells, optionally scan."""
        now = datetime.now()

        # 1. Check positions + sell triggers
        sells = self.check_positions()
        if sells:
            self._log(f"{len(sells)} sell trigger(s) detected!")
            for pos, reason, desc in sells:
                self._log(f"  SELL {pos['ticker']}: {desc}")
            self.send_sell_alerts(sells)

        # 2. Scan for new picks (on schedule)
        last_scan = self.state.get("last_scan")
        hours_since_scan = 999
        if last_scan:
            try:
                last_dt = datetime.fromisoformat(last_scan)
                hours_since_scan = (now - last_dt).total_seconds() / 3600
            except Exception:
                pass

        if hours_since_scan >= ALERT_SCAN_HOURS:
            picks = self.scan_for_picks()
            if picks:
                self.send_buy_alerts(picks)

        # 3. Daily summary near market close
        if _market_close_today():
            last_summary = self.state.get("last_daily_summary")
            sent_today = False
            if last_summary:
                try:
                    last_dt = datetime.fromisoformat(last_summary)
                    sent_today = last_dt.date() == now.date()
                except Exception:
                    pass
            if not sent_today:
                self.send_daily_summary()

    def start(self, log_callback=None):
        """Start the monitoring loop in a background thread."""
        if self._running:
            self._log("Monitor already running")
            return

        self._log_callback = log_callback
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._log(f"Alert monitor started (check every {ALERT_PRICE_CHECK_MINUTES}min, "
                  f"scan every {ALERT_SCAN_HOURS}h)")

    def stop(self):
        """Stop the monitoring loop."""
        self._running = False
        self._log("Alert monitor stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def _loop(self):
        """Background loop — runs until stopped."""
        while self._running:
            try:
                if _is_market_hours():
                    self.run_once()
                else:
                    self._log("Market closed — sleeping...")
            except Exception as e:
                self._log(f"Monitor error: {e}")
                logger.exception("Monitor loop error")

            # Sleep in small increments so stop() is responsive
            sleep_secs = ALERT_PRICE_CHECK_MINUTES * 60
            for _ in range(int(sleep_secs)):
                if not self._running:
                    break
                time.sleep(1)

    def get_status(self) -> dict:
        """Return current monitor status for display."""
        return {
            "running": self._running,
            "last_scan": self.state.get("last_scan"),
            "last_price_check": self.state.get("last_price_check"),
            "last_daily_summary": self.state.get("last_daily_summary"),
            "alerts_sent": self.state.get("alerts_sent", 0),
            "buy_alerts": self.state.get("buy_alerts", 0),
            "sell_alerts": self.state.get("sell_alerts", 0),
            "summary_alerts": self.state.get("summary_alerts", 0),
            "history": self.state.get("history", [])[-20:],
        }
