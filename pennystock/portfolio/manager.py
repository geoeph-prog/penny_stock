"""
Portfolio manager for user's real stock holdings.

Unlike the Simulation (which runs autonomously with virtual money),
the Portfolio tracks stocks the user has actually purchased.
Alerts (buy/sell signals) are tied to this portfolio.

- User manually adds positions (ticker, shares, entry price, date)
- Tracks current prices and P&L
- Generates sell signals (SL, TP, trailing stop, max hold)
- Buy signals come from the stock picker (only HIGH confidence)
- Persists to portfolio_state.json
"""

import json
import os
from datetime import datetime

import yfinance as yf
from loguru import logger

from pennystock.config import (
    SELL_MAX_HOLD_DAYS,
    SELL_STOP_LOSS_PCT,
    SELL_TAKE_PROFIT_PCT,
    SELL_TRAILING_STOP_ACTIVATE,
    SELL_TRAILING_STOP_DISTANCE,
)

STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "portfolio_state.json"
)
STATE_PATH = os.path.normpath(STATE_PATH)


class PortfolioManager:

    def __init__(self):
        self.state = self._load_state()

    # ─── State persistence ────────────────────────────────────

    def _load_state(self) -> dict:
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load portfolio state: {e}")
        return self._initial_state()

    def _save_state(self):
        try:
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save portfolio state: {e}")

    def _initial_state(self) -> dict:
        return {
            "positions": [],
            "trade_history": [],
            "watchlist": [],
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def reset(self):
        self.state = self._initial_state()
        self._save_state()

    # ─── Position management ──────────────────────────────────

    def add_position(self, ticker: str, shares: int, entry_price: float,
                     entry_date: str = "", company: str = "",
                     progress_callback=None) -> bool:
        """Add a real stock position to the portfolio."""
        ticker = ticker.upper().strip()
        if not ticker or shares <= 0 or entry_price <= 0:
            return False

        # Check if already holding this ticker
        for pos in self.state["positions"]:
            if pos["ticker"] == ticker:
                # Average into position
                old_cost = pos["entry_price"] * pos["shares"]
                new_cost = entry_price * shares
                total_shares = pos["shares"] + shares
                pos["shares"] = total_shares
                pos["entry_price"] = round((old_cost + new_cost) / total_shares, 4)
                if progress_callback:
                    progress_callback(
                        f"Averaged into {ticker}: {total_shares} shares @ "
                        f"${pos['entry_price']:.4f}"
                    )
                self._save_state()
                return True

        position = {
            "ticker": ticker,
            "company": company,
            "shares": shares,
            "entry_price": round(entry_price, 4),
            "entry_date": entry_date or datetime.now().isoformat(),
            "current_price": round(entry_price, 4),
            "high_since_entry": round(entry_price, 4),
            "trailing_stop_active": False,
            "last_updated": datetime.now().isoformat(),
        }
        self.state["positions"].append(position)

        self.state["trade_history"].append({
            "ticker": ticker,
            "action": "BUY",
            "shares": shares,
            "price": round(entry_price, 4),
            "date": entry_date or datetime.now().isoformat(),
            "cost": round(entry_price * shares, 2),
        })

        if progress_callback:
            progress_callback(
                f"Added {ticker}: {shares} shares @ ${entry_price:.4f}"
            )

        self._save_state()
        return True

    def remove_position(self, ticker: str, sell_price: float = 0,
                        reason: str = "manual", progress_callback=None) -> bool:
        """Remove a position (user sold it)."""
        ticker = ticker.upper().strip()
        pos = None
        for p in self.state["positions"]:
            if p["ticker"] == ticker:
                pos = p
                break

        if not pos:
            return False

        if sell_price <= 0:
            sell_price = pos.get("current_price", pos["entry_price"])

        return_pct = ((sell_price - pos["entry_price"]) / pos["entry_price"]) * 100
        trade_pnl = (sell_price - pos["entry_price"]) * pos["shares"]

        self.state["trade_history"].append({
            "ticker": ticker,
            "action": "SELL",
            "shares": pos["shares"],
            "price": round(sell_price, 4),
            "entry_price": pos["entry_price"],
            "date": datetime.now().isoformat(),
            "return_pct": round(return_pct, 2),
            "trade_pnl": round(trade_pnl, 2),
            "sell_reason": reason,
        })

        self.state["positions"] = [
            p for p in self.state["positions"] if p["ticker"] != ticker
        ]

        if progress_callback:
            marker = "W" if return_pct > 0 else "L"
            progress_callback(
                f"Sold {ticker} x{pos['shares']} @ ${sell_price:.4f} "
                f"({return_pct:+.1f}% {marker}) — {reason}"
            )

        self._save_state()
        return True

    # ─── Price updates ────────────────────────────────────────

    def refresh_prices(self, progress_callback=None):
        """Fetch current prices for all held positions."""
        positions = self.state["positions"]
        if not positions:
            return

        tickers = [p["ticker"] for p in positions]
        if progress_callback:
            progress_callback(f"Refreshing prices for {len(tickers)} positions...")

        try:
            data = yf.download(tickers, period="5d", progress=False, threads=True)
        except Exception as e:
            logger.error(f"Price refresh failed: {e}")
            return

        now = datetime.now().isoformat()
        for pos in positions:
            ticker = pos["ticker"]
            try:
                if len(tickers) == 1:
                    close_series = data["Close"]
                    high_series = data.get("High", data["Close"])
                else:
                    close_series = data["Close"][ticker]
                    high_series = data.get("High", data["Close"])
                    if hasattr(high_series, "columns"):
                        high_series = high_series[ticker]

                close_series = close_series.dropna()
                high_series = high_series.dropna()

                if not close_series.empty:
                    last_close = close_series.iloc[-1]
                    pos["current_price"] = round(float(last_close.item() if hasattr(last_close, 'item') else last_close), 4)
                    high_max = high_series.max()
                    recent_high = float(high_max.item() if hasattr(high_max, 'item') else high_max)
                    if recent_high > pos.get("high_since_entry", 0):
                        pos["high_since_entry"] = round(recent_high, 4)
                    pos["last_updated"] = now
            except Exception as e:
                logger.debug(f"Price update failed for {ticker}: {e}")

        self.state["last_updated"] = now
        self._save_state()

    # ─── Sell signal detection ────────────────────────────────

    def check_sell_signals(self) -> list:
        """Check each position for sell triggers. Returns [(position, reason, description)]."""
        sells = []
        today = datetime.now()

        for pos in self.state["positions"]:
            try:
                entry_date = datetime.fromisoformat(pos["entry_date"])
            except Exception:
                entry_date = today
            days_held = (today - entry_date).days
            entry_price = pos["entry_price"]
            current_price = pos.get("current_price", entry_price)
            high_since = pos.get("high_since_entry", entry_price)
            return_pct = ((current_price - entry_price) / entry_price) * 100

            # 1. Max hold period
            if days_held >= SELL_MAX_HOLD_DAYS:
                sells.append((pos, "max_hold", f"Held {days_held}d (max {SELL_MAX_HOLD_DAYS}d)"))
                continue

            # 2. Stop-loss
            if SELL_STOP_LOSS_PCT != 0 and return_pct <= SELL_STOP_LOSS_PCT:
                sells.append((pos, "stop_loss", f"Hit stop-loss {SELL_STOP_LOSS_PCT}%"))
                continue

            # 3. Take-profit
            if SELL_TAKE_PROFIT_PCT != 0 and return_pct >= SELL_TAKE_PROFIT_PCT:
                sells.append((pos, "take_profit", f"Hit take-profit +{SELL_TAKE_PROFIT_PCT}%"))
                continue

            # 4. Trailing stop
            if SELL_TRAILING_STOP_ACTIVATE > 0 and SELL_TRAILING_STOP_DISTANCE > 0:
                peak_gain = ((high_since - entry_price) / entry_price) * 100
                if peak_gain >= SELL_TRAILING_STOP_ACTIVATE:
                    pos["trailing_stop_active"] = True
                    trail_price = high_since * (1 - SELL_TRAILING_STOP_DISTANCE / 100)
                    if current_price <= trail_price:
                        sells.append((
                            pos, "trailing_stop",
                            f"Trailing stop: peaked at +{peak_gain:.1f}%, "
                            f"dropped to ${current_price:.4f}"
                        ))
                        continue

        return sells

    # ─── Portfolio summary ────────────────────────────────────

    def get_portfolio_summary(self) -> dict:
        positions = self.state["positions"]

        invested = sum(
            p.get("current_price", p["entry_price"]) * p["shares"]
            for p in positions
        )
        cost_basis = sum(p["entry_price"] * p["shares"] for p in positions)

        sell_trades = [t for t in self.state["trade_history"] if t["action"] == "SELL"]
        wins = sum(1 for t in sell_trades if t.get("return_pct", 0) > 0)
        total_trades = len(sell_trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        realized_pnl = sum(t.get("trade_pnl", 0) for t in sell_trades)
        unrealized_pnl = invested - cost_basis

        return {
            "total_value": round(invested, 2),
            "cost_basis": round(cost_basis, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(realized_pnl, 2),
            "total_pnl": round(realized_pnl + unrealized_pnl, 2),
            "num_positions": len(positions),
            "total_trades": total_trades,
            "wins": wins,
            "win_rate": round(win_rate, 1),
        }
