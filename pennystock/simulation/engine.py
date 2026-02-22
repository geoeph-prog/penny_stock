"""
Paper trading simulation engine with self-learning.

Manages a virtual portfolio:
- Auto-picks stocks using the learned algorithm
- Monitors positions with real-time prices
- Auto-sells based on optimized sell strategy
- Learns from each trade to improve position sizing
- Persists state to simulation_state.json
"""

import json
import os
import time
from datetime import datetime, timedelta

import yfinance as yf
from loguru import logger

from pennystock.config import (
    SELL_MAX_HOLD_DAYS,
    SELL_STOP_LOSS_PCT,
    SELL_TAKE_PROFIT_PCT,
    SELL_TRAILING_STOP_ACTIVATE,
    SELL_TRAILING_STOP_DISTANCE,
    SIMULATION_INITIAL_CAPITAL,
    SIMULATION_MAX_POSITIONS,
    SIMULATION_STATE_FILE,
    SIMULATION_MIN_TRADES_FOR_LEARNING,
)

STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", SIMULATION_STATE_FILE
)
STATE_PATH = os.path.normpath(STATE_PATH)


class SimulationEngine:

    def __init__(self):
        self.state = self._load_state()

    # ─── State persistence ────────────────────────────────────

    def _load_state(self) -> dict:
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load simulation state: {e}")
        return self._initial_state()

    def _save_state(self):
        try:
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save simulation state: {e}")

    def _initial_state(self) -> dict:
        return {
            "cash": SIMULATION_INITIAL_CAPITAL,
            "initial_capital": SIMULATION_INITIAL_CAPITAL,
            "positions": [],
            "trade_history": [],
            "score_performance": {},
            "sell_follow_ups": [],
            "insights": [],
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def reset(self):
        self.state = self._initial_state()
        self._save_state()

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
                    pos["current_price"] = round(float(close_series.iloc[-1]), 4)
                    recent_high = float(high_series.max())
                    if recent_high > pos.get("high_since_entry", 0):
                        pos["high_since_entry"] = round(recent_high, 4)
                    pos["last_updated"] = now
            except Exception as e:
                logger.debug(f"Price update failed for {ticker}: {e}")

        # Also check follow-up stocks
        self._update_follow_ups(progress_callback)

        self.state["last_updated"] = now
        self._save_state()

    def _update_follow_ups(self, progress_callback=None):
        """Check prices of recently sold stocks to evaluate sell decisions."""
        follow_ups = self.state.get("sell_follow_ups", [])
        active = [f for f in follow_ups if not f.get("verdict")]
        if not active:
            return

        for fu in active:
            sell_date = datetime.fromisoformat(fu["sell_date"])
            days_since = (datetime.now() - sell_date).days
            if days_since < 5:
                continue  # Too early to judge

            try:
                ticker = fu["ticker"]
                t = yf.Ticker(ticker)
                hist = t.history(period="10d")
                if hist.empty:
                    continue
                current = float(hist["Close"].iloc[-1])
                fu["current_price"] = round(current, 4)

                if current > fu["sell_price"] * 1.05:
                    fu["verdict"] = "premature_sell"
                    self._add_insight(
                        f"{ticker}: Sold at ${fu['sell_price']:.4f} but rose to "
                        f"${current:.4f} — sold too early ({fu['sell_reason']})"
                    )
                else:
                    fu["verdict"] = "good_sell"
                    self._add_insight(
                        f"{ticker}: Sold at ${fu['sell_price']:.4f}, now at "
                        f"${current:.4f} — good sell ({fu['sell_reason']})"
                    )

                # Learn from the verdict
                self._learn_sell_quality(fu)
            except Exception:
                continue

    # ─── Sell signal detection ────────────────────────────────

    def check_sell_signals(self) -> list:
        """Check each position for sell triggers. Returns [(position, reason)]."""
        sells = []
        today = datetime.now()

        for pos in self.state["positions"]:
            entry_date = datetime.fromisoformat(pos["entry_date"])
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

    # ─── Trade execution ──────────────────────────────────────

    def execute_sells(self, sells, progress_callback=None):
        """Execute sell orders."""
        for pos, reason, description in sells:
            current_price = pos.get("current_price", pos["entry_price"])
            proceeds = current_price * pos["shares"]
            return_pct = ((current_price - pos["entry_price"]) / pos["entry_price"]) * 100
            trade_pnl = (current_price - pos["entry_price"]) * pos["shares"]

            trade = {
                "ticker": pos["ticker"],
                "action": "SELL",
                "shares": pos["shares"],
                "price": round(current_price, 4),
                "entry_price": pos["entry_price"],
                "date": datetime.now().isoformat(),
                "return_pct": round(return_pct, 2),
                "trade_pnl": round(trade_pnl, 2),
                "sell_reason": reason,
                "description": description,
                "entry_score": pos.get("entry_score", 0),
                "entry_confidence": pos.get("entry_confidence", ""),
                "pre_pump_confidence": pos.get("pre_pump_confidence", ""),
                "sub_scores": pos.get("sub_scores", {}),
            }

            self.state["cash"] += proceeds
            self.state["trade_history"].append(trade)

            # Add to follow-ups for post-sell analysis
            self.state.setdefault("sell_follow_ups", []).append({
                "ticker": pos["ticker"],
                "sell_date": datetime.now().isoformat(),
                "sell_price": round(current_price, 4),
                "entry_price": pos["entry_price"],
                "return_pct": round(return_pct, 2),
                "sell_reason": reason,
                "verdict": None,
                "current_price": None,
            })

            # Learn from this trade
            self._learn_from_trade(trade)

            if progress_callback:
                marker = "W" if return_pct > 0 else "L"
                progress_callback(
                    f"  SELL {pos['ticker']} x{pos['shares']} @ ${current_price:.4f} "
                    f"({return_pct:+.1f}% {marker}) — {description}"
                )

        # Remove sold positions
        sold_tickers = {pos["ticker"] for pos, _, _ in sells}
        self.state["positions"] = [
            p for p in self.state["positions"]
            if p["ticker"] not in sold_tickers
        ]
        self._save_state()

    def find_and_buy(self, progress_callback=None):
        """Find new picks and buy to fill empty position slots."""
        open_slots = SIMULATION_MAX_POSITIONS - len(self.state["positions"])
        if open_slots <= 0:
            if progress_callback:
                progress_callback("  All position slots filled.")
            return

        if progress_callback:
            progress_callback(f"  Finding {open_slots} new picks...")

        from pennystock.algorithm import pick_stocks
        picks = pick_stocks(
            top_n=open_slots + 5,  # Get extras in case some are already held
            progress_callback=progress_callback,
        )
        if not picks:
            if progress_callback:
                progress_callback("  No picks found.")
            return

        # Filter out stocks we already hold
        held = {p["ticker"] for p in self.state["positions"]}
        available = [p for p in picks if p["ticker"] not in held]

        bought = 0
        for pick in available[:open_slots]:
            if self.state["cash"] < 10:
                break

            price = pick["price"]
            if price <= 0:
                continue

            shares = self._calculate_shares(pick, self.state["cash"] / max(1, open_slots - bought))
            if shares <= 0:
                continue

            cost = price * shares
            if cost > self.state["cash"]:
                shares = int(self.state["cash"] / price)
                cost = price * shares
            if shares <= 0:
                continue

            # Execute buy
            self.state["cash"] -= cost
            ss = pick.get("sub_scores", {})
            ki = pick.get("key_indicators", {})

            position = {
                "ticker": pick["ticker"],
                "company": pick.get("company", ""),
                "shares": shares,
                "entry_price": round(price, 4),
                "entry_date": datetime.now().isoformat(),
                "entry_score": round(pick["final_score"], 1),
                "entry_confidence": pick.get("confidence", ""),
                "pre_pump_confidence": ki.get("pre_pump_confidence", ""),
                "pre_pump_confluence": ki.get("pre_pump_confluence", 0),
                "sub_scores": {k: round(v, 1) for k, v in ss.items()},
                "current_price": round(price, 4),
                "high_since_entry": round(price, 4),
                "trailing_stop_active": False,
                "last_updated": datetime.now().isoformat(),
            }
            self.state["positions"].append(position)

            trade = {
                "ticker": pick["ticker"],
                "action": "BUY",
                "shares": shares,
                "price": round(price, 4),
                "date": datetime.now().isoformat(),
                "cost": round(cost, 2),
                "entry_score": round(pick["final_score"], 1),
                "entry_confidence": pick.get("confidence", ""),
                "description": f"Pick #{bought+1}, Score {pick['final_score']:.1f}",
            }
            self.state["trade_history"].append(trade)

            if progress_callback:
                progress_callback(
                    f"  BUY {pick['ticker']} x{shares} @ ${price:.4f} "
                    f"(${cost:.2f}) — Score {pick['final_score']:.1f}"
                )
            bought += 1

        self._save_state()

    # ─── Full auto-trade cycle ────────────────────────────────

    def run_auto_cycle(self, progress_callback=None):
        """Full cycle: refresh -> check sells -> execute sells -> buy new -> save."""
        if progress_callback:
            progress_callback("=== Auto-Trade Cycle ===")
            progress_callback(f"  Portfolio: ${self.get_portfolio_value():.2f} | "
                              f"Cash: ${self.state['cash']:.2f}")

        # 1. Refresh prices
        self.refresh_prices(progress_callback)

        # 2. Check sell signals
        sells = self.check_sell_signals()
        if sells:
            if progress_callback:
                progress_callback(f"\n  {len(sells)} sell signal(s) triggered:")
            self.execute_sells(sells, progress_callback)
        else:
            if progress_callback:
                progress_callback("  No sell signals triggered.")

        # 3. Buy new picks if slots available
        open_slots = SIMULATION_MAX_POSITIONS - len(self.state["positions"])
        if open_slots > 0:
            if progress_callback:
                progress_callback(f"\n  {open_slots} empty slot(s), finding picks...")
            self.find_and_buy(progress_callback)

        # 4. Summary
        if progress_callback:
            summary = self.get_portfolio_summary()
            progress_callback(f"\n=== Cycle Complete ===")
            progress_callback(f"  Value: ${summary['total_value']:.2f} "
                              f"({summary['total_return_pct']:+.1f}%)")
            progress_callback(f"  Positions: {summary['num_positions']}/{SIMULATION_MAX_POSITIONS}")
            progress_callback(f"  Cash: ${summary['cash']:.2f}")
            if summary['total_trades'] > 0:
                progress_callback(f"  Trades: {summary['total_trades']} "
                                  f"({summary['win_rate']:.0f}% win rate)")

        self._save_state()

    # ─── Self-learning ────────────────────────────────────────

    def _learn_from_trade(self, trade):
        """Update performance tracking after a sell."""
        perf = self.state.setdefault("score_performance", {})
        return_pct = trade.get("return_pct", 0)
        is_win = return_pct > 0

        # Track by score range
        score = trade.get("entry_score", 0)
        bucket = f"score_{int(score // 10) * 10}_{int(score // 10) * 10 + 10}"
        if bucket not in perf:
            perf[bucket] = {"trades": 0, "wins": 0, "total_return": 0}
        perf[bucket]["trades"] += 1
        if is_win:
            perf[bucket]["wins"] += 1
        perf[bucket]["total_return"] = round(perf[bucket]["total_return"] + return_pct, 2)

        # Track by pre_pump confidence
        pp_conf = trade.get("pre_pump_confidence", "LOW")
        pp_key = f"pp_{pp_conf}"
        if pp_key not in perf:
            perf[pp_key] = {"trades": 0, "wins": 0, "total_return": 0}
        perf[pp_key]["trades"] += 1
        if is_win:
            perf[pp_key]["wins"] += 1
        perf[pp_key]["total_return"] = round(perf[pp_key]["total_return"] + return_pct, 2)

        # Track by sell reason
        reason = trade.get("sell_reason", "unknown")
        rk = f"reason_{reason}"
        if rk not in perf:
            perf[rk] = {"trades": 0, "wins": 0, "total_return": 0}
        perf[rk]["trades"] += 1
        if is_win:
            perf[rk]["wins"] += 1
        perf[rk]["total_return"] = round(perf[rk]["total_return"] + return_pct, 2)

        # Track by sub-score dominance
        ss = trade.get("sub_scores", {})
        if ss:
            dominant = max(ss, key=ss.get)
            dk = f"dominant_{dominant}"
            if dk not in perf:
                perf[dk] = {"trades": 0, "wins": 0, "total_return": 0}
            perf[dk]["trades"] += 1
            if is_win:
                perf[dk]["wins"] += 1
            perf[dk]["total_return"] = round(perf[dk]["total_return"] + return_pct, 2)

        # Generate insight
        if is_win:
            self._add_insight(
                f"WIN: {trade['ticker']} {return_pct:+.1f}% via {reason} "
                f"(score {score:.0f}, pp={pp_conf})"
            )
        else:
            self._add_insight(
                f"LOSS: {trade['ticker']} {return_pct:+.1f}% via {reason} "
                f"(score {score:.0f}, pp={pp_conf})"
            )

    def _learn_sell_quality(self, follow_up):
        """Learn whether a sell decision was good or premature."""
        perf = self.state.setdefault("score_performance", {})
        reason = follow_up.get("sell_reason", "unknown")
        verdict = follow_up.get("verdict", "unknown")

        vk = f"verdict_{reason}_{verdict}"
        if vk not in perf:
            perf[vk] = {"count": 0}
        perf[vk]["count"] += 1

    def _calculate_shares(self, pick, cash_per_position):
        """Calculate shares, adjusted by learned performance data."""
        price = pick["price"]
        if price <= 0:
            return 0

        base_shares = int(cash_per_position / price)
        perf = self.state.get("score_performance", {})
        total_sells = sum(
            v["trades"] for k, v in perf.items()
            if k.startswith("score_") and "trades" in v
        )

        if total_sells < SIMULATION_MIN_TRADES_FOR_LEARNING:
            return base_shares

        multiplier = 1.0

        # Adjust by score bucket performance
        score = pick.get("final_score", 50)
        bucket = f"score_{int(score // 10) * 10}_{int(score // 10) * 10 + 10}"
        bucket_data = perf.get(bucket)
        if bucket_data and bucket_data["trades"] >= 3:
            wr = bucket_data["wins"] / bucket_data["trades"]
            avg_r = bucket_data["total_return"] / bucket_data["trades"]
            if wr > 0.6 and avg_r > 2:
                multiplier = 1.3
            elif wr < 0.3 or avg_r < -5:
                multiplier = 0.7

        # Adjust by pre_pump confidence
        ki = pick.get("key_indicators", {})
        pp_conf = ki.get("pre_pump_confidence", "LOW")
        pp_data = perf.get(f"pp_{pp_conf}")
        if pp_data and pp_data["trades"] >= 3:
            pp_wr = pp_data["wins"] / pp_data["trades"]
            if pp_conf == "HIGH" and pp_wr > 0.6:
                multiplier *= 1.2
            elif pp_wr < 0.3:
                multiplier *= 0.8

        multiplier = max(0.5, min(1.5, multiplier))
        return max(1, int(base_shares * multiplier))

    def _add_insight(self, text):
        """Add a timestamped insight."""
        insights = self.state.setdefault("insights", [])
        insights.append({
            "date": datetime.now().isoformat(),
            "text": text,
        })
        # Keep last 100 insights
        if len(insights) > 100:
            self.state["insights"] = insights[-100:]

    # ─── Portfolio summary ────────────────────────────────────

    def get_portfolio_value(self) -> float:
        invested = sum(
            p.get("current_price", p["entry_price"]) * p["shares"]
            for p in self.state["positions"]
        )
        return self.state["cash"] + invested

    def get_portfolio_summary(self) -> dict:
        positions = self.state["positions"]
        cash = self.state["cash"]
        initial = self.state["initial_capital"]

        invested = sum(
            p.get("current_price", p["entry_price"]) * p["shares"]
            for p in positions
        )
        total_value = cash + invested
        total_return = total_value - initial
        total_return_pct = (total_return / initial) * 100 if initial > 0 else 0

        # Compute from sell trades
        sell_trades = [t for t in self.state["trade_history"] if t["action"] == "SELL"]
        wins = sum(1 for t in sell_trades if t.get("return_pct", 0) > 0)
        total_trades = len(sell_trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t.get("trade_pnl", 0) for t in sell_trades)
        avg_return = (sum(t.get("return_pct", 0) for t in sell_trades) / total_trades
                      if total_trades > 0 else 0)

        # Unrealized P&L
        unrealized = sum(
            (p.get("current_price", p["entry_price"]) - p["entry_price"]) * p["shares"]
            for p in positions
        )

        return {
            "total_value": round(total_value, 2),
            "cash": round(cash, 2),
            "invested": round(invested, 2),
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct, 2),
            "unrealized_pnl": round(unrealized, 2),
            "realized_pnl": round(total_pnl, 2),
            "num_positions": len(positions),
            "total_trades": total_trades,
            "wins": wins,
            "win_rate": round(win_rate, 1),
            "avg_return": round(avg_return, 2),
            "initial_capital": initial,
        }

    def get_learning_summary(self) -> dict:
        """Return performance data for display."""
        perf = self.state.get("score_performance", {})
        result = {}

        for key, data in perf.items():
            if "trades" in data and data["trades"] > 0:
                result[key] = {
                    "trades": data["trades"],
                    "wins": data.get("wins", 0),
                    "win_rate": round(data.get("wins", 0) / data["trades"] * 100, 1),
                    "avg_return": round(data["total_return"] / data["trades"], 2),
                }
            elif "count" in data:
                result[key] = {"count": data["count"]}

        return result
