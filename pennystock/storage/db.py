"""SQLite database for persisting picks, runs, and backtest results."""

import json
import os
import sqlite3
import time
from datetime import datetime

from loguru import logger


DB_FILE = "pennystock.db"


class Database:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_FILE
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS picks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    final_score REAL,
                    price REAL,
                    sub_scores TEXT,
                    full_result TEXT
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    stats TEXT,
                    run_type TEXT DEFAULT 'daily'
                );

                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    report TEXT,
                    metrics TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_picks_ticker ON picks(ticker);
                CREATE INDEX IF NOT EXISTS idx_picks_timestamp ON picks(timestamp);
            """)

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def save_pick(self, pick: dict):
        """Save a single stock pick."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO picks (timestamp, ticker, final_score, price, sub_scores, full_result) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(),
                    pick.get("ticker", ""),
                    pick.get("final_score", 0),
                    pick.get("price", 0),
                    json.dumps(pick.get("sub_scores", {})),
                    json.dumps(pick, default=str),
                ),
            )

    def save_run(self, stats: dict):
        """Save run statistics."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (timestamp, stats) VALUES (?, ?)",
                (datetime.now().isoformat(), json.dumps(stats)),
            )

    def save_backtest(self, report: str, metrics: dict):
        """Save backtest results."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO backtest_results (timestamp, report, metrics) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), report, json.dumps(metrics, default=str)),
            )

    def get_recent_picks(self, n: int = 10) -> list:
        """Get the N most recent picks."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT timestamp, ticker, final_score, price, sub_scores "
                "FROM picks ORDER BY timestamp DESC LIMIT ?",
                (n,),
            )
            rows = cursor.fetchall()

        return [
            {
                "timestamp": row[0],
                "ticker": row[1],
                "final_score": row[2],
                "price": row[3],
                "sub_scores": json.loads(row[4]) if row[4] else {},
            }
            for row in rows
        ]

    def get_ticker_history(self, ticker: str) -> list:
        """Get all past picks for a specific ticker."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT timestamp, final_score, price, sub_scores "
                "FROM picks WHERE ticker = ? ORDER BY timestamp DESC",
                (ticker,),
            )
            rows = cursor.fetchall()

        return [
            {
                "timestamp": row[0],
                "final_score": row[1],
                "price": row[2],
                "sub_scores": json.loads(row[3]) if row[3] else {},
            }
            for row in rows
        ]

    def get_run_history(self, n: int = 10) -> list:
        """Get recent run stats."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT timestamp, stats FROM runs ORDER BY timestamp DESC LIMIT ?",
                (n,),
            )
            return [
                {"timestamp": row[0], "stats": json.loads(row[1]) if row[1] else {}}
                for row in cursor.fetchall()
            ]
