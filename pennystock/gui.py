"""
PyQt6 GUI for the Stock Analyzer ($2-$5).

Six tabs:
  Tab 1 - Algorithm Builder: Build, Backtest, and Optimize the algorithm (3 sub-tabs)
  Tab 2 - Stock Picker: Apply algorithm to find today's top 5-8 picks
  Tab 3 - Simulation: Autonomous paper trading with self-learning
  Tab 4 - My Portfolio: Track real holdings with buy/sell alerts
  Tab 5 - Analyze Stock: Comprehensive deep dive on a single ticker
  Tab 6 - Alerts: Email notifications for portfolio buy/sell signals
"""

import os
import sys
import threading
from datetime import date, datetime

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QProgressBar, QLabel, QHeaderView, QLineEdit, QScrollArea, QFrame,
    QGridLayout, QSplitter, QSizePolicy, QDateEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QMessageBox,
)
from PyQt6.QtGui import QFont, QColor

from pennystock import __version__
from pennystock.config import ALGORITHM_VERSION
from pennystock.algorithm import build_algorithm, pick_stocks, load_algorithm, ALGORITHM_FILE


# ── Dark Theme ──────────────────────────────────────────────────────
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    background: #1e1e2e;
}
QTabBar::tab {
    background: #313244;
    color: #cdd6f4;
    padding: 10px 25px;
    border: 1px solid #45475a;
    font-size: 13px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background: #45475a;
    color: #f5c2e7;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    padding: 12px 30px;
    font-size: 14px;
    font-weight: bold;
    border-radius: 6px;
}
QPushButton:hover {
    background-color: #74c7ec;
}
QPushButton:disabled {
    background-color: #585b70;
    color: #6c7086;
}
QTextEdit {
    background-color: #11111b;
    color: #a6e3a1;
    border: 1px solid #45475a;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
}
QTableWidget {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #45475a;
    gridline-color: #313244;
    font-size: 12px;
}
QTableWidget::item {
    padding: 6px;
}
QTableWidget::item:selected {
    background-color: #45475a;
}
QHeaderView::section {
    background-color: #313244;
    color: #f5c2e7;
    padding: 8px;
    border: 1px solid #45475a;
    font-weight: bold;
}
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    background-color: #313244;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 3px;
}
QLabel {
    color: #cdd6f4;
    font-size: 13px;
}
QLineEdit {
    background-color: #11111b;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 14px;
    font-weight: bold;
}
QLineEdit:focus {
    border: 2px solid #89b4fa;
}
QSpinBox, QDoubleSpinBox {
    background-color: #11111b;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px;
    font-size: 13px;
}
"""


# ── Worker Thread ───────────────────────────────────────────────────
class Worker(QThread):
    """Runs long tasks in background so GUI doesn't freeze."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, progress_callback=self.progress.emit, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.progress.emit(f"\nERROR: {e}")
            self.finished.emit(None)


# ═══════════════════════════════════════════════════════════════════
# TAB 1: ALGORITHM BUILDER (Build + Backtest + Optimize)
# ═══════════════════════════════════════════════════════════════════

class AlgorithmBuilderTab(QWidget):
    """Combines Build, Backtest, and Optimize into one integrated tab with sub-tabs."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        header = QLabel("Algorithm Builder")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #f5c2e7; padding: 10px;")
        layout.addWidget(header)

        desc = QLabel(
            "Build, test, and optimize your stock-picking algorithm. "
            "The algorithm learns from historical winners vs losers, integrates market & sector sentiment, "
            "and can be optimized with a full 3-year backtest."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 0 10px 5px 10px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Sub-tabs
        self.sub_tabs = QTabWidget()
        self.sub_tabs.addTab(self._build_build_tab(), "  Build  ")
        self.sub_tabs.addTab(self._build_backtest_tab(), "  Backtest  ")
        self.sub_tabs.addTab(self._build_optimize_tab(), "  Optimize  ")
        layout.addWidget(self.sub_tabs)

        self.setLayout(layout)

    # ── Build sub-tab ─────────────────────────────────────────
    def _build_build_tab(self):
        widget = QWidget()
        self._build_worker = None
        layout = QVBoxLayout()

        desc = QLabel(
            "Analyzes all stocks ($2-$5) over the past 3 months.\n"
            "Finds stocks that gained steadily over 2-4+ weeks (not pump & dumps).\n"
            "Compares winners vs losers to build a unified algorithm with kill filters + weighted scoring."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 5px 10px;")
        layout.addWidget(desc)

        btn_row = QHBoxLayout()
        self.build_btn = QPushButton("Build Algorithm")
        self.build_btn.clicked.connect(self._start_build)
        btn_row.addWidget(self.build_btn)
        self.build_status = QLabel("")
        btn_row.addWidget(self.build_status)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.build_progress = QProgressBar()
        self.build_progress.setRange(0, 0)
        self.build_progress.setVisible(False)
        layout.addWidget(self.build_progress)

        self.build_log = QTextEdit()
        self.build_log.setReadOnly(True)
        layout.addWidget(self.build_log)

        widget.setLayout(layout)
        return widget

    def _start_build(self):
        self.build_btn.setEnabled(False)
        self.build_progress.setVisible(True)
        self.build_log.clear()
        self.build_log.append("Starting algorithm build...")
        self._build_worker = Worker(build_algorithm)
        self._build_worker.progress.connect(lambda m: self._append_log(self.build_log, m))
        self._build_worker.finished.connect(self._build_done)
        self._build_worker.start()

    def _build_done(self, result):
        self.build_progress.setVisible(False)
        self.build_btn.setEnabled(True)
        if result:
            n_factors = len(result.get("factors", []))
            n_winners = result.get("training_summary", {}).get("winners", 0)
            # Verify the on-disk file actually matches what we just built —
            # catches cwd/permissions issues that would otherwise look fine.
            saved = load_algorithm()
            disk_factors = len(saved.get("factors", []))
            disk_built = saved.get("built_date")
            on_disk_matches = (
                disk_factors == n_factors and disk_built == result.get("built_date")
            )
            if on_disk_matches:
                self.build_status.setText(f"Done! {n_winners} winners, {n_factors} factors learned.")
                self.build_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
                self.build_log.append(f"\nAlgorithm saved to {ALGORITHM_FILE}")
            else:
                self.build_status.setText(
                    "Built, but on-disk file doesn't match - check permissions/path."
                )
                self.build_status.setStyleSheet("color: #f9e2af; font-weight: bold;")
                self.build_log.append(
                    f"\nWARNING: algorithm.json at {ALGORITHM_FILE} "
                    f"shows {disk_factors} factors (built {disk_built}); "
                    f"expected {n_factors} factors (built {result.get('built_date')})."
                )
        else:
            self.build_status.setText("Failed - check log for details.")
            self.build_status.setStyleSheet("color: #f38ba8; font-weight: bold;")

    # ── Backtest sub-tab ──────────────────────────────────────
    def _build_backtest_tab(self):
        widget = QWidget()
        self._bt_worker = None
        self._bt_result = None
        layout = QVBoxLayout()

        desc = QLabel(
            "Run the algorithm on a past date and see what would have happened.\n"
            "Picks are scored using historical price data; forward returns show actual results."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 5px 10px;")
        layout.addWidget(desc)

        input_row = QHBoxLayout()
        date_label = QLabel("Target Date:")
        date_label.setStyleSheet("font-size: 14px; font-weight: bold; padding-left: 10px;")
        input_row.addWidget(date_label)

        self.bt_date = QDateEdit()
        self.bt_date.setCalendarPopup(True)
        self.bt_date.setDisplayFormat("yyyy-MM-dd")
        self.bt_date.setDate(QDate(2025, 8, 1))
        self.bt_date.setMaximumDate(QDate.currentDate().addDays(-15))
        self.bt_date.setMinimumDate(QDate(2024, 1, 1))
        self.bt_date.setStyleSheet(
            "QDateEdit { background-color: #11111b; color: #cdd6f4; "
            "border: 1px solid #45475a; border-radius: 4px; padding: 8px 12px; "
            "font-size: 14px; font-weight: bold; }"
        )
        self.bt_date.setMaximumWidth(180)
        input_row.addWidget(self.bt_date)

        self.bt_run_btn = QPushButton("Run Backtest")
        self.bt_run_btn.clicked.connect(self._start_backtest)
        input_row.addWidget(self.bt_run_btn)

        self.bt_status = QLabel("")
        input_row.addWidget(self.bt_status)
        input_row.addStretch()
        layout.addLayout(input_row)

        self.bt_progress = QProgressBar()
        self.bt_progress.setRange(0, 0)
        self.bt_progress.setVisible(False)
        layout.addWidget(self.bt_progress)

        self.bt_table = QTableWidget()
        self.bt_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.bt_table.setVisible(False)
        layout.addWidget(self.bt_table)

        self.bt_summary_frame = QFrame()
        self.bt_summary_frame.setStyleSheet(
            "QFrame { background-color: #181825; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 10px; }"
        )
        self.bt_summary_frame.setVisible(False)
        self.bt_summary_layout = QVBoxLayout()
        self.bt_summary_frame.setLayout(self.bt_summary_layout)
        layout.addWidget(self.bt_summary_frame)

        self.bt_log = QTextEdit()
        self.bt_log.setReadOnly(True)
        layout.addWidget(self.bt_log)

        widget.setLayout(layout)
        return widget

    def _start_backtest(self):
        target_date = self.bt_date.date().toString("yyyy-MM-dd")
        self.bt_run_btn.setEnabled(False)
        self.bt_progress.setVisible(True)
        self.bt_table.setVisible(False)
        self.bt_summary_frame.setVisible(False)
        self.bt_log.clear()
        self.bt_status.setText(f"Running backtest for {target_date}...")
        self.bt_status.setStyleSheet("color: #89b4fa;")

        from pennystock.backtest.historical import run_historical_backtest
        self._bt_worker = Worker(run_historical_backtest, target_date)
        self._bt_worker.progress.connect(lambda m: self._append_log(self.bt_log, m))
        self._bt_worker.finished.connect(self._backtest_done)
        self._bt_worker.start()

    def _backtest_done(self, result):
        self.bt_progress.setVisible(False)
        self.bt_run_btn.setEnabled(True)
        self._bt_result = result

        if not result or not result.get("picks"):
            self.bt_status.setText("Backtest failed or no picks found")
            self.bt_status.setStyleSheet("color: #f38ba8; font-weight: bold;")
            return

        picks = result["picks"]
        summary = result.get("summary", {})
        hold_days = result.get("hold_days", [3, 5, 7, 10, 14])
        hold_keys = [f"{d}d" for d in hold_days]

        best_wr_key = max(hold_keys, key=lambda k: summary.get(f"top_picks_{k}", {}).get("win_rate", 0))
        best_wr = summary.get(f"top_picks_{best_wr_key}", {}).get("win_rate", 0)
        self.bt_status.setText(
            f"{result['target_date']}: {len(picks)} picks | Best: {best_wr:.0f}% win @ {best_wr_key}"
        )
        color = "#a6e3a1" if best_wr > 50 else "#f9e2af" if best_wr >= 40 else "#f38ba8"
        self.bt_status.setStyleSheet(f"color: {color}; font-weight: bold;")

        base_headers = ["Rank", "Ticker", "Entry $", "Score", "Setup", "Tech", "PrePump"]
        ret_headers = [f"{k} Ret" for k in hold_keys]
        all_headers = base_headers + ret_headers

        self.bt_table.setColumnCount(len(all_headers))
        self.bt_table.setHorizontalHeaderLabels(all_headers)
        self.bt_table.setRowCount(len(picks))

        for row, p in enumerate(picks):
            ss = p.get("sub_scores", {})
            fr = p.get("forward_returns", {})
            items = [
                (f"#{row+1}", None), (p["ticker"], None),
                (f"${p['entry_price']:.4f}", None),
                (f"{p['score']:.1f}", self._score_color(p["score"])),
                (f"{ss.get('setup', 0):.0f}", None),
                (f"{ss.get('technical', 0):.0f}", None),
                (f"{ss.get('pre_pump', 0):.0f}", None),
            ]
            for key in hold_keys:
                r = fr.get(key, {}).get("return_pct")
                if r is not None:
                    sl = fr.get(key, {}).get("stop_loss_triggered", False)
                    text = f"{r:+.1f}%{'*' if sl else ''}"
                    c = QColor("#a6e3a1") if r > 0 else QColor("#f38ba8")
                    items.append((text, c))
                else:
                    items.append(("N/A", None))

            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(color)
                self.bt_table.setItem(row, col, item)

        self.bt_table.setVisible(True)
        self._build_bt_summary(summary, hold_keys)

    def _build_bt_summary(self, summary, hold_keys):
        while self.bt_summary_layout.count():
            child = self.bt_summary_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for group, label in [("top_picks", "Top Picks"), ("all_scored", "All Scored")]:
            hdr = QLabel(f"<b style='color:#f5c2e7'>{label}</b>")
            hdr.setTextFormat(Qt.TextFormat.RichText)
            hdr.setStyleSheet("font-size: 13px; padding: 4px 5px 0 5px; border: none;")
            self.bt_summary_layout.addWidget(hdr)

            for horizon in hold_keys:
                key = f"{group}_{horizon}"
                s = summary.get(key, {})
                if s.get("count", 0) == 0:
                    continue
                wr = s["win_rate"]
                wr_color = "#a6e3a1" if wr > 50 else "#f9e2af" if wr >= 40 else "#f38ba8"
                text = (
                    f"  <b>{horizon}</b>: "
                    f"<span style='color:{wr_color}'>{wr:.0f}%W</span> "
                    f"Avg:{s['avg_return']:+.1f}% "
                    f"| n={s['count']}"
                )
                lbl = QLabel(text)
                lbl.setTextFormat(Qt.TextFormat.RichText)
                lbl.setStyleSheet("font-size: 11px; padding: 1px 5px 1px 15px; border: none;")
                self.bt_summary_layout.addWidget(lbl)

        self.bt_summary_frame.setVisible(True)

    # ── Optimize sub-tab ──────────────────────────────────────
    def _build_optimize_tab(self):
        widget = QWidget()
        self._opt_worker = None
        self._opt_result = None
        layout = QVBoxLayout()

        desc = QLabel(
            "Runs the algorithm on ~70 dates over 3 years.\n"
            "Tests thousands of sell strategies to find optimal parameters.\n"
            "This takes 15-30 minutes. Results can be applied to the live algorithm."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 5px 10px;")
        layout.addWidget(desc)

        btn_row = QHBoxLayout()
        self.opt_run_btn = QPushButton("Run Full Optimization")
        self.opt_run_btn.setStyleSheet(
            "QPushButton { background-color: #f5c2e7; color: #1e1e2e; "
            "padding: 12px 30px; font-size: 14px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #f38ba8; }"
            "QPushButton:disabled { background-color: #585b70; color: #6c7086; }"
        )
        self.opt_run_btn.clicked.connect(self._start_optimization)
        btn_row.addWidget(self.opt_run_btn)

        self.opt_apply_btn = QPushButton("Apply Recommendations")
        self.opt_apply_btn.setStyleSheet(
            "QPushButton { background-color: #a6e3a1; color: #1e1e2e; "
            "padding: 12px 20px; font-size: 13px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #94e2d5; }"
            "QPushButton:disabled { background-color: #585b70; color: #6c7086; }"
        )
        self.opt_apply_btn.clicked.connect(self._apply_recommendations)
        self.opt_apply_btn.setVisible(False)
        btn_row.addWidget(self.opt_apply_btn)

        self.opt_reset_btn = QPushButton("Reset to Defaults")
        self.opt_reset_btn.setStyleSheet(
            "QPushButton { background-color: #fab387; color: #1e1e2e; "
            "padding: 12px 20px; font-size: 13px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #f38ba8; }"
        )
        self.opt_reset_btn.clicked.connect(self._reset_config)
        self.opt_reset_btn.setVisible(False)
        btn_row.addWidget(self.opt_reset_btn)

        self.opt_status = QLabel("")
        btn_row.addWidget(self.opt_status)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.opt_progress = QProgressBar()
        self.opt_progress.setRange(0, 0)
        self.opt_progress.setVisible(False)
        layout.addWidget(self.opt_progress)

        self.opt_summary_frame = QFrame()
        self.opt_summary_frame.setStyleSheet(
            "QFrame { background-color: #181825; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 10px; }"
        )
        self.opt_summary_frame.setVisible(False)
        self.opt_summary_layout = QVBoxLayout()
        self.opt_summary_frame.setLayout(self.opt_summary_layout)
        layout.addWidget(self.opt_summary_frame)

        self.opt_table = QTableWidget()
        self.opt_table.setColumnCount(10)
        self.opt_table.setHorizontalHeaderLabels([
            "#", "Top N", "Hold", "Stop-Loss", "Take-Profit",
            "Trail", "Win Rate", "Avg Ret", "Med Ret", "Trades",
        ])
        self.opt_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.opt_table.setVisible(False)
        layout.addWidget(self.opt_table)

        self.opt_log = QTextEdit()
        self.opt_log.setReadOnly(True)
        layout.addWidget(self.opt_log)

        widget.setLayout(layout)

        # Check if optimized config already exists
        self._check_optimized_config()
        return widget

    def _check_optimized_config(self):
        opt_path = os.path.join(os.path.dirname(__file__), "..", "optimized_config.json")
        if os.path.exists(os.path.normpath(opt_path)):
            self.opt_reset_btn.setVisible(True)
            self.opt_status.setText("Optimized config active")
            self.opt_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")

    def _start_optimization(self):
        self.opt_run_btn.setEnabled(False)
        self.opt_progress.setVisible(True)
        self.opt_apply_btn.setVisible(False)
        self.opt_summary_frame.setVisible(False)
        self.opt_table.setVisible(False)
        self.opt_log.clear()
        self.opt_status.setText("Running full optimization...")
        self.opt_status.setStyleSheet("color: #89b4fa;")

        from pennystock.backtest.optimizer import run_algorithm_optimization
        self._opt_worker = Worker(run_algorithm_optimization)
        self._opt_worker.progress.connect(lambda m: self._append_log(self.opt_log, m))
        self._opt_worker.finished.connect(self._optimization_done)
        self._opt_worker.start()

    def _optimization_done(self, result):
        self.opt_progress.setVisible(False)
        self.opt_run_btn.setEnabled(True)
        self._opt_result = result

        if not result or "error" in result:
            self.opt_status.setText("Optimization failed -- check log")
            self.opt_status.setStyleSheet("color: #f38ba8; font-weight: bold;")
            return

        recs = result.get("recommendations", {})
        sell = recs.get("sell_strategy", {})
        sell_wr = sell.get("expected_win_rate", 0)
        sell_avg = sell.get("expected_avg_return", 0)
        dates = result.get("dates_tested", 0)

        self.opt_status.setText(
            f"Done! {dates} dates | Best: {sell_wr:.0f}% win, {sell_avg:+.1f}% avg"
        )
        color = "#a6e3a1" if sell_wr > 50 else "#f9e2af"
        self.opt_status.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.opt_apply_btn.setVisible(True)

        # Build summary
        self._build_opt_summary(recs)

        # Build strategies table
        top_20 = result.get("sell_optimization", {}).get("top_20", [])
        if top_20:
            self.opt_table.setRowCount(len(top_20))
            for row, r in enumerate(top_20):
                trail_str = (f"{r['trail_activate']}/{r['trail_distance']}"
                             if r['trail_activate'] > 0 else "off")
                sl_str = f"{r['stop_loss']}%" if r['stop_loss'] != 0 else "off"
                tp_str = f"+{r['take_profit']}%" if r['take_profit'] != 0 else "off"
                items = [
                    (f"#{row+1}", None), (str(r["top_n"]), None),
                    (f"{r['hold_days']}d", None), (sl_str, None), (tp_str, None),
                    (trail_str, None),
                    (f"{r['win_rate']:.0f}%",
                     QColor("#a6e3a1") if r["win_rate"] > 50 else QColor("#f38ba8")),
                    (f"{r['avg_return']:+.1f}%",
                     QColor("#a6e3a1") if r["avg_return"] > 0 else QColor("#f38ba8")),
                    (f"{r['median_return']:+.1f}%",
                     QColor("#a6e3a1") if r["median_return"] > 0 else QColor("#f38ba8")),
                    (str(r["total_trades"]), None),
                ]
                for col, (text, color) in enumerate(items):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if color:
                        item.setForeground(color)
                    self.opt_table.setItem(row, col, item)
            self.opt_table.setVisible(True)

    def _build_opt_summary(self, recs):
        while self.opt_summary_layout.count():
            child = self.opt_summary_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        sell = recs.get("sell_strategy", {})
        if sell:
            hdr = QLabel("<b style='color:#f5c2e7; font-size:14px;'>Recommended Sell Strategy</b>")
            hdr.setTextFormat(Qt.TextFormat.RichText)
            hdr.setStyleSheet("border: none; padding: 4px 5px;")
            self.opt_summary_layout.addWidget(hdr)

            hold = sell.get("SELL_MAX_HOLD_DAYS", "?")
            sl = sell.get("SELL_STOP_LOSS_PCT", 0)
            tp = sell.get("SELL_TAKE_PROFIT_PCT", 0)
            ta = sell.get("SELL_TRAILING_STOP_ACTIVATE", 0)
            td = sell.get("SELL_TRAILING_STOP_DISTANCE", 0)
            wr = sell.get("expected_win_rate", 0)
            ar = sell.get("expected_avg_return", 0)

            sl_str = f"{sl}%" if sl != 0 else "disabled"
            tp_str = f"+{tp}%" if tp != 0 else "disabled"
            trail_str = f"activate at +{ta}%, trail {td}% from peak" if ta > 0 else "disabled"
            wr_color = "#a6e3a1" if wr > 50 else "#f9e2af" if wr >= 40 else "#f38ba8"

            text = (
                f"  Hold: <b>{hold}d</b> | Stop-Loss: <b>{sl_str}</b> | "
                f"Take-Profit: <b>{tp_str}</b> | Trailing: <b>{trail_str}</b><br>"
                f"  Expected: <span style='color:{wr_color}'><b>{wr:.0f}% win, "
                f"{ar:+.1f}% avg return</b></span>"
            )
            lbl = QLabel(text)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("font-size: 12px; padding: 2px 15px; border: none;")
            self.opt_summary_layout.addWidget(lbl)

        self.opt_summary_frame.setVisible(True)

    def _apply_recommendations(self):
        if not self._opt_result:
            return
        recs = self._opt_result.get("recommendations", {})
        if not recs:
            return
        from pennystock.backtest.optimizer import apply_optimized_config
        path = apply_optimized_config(recs)
        self.opt_status.setText(f"Applied! Saved to {path}")
        self.opt_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self.opt_apply_btn.setVisible(False)
        self.opt_reset_btn.setVisible(True)

    def _reset_config(self):
        opt_path = os.path.join(os.path.dirname(__file__), "..", "optimized_config.json")
        opt_path = os.path.normpath(opt_path)
        if os.path.exists(opt_path):
            os.remove(opt_path)
            self.opt_status.setText("Reset to defaults. Restart to take effect.")
            self.opt_status.setStyleSheet("color: #fab387; font-weight: bold;")
            self.opt_reset_btn.setVisible(False)

    # ── Shared helpers ────────────────────────────────────────
    @staticmethod
    def _append_log(log_widget, msg):
        log_widget.append(msg)
        scrollbar = log_widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @staticmethod
    def _score_color(score):
        if score >= 65:
            return QColor("#a6e3a1")
        elif score >= 50:
            return QColor("#f9e2af")
        else:
            return QColor("#f38ba8")


# ═══════════════════════════════════════════════════════════════════
# TAB 2: STOCK PICKER
# ═══════════════════════════════════════════════════════════════════

class StockPickerTab(QWidget):
    """Picks top 5-8 stocks based on the algorithm, shows market/sector sentiment."""

    def __init__(self):
        super().__init__()
        self.worker = None
        layout = QVBoxLayout()

        header = QLabel("Stock Picker")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #f5c2e7; padding: 10px;")
        layout.addWidget(header)

        desc = QLabel(
            "Scans all $2-$5 stocks and applies the algorithm to find today's top picks.\n"
            "Includes market & sector sentiment as scoring factors. Use these picks to decide what to buy."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 0 10px 5px 10px;")
        layout.addWidget(desc)

        # Market sentiment display
        self.market_frame = QFrame()
        self.market_frame.setStyleSheet(
            "QFrame { background-color: #181825; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 8px; }"
        )
        self.market_grid = QGridLayout()
        self._market_labels = {}
        for col, (key, title, default) in enumerate([
            ("market", "Market Trend", "--"),
            ("vix", "VIX Level", "--"),
            ("spy", "SPY 5D", "--"),
            ("sentiment", "Conditions", "Run scan to check"),
        ]):
            title_lbl = QLabel(title)
            title_lbl.setStyleSheet("color: #585b70; font-size: 10px; border: none;")
            title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_lbl = QLabel(default)
            val_lbl.setStyleSheet("color: #cdd6f4; font-size: 14px; font-weight: bold; border: none;")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.market_grid.addWidget(val_lbl, 0, col)
            self.market_grid.addWidget(title_lbl, 1, col)
            self._market_labels[key] = val_lbl
        self.market_frame.setLayout(self.market_grid)
        layout.addWidget(self.market_frame)

        # Algorithm info + button row
        algo = load_algorithm()
        btn_row = QHBoxLayout()
        if algo:
            self.algo_info = QLabel(
                f"Algorithm loaded (built {algo.get('built_date', 'unknown')}, "
                f"{len(algo.get('factors', []))} factors)"
            )
            self.algo_info.setStyleSheet("color: #a6e3a1; padding: 0 10px;")
        else:
            self.algo_info = QLabel("No algorithm found. Build one first in Algorithm Builder tab.")
            self.algo_info.setStyleSheet("color: #fab387; padding: 0 10px;")
        btn_row.addWidget(self.algo_info)

        self.pick_btn = QPushButton("Find Top Stocks")
        self.pick_btn.clicked.connect(self._start_pick)
        btn_row.addWidget(self.pick_btn)

        self.status_label = QLabel("")
        btn_row.addWidget(self.status_label)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels([
            "Rank", "Ticker", "Price", "Score",
            "Setup", "Technical", "Pre-Pump", "Fundamental", "Catalyst",
            "Market", "Key Info"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setVisible(False)
        layout.addWidget(self.table)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def _start_pick(self):
        algo = load_algorithm()
        if not algo:
            self.algo_info.setText("No algorithm found. Build one first in Algorithm Builder tab.")
            self.algo_info.setStyleSheet("color: #f38ba8; padding: 0 10px;")
            return

        self.algo_info.setText(
            f"Algorithm loaded (built {algo.get('built_date', 'unknown')}, "
            f"{len(algo.get('factors', []))} factors)"
        )
        self.algo_info.setStyleSheet("color: #a6e3a1; padding: 0 10px;")

        self.pick_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.table.setVisible(False)
        self.log.clear()
        self._append_log("Starting stock scan with market & sector sentiment...")

        self.worker = Worker(pick_stocks, top_n=5)
        self.worker.progress.connect(self._append_log)
        self.worker.finished.connect(self._pick_done)
        self.worker.start()

    def _append_log(self, msg):
        self.log.append(msg)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _pick_done(self, picks):
        self.progress_bar.setVisible(False)
        self.pick_btn.setEnabled(True)

        if not picks:
            self.status_label.setText("No picks found.")
            self.status_label.setStyleSheet("color: #f38ba8;")
            return

        self.status_label.setText(f"Found {len(picks)} picks!")
        self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")

        # Update market sentiment display from first pick's data
        if picks:
            ss = picks[0].get("sub_scores", {})
            mkt_score = ss.get("market", 50)
            sector_score = ss.get("sector", 50)

            trend = "Bullish" if mkt_score > 60 else "Bearish" if mkt_score < 40 else "Neutral"
            trend_color = "#a6e3a1" if mkt_score > 60 else "#f38ba8" if mkt_score < 40 else "#f9e2af"
            self._market_labels["market"].setText(trend)
            self._market_labels["market"].setStyleSheet(
                f"color: {trend_color}; font-size: 14px; font-weight: bold; border: none;"
            )

            cond = "Favorable" if mkt_score > 55 else "Caution" if mkt_score > 40 else "Risky"
            cond_color = "#a6e3a1" if mkt_score > 55 else "#f9e2af" if mkt_score > 40 else "#f38ba8"
            self._market_labels["sentiment"].setText(cond)
            self._market_labels["sentiment"].setStyleSheet(
                f"color: {cond_color}; font-size: 14px; font-weight: bold; border: none;"
            )

        # Populate table
        self.table.setRowCount(len(picks))
        for row, pick in enumerate(picks):
            ss = pick.get("sub_scores", {})
            ki = pick.get("key_indicators", {})

            float_val = ki.get("float_shares", 0) or 0
            float_str = f"{float_val/1e6:.1f}M" if float_val > 0 else "N/A"
            insider_val = (ki.get("insider_pct") or 0) * 100
            key_info = f"Float:{float_str} Ins:{insider_val:.0f}%"

            items = [
                (f"#{row + 1}", None),
                (pick["ticker"], None),
                (f"${pick['price']:.2f}", None),
                (f"{pick['final_score']:.1f}", self._score_color(pick["final_score"])),
                (f"{ss.get('setup', 0):.0f}", self._score_color(ss.get("setup", 0))),
                (f"{ss.get('technical', 0):.0f}", self._score_color(ss.get("technical", 0))),
                (f"{ss.get('pre_pump', 0):.0f}", self._score_color(ss.get("pre_pump", 0))),
                (f"{ss.get('fundamental', 0):.0f}", self._score_color(ss.get("fundamental", 0))),
                (f"{ss.get('catalyst', 0):.0f}", self._score_color(ss.get("catalyst", 0))),
                (f"{ss.get('market', 50):.0f}", self._score_color(ss.get("market", 50))),
                (key_info, None),
            ]
            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(color)
                self.table.setItem(row, col, item)

        self.table.setVisible(True)

    @staticmethod
    def _score_color(score):
        if score >= 70:
            return QColor("#a6e3a1")
        elif score >= 50:
            return QColor("#f9e2af")
        else:
            return QColor("#f38ba8")




# ═══════════════════════════════════════════════════════════════════
# TAB 3: SIMULATION (Autonomous Paper Trading)
# ═══════════════════════════════════════════════════════════════════

class SimulationTab(QWidget):
    """Autonomous paper trading simulator that picks, buys, sells, and learns."""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.engine = None
        self._auto_timer = None
        layout = QVBoxLayout()

        header = QLabel("Simulation  --  Autonomous Paper Trading")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #f5c2e7; padding: 10px;")
        layout.addWidget(header)

        desc = QLabel(
            "Virtual $5,000 portfolio that runs autonomously. The simulator picks stocks, buys, sells,\n"
            "and learns from every trade to improve the algorithm. No alerts are sent -- this is internal only."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 0 10px 5px 10px;")
        layout.addWidget(desc)

        # ── Portfolio Summary Cards ───────────────────────────
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            "QFrame { background-color: #181825; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 10px; }"
        )
        self.summary_grid = QGridLayout()
        self.summary_frame.setLayout(self.summary_grid)
        layout.addWidget(self.summary_frame)

        self._summary_labels = {}
        self._build_summary_cards()

        # ── Button Row ────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh Prices")
        self.refresh_btn.setStyleSheet(
            "QPushButton { background-color: #89b4fa; padding: 10px 20px; }"
            "QPushButton:hover { background-color: #74c7ec; }"
            "QPushButton:disabled { background-color: #585b70; color: #6c7086; }"
        )
        self.refresh_btn.clicked.connect(self._start_refresh)
        btn_row.addWidget(self.refresh_btn)

        self.auto_trade_btn = QPushButton("Run Trade Cycle")
        self.auto_trade_btn.setStyleSheet(
            "QPushButton { background-color: #a6e3a1; color: #1e1e2e; "
            "padding: 10px 20px; font-weight: bold; }"
            "QPushButton:hover { background-color: #94e2d5; }"
            "QPushButton:disabled { background-color: #585b70; color: #6c7086; }"
        )
        self.auto_trade_btn.clicked.connect(self._start_auto_trade)
        btn_row.addWidget(self.auto_trade_btn)

        self.auto_run_btn = QPushButton("Start Auto-Run")
        self.auto_run_btn.setStyleSheet(
            "QPushButton { background-color: #cba6f7; color: #1e1e2e; "
            "padding: 10px 20px; font-weight: bold; }"
            "QPushButton:hover { background-color: #b4befe; }"
        )
        self.auto_run_btn.clicked.connect(self._toggle_auto_run)
        btn_row.addWidget(self.auto_run_btn)

        self.check_signals_btn = QPushButton("Check Sell Signals")
        self.check_signals_btn.setStyleSheet(
            "QPushButton { background-color: #f9e2af; color: #1e1e2e; "
            "padding: 10px 20px; font-weight: bold; }"
            "QPushButton:hover { background-color: #fab387; }"
        )
        self.check_signals_btn.clicked.connect(self._start_check_signals)
        btn_row.addWidget(self.check_signals_btn)

        self.exec_sells_btn = QPushButton("Execute Sell Signals")
        self.exec_sells_btn.setStyleSheet(
            "QPushButton { background-color: #f38ba8; color: #1e1e2e; "
            "padding: 10px 20px; font-weight: bold; }"
            "QPushButton:hover { background-color: #eba0ac; }"
            "QPushButton:disabled { background-color: #585b70; color: #6c7086; }"
        )
        self.exec_sells_btn.setEnabled(False)
        self.exec_sells_btn.clicked.connect(self._execute_pending_sells)
        btn_row.addWidget(self.exec_sells_btn)

        self.reset_btn = QPushButton("Reset Portfolio")
        self.reset_btn.setStyleSheet(
            "QPushButton { background-color: #45475a; color: #cdd6f4; "
            "padding: 10px 20px; font-weight: bold; }"
            "QPushButton:hover { background-color: #585b70; }"
        )
        self.reset_btn.clicked.connect(self._reset_portfolio)
        btn_row.addWidget(self.reset_btn)

        self.status_label = QLabel("")
        btn_row.addWidget(self.status_label)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ── Positions Table ───────────────────────────────────
        pos_label = QLabel("Open Positions")
        pos_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f5c2e7; padding: 8px 10px 2px 10px;")
        layout.addWidget(pos_label)

        self.pos_table = QTableWidget()
        self.pos_table.setColumnCount(10)
        self.pos_table.setHorizontalHeaderLabels([
            "Ticker", "Company", "Shares", "Entry $", "Current $",
            "P&L $", "P&L %", "Days Held", "Score", "Action",
        ])
        self.pos_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pos_table.setMaximumHeight(250)
        layout.addWidget(self.pos_table)

        # ── Trade History & Learning split ─────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        history_widget = QWidget()
        hist_layout = QVBoxLayout()
        hist_layout.setContentsMargins(0, 0, 0, 0)
        hist_label = QLabel("Trade History")
        hist_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f5c2e7; padding: 4px 0;")
        hist_layout.addWidget(hist_label)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels([
            "Date", "Action", "Ticker", "Shares", "Price", "P&L %", "Reason",
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hist_layout.addWidget(self.history_table)
        history_widget.setLayout(hist_layout)
        splitter.addWidget(history_widget)

        learn_widget = QWidget()
        learn_layout = QVBoxLayout()
        learn_layout.setContentsMargins(0, 0, 0, 0)
        learn_label = QLabel("Self-Learning Insights")
        learn_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f5c2e7; padding: 4px 0;")
        learn_layout.addWidget(learn_label)

        self.insights_log = QTextEdit()
        self.insights_log.setReadOnly(True)
        self.insights_log.setStyleSheet(
            "QTextEdit { background-color: #11111b; color: #cdd6f4; "
            "border: 1px solid #45475a; font-family: 'Consolas', 'Courier New', monospace; "
            "font-size: 11px; padding: 6px; }"
        )
        learn_layout.addWidget(self.insights_log)
        learn_widget.setLayout(learn_layout)
        splitter.addWidget(learn_widget)

        layout.addWidget(splitter)

        # ── Activity Log ──────────────────────────────────────
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        layout.addWidget(self.log)

        self.setLayout(layout)
        self._pending_sells = []

        # Load state and auto-refresh prices on startup
        QTimer.singleShot(100, self._initial_load)

    def _get_engine(self):
        if self.engine is None:
            from pennystock.simulation.engine import SimulationEngine
            self.engine = SimulationEngine()
        else:
            self.engine.state = self.engine._load_state()
        return self.engine

    def _initial_load(self):
        """Load saved state and auto-refresh prices if there are positions."""
        engine = self._get_engine()
        self._update_summary(engine.get_portfolio_summary())
        self._update_positions(engine.state.get("positions", []))
        self._update_history(engine.state.get("trade_history", []))
        self._update_insights(engine)
        # Auto-refresh prices on load if positions exist
        if engine.state.get("positions"):
            self._append_log("Loading saved simulation state...")
            self._append_log(f"  {len(engine.state['positions'])} positions, "
                             f"{len(engine.state.get('trade_history', []))} trades in history")
            self._start_refresh()

    def _load_and_display(self):
        engine = self._get_engine()
        self._update_summary(engine.get_portfolio_summary())
        self._update_positions(engine.state.get("positions", []))
        self._update_history(engine.state.get("trade_history", []))
        self._update_insights(engine)

    def _build_summary_cards(self):
        cards = [
            ("total_value", "Portfolio Value", "$5,000.00", "#cdd6f4"),
            ("cash", "Cash", "$5,000.00", "#89b4fa"),
            ("unrealized_pnl", "Unrealized P&L", "$0.00", "#cdd6f4"),
            ("realized_pnl", "Realized P&L", "$0.00", "#cdd6f4"),
            ("total_return_pct", "Total Return", "0.0%", "#cdd6f4"),
            ("win_rate", "Win Rate", "0.0%", "#6c7086"),
            ("total_trades", "Total Trades", "0", "#6c7086"),
            ("num_positions", "Positions", "0/10", "#6c7086"),
        ]
        for col, (key, label, default, color) in enumerate(cards):
            card = QFrame()
            card.setStyleSheet(
                "QFrame { background-color: #1e1e2e; border: 1px solid #45475a; "
                "border-radius: 4px; padding: 4px; margin: 2px; }"
            )
            card_layout = QVBoxLayout()
            card_layout.setSpacing(2)
            card_layout.setContentsMargins(6, 4, 6, 4)

            val_label = QLabel(default)
            val_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold; border: none;")
            val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(val_label)

            name_label = QLabel(label)
            name_label.setStyleSheet("color: #6c7086; font-size: 10px; border: none;")
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(name_label)

            card.setLayout(card_layout)
            self.summary_grid.addWidget(card, 0, col)
            self._summary_labels[key] = val_label

    def _update_summary(self, summary):
        from pennystock.config import SIMULATION_MAX_POSITIONS
        tv = summary["total_value"]
        ret_pct = summary["total_return_pct"]
        wr = summary["win_rate"]
        ur = summary["unrealized_pnl"]
        rr = summary["realized_pnl"]

        self._summary_labels["total_value"].setText(f"${tv:,.2f}")
        tv_color = "#a6e3a1" if ret_pct >= 0 else "#f38ba8"
        self._summary_labels["total_value"].setStyleSheet(
            f"color: {tv_color}; font-size: 14px; font-weight: bold; border: none;"
        )
        self._summary_labels["cash"].setText(f"${summary['cash']:,.2f}")
        self._summary_labels["unrealized_pnl"].setText(f"${ur:+,.2f}")
        ur_color = "#a6e3a1" if ur >= 0 else "#f38ba8"
        self._summary_labels["unrealized_pnl"].setStyleSheet(
            f"color: {ur_color}; font-size: 14px; font-weight: bold; border: none;"
        )
        self._summary_labels["realized_pnl"].setText(f"${rr:+,.2f}")
        rr_color = "#a6e3a1" if rr >= 0 else "#f38ba8"
        self._summary_labels["realized_pnl"].setStyleSheet(
            f"color: {rr_color}; font-size: 14px; font-weight: bold; border: none;"
        )
        self._summary_labels["total_return_pct"].setText(f"{ret_pct:+.1f}%")
        self._summary_labels["total_return_pct"].setStyleSheet(
            f"color: {tv_color}; font-size: 14px; font-weight: bold; border: none;"
        )
        self._summary_labels["win_rate"].setText(f"{wr:.0f}%")
        wr_color = "#a6e3a1" if wr > 50 else "#f9e2af" if wr >= 40 else "#f38ba8" if summary["total_trades"] > 0 else "#6c7086"
        self._summary_labels["win_rate"].setStyleSheet(
            f"color: {wr_color}; font-size: 14px; font-weight: bold; border: none;"
        )
        self._summary_labels["total_trades"].setText(str(summary["total_trades"]))
        self._summary_labels["num_positions"].setText(
            f"{summary['num_positions']}/{SIMULATION_MAX_POSITIONS}"
        )

    def _update_positions(self, positions):
        self.pos_table.setRowCount(len(positions))
        from datetime import datetime
        for row, pos in enumerate(positions):
            entry = pos["entry_price"]
            current = pos.get("current_price", entry)
            pnl = (current - entry) * pos["shares"]
            pnl_pct = ((current - entry) / entry) * 100 if entry > 0 else 0
            try:
                entry_date = datetime.fromisoformat(pos["entry_date"])
                days_held = (datetime.now() - entry_date).days
            except Exception:
                days_held = 0
            pnl_color = QColor("#a6e3a1") if pnl >= 0 else QColor("#f38ba8")
            items = [
                (pos["ticker"], None), (pos.get("company", "")[:20], None),
                (str(pos["shares"]), None), (f"${entry:.4f}", None),
                (f"${current:.4f}", None), (f"${pnl:+.2f}", pnl_color),
                (f"{pnl_pct:+.1f}%", pnl_color), (str(days_held), None),
                (f"{pos.get('entry_score', 0):.0f}", None),
            ]
            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(color)
                self.pos_table.setItem(row, col, item)

            # Sell button in last column
            sell_btn = QPushButton("Sell")
            sell_btn.setStyleSheet(
                "QPushButton { background-color: #f38ba8; color: #1e1e2e; "
                "padding: 4px 10px; font-size: 11px; font-weight: bold; }"
            )
            sell_btn.clicked.connect(lambda checked, t=pos["ticker"]: self._manual_sell(t))
            self.pos_table.setCellWidget(row, 9, sell_btn)

    def _manual_sell(self, ticker):
        """Manually sell a simulation position."""
        engine = self._get_engine()
        pos = None
        for p in engine.state["positions"]:
            if p["ticker"] == ticker:
                pos = p
                break
        if not pos:
            return
        sells = [(pos, "manual_sell", "Manual sell from GUI")]
        engine.execute_sells(sells, progress_callback=self._append_log)
        self._append_log(f"Manually sold {ticker}")
        self._load_and_display()

    def _update_history(self, trade_history):
        trades = list(reversed(trade_history[-50:]))
        self.history_table.setRowCount(len(trades))
        for row, t in enumerate(trades):
            is_buy = t["action"] == "BUY"
            action_color = QColor("#89b4fa") if is_buy else (
                QColor("#a6e3a1") if t.get("return_pct", 0) > 0 else QColor("#f38ba8")
            )
            pnl_str = f"{t.get('return_pct', 0):+.1f}%" if not is_buy else ""
            reason = t.get("sell_reason", t.get("description", ""))
            items = [
                (t.get("date", "")[:16], None), (t["action"], action_color),
                (t["ticker"], None), (str(t["shares"]), None),
                (f"${t['price']:.4f}", None),
                (pnl_str, action_color if not is_buy else None),
                (reason, None),
            ]
            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(color)
                self.history_table.setItem(row, col, item)

    def _update_insights(self, engine):
        self.insights_log.clear()
        learning = engine.get_learning_summary()
        if learning:
            self.insights_log.append("=== Performance by Category ===\n")
            score_data = {k: v for k, v in learning.items() if k.startswith("score_")}
            if score_data:
                self.insights_log.append("By Score Range:")
                for k in sorted(score_data.keys()):
                    d = score_data[k]
                    self.insights_log.append(
                        f"  {k.replace('score_', '')}: "
                        f"{d['win_rate']:.0f}%W, avg {d['avg_return']:+.1f}%, n={d['trades']}"
                    )
                self.insights_log.append("")
            pp_data = {k: v for k, v in learning.items() if k.startswith("pp_")}
            if pp_data:
                self.insights_log.append("By Pre-Pump Confidence:")
                for k in sorted(pp_data.keys()):
                    d = pp_data[k]
                    self.insights_log.append(
                        f"  {k.replace('pp_', '')}: "
                        f"{d['win_rate']:.0f}%W, avg {d['avg_return']:+.1f}%, n={d['trades']}"
                    )
                self.insights_log.append("")
            reason_data = {k: v for k, v in learning.items() if k.startswith("reason_")}
            if reason_data:
                self.insights_log.append("By Sell Reason:")
                for k in sorted(reason_data.keys()):
                    d = reason_data[k]
                    self.insights_log.append(
                        f"  {k.replace('reason_', '')}: "
                        f"{d['win_rate']:.0f}%W, avg {d['avg_return']:+.1f}%, n={d['trades']}"
                    )

        insights = engine.state.get("insights", [])
        if insights:
            self.insights_log.append("\n=== Recent Activity ===\n")
            for ins in reversed(insights[-20:]):
                self.insights_log.append(f"  [{ins['date'][:16]}] {ins['text']}")

        if not learning and not insights:
            self.insights_log.append("No trades yet. Run a Trade Cycle to get started!")

    def _append_log(self, msg):
        self.log.append(msg)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _start_refresh(self):
        self.refresh_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log.clear()
        self.status_label.setText("Refreshing prices...")
        self.status_label.setStyleSheet("color: #89b4fa;")
        engine = self._get_engine()
        self.worker = Worker(engine.refresh_prices)
        self.worker.progress.connect(self._append_log)
        self.worker.finished.connect(self._refresh_done)
        self.worker.start()

    def _refresh_done(self, _result):
        self.progress_bar.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.status_label.setText("Prices updated")
        self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self._load_and_display()

    def _start_auto_trade(self):
        self.auto_trade_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log.clear()
        self.status_label.setText("Running auto-trade cycle...")
        self.status_label.setStyleSheet("color: #89b4fa;")
        engine = self._get_engine()
        self.worker = Worker(engine.run_auto_cycle)
        self.worker.progress.connect(self._append_log)
        self.worker.finished.connect(self._auto_trade_done)
        self.worker.start()

    def _auto_trade_done(self, _result):
        self.progress_bar.setVisible(False)
        self.auto_trade_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        engine = self._get_engine()
        summary = engine.get_portfolio_summary()
        ret = summary["total_return_pct"]
        color = "#a6e3a1" if ret >= 0 else "#f38ba8"
        self.status_label.setText(
            f"Cycle complete | ${summary['total_value']:,.2f} ({ret:+.1f}%)"
        )
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._load_and_display()

    def _toggle_auto_run(self):
        """Toggle autonomous auto-run timer."""
        if self._auto_timer and self._auto_timer.isActive():
            self._auto_timer.stop()
            self.auto_run_btn.setText("Start Auto-Run")
            self.auto_run_btn.setStyleSheet(
                "QPushButton { background-color: #cba6f7; color: #1e1e2e; "
                "padding: 10px 20px; font-weight: bold; }"
                "QPushButton:hover { background-color: #b4befe; }"
            )
            self._append_log("Auto-run stopped")
        else:
            self._auto_timer = QTimer()
            self._auto_timer.timeout.connect(self._auto_run_cycle)
            self._auto_timer.start(4 * 60 * 60 * 1000)  # Every 4 hours
            self.auto_run_btn.setText("Stop Auto-Run")
            self.auto_run_btn.setStyleSheet(
                "QPushButton { background-color: #f38ba8; color: #1e1e2e; "
                "padding: 10px 20px; font-weight: bold; }"
                "QPushButton:hover { background-color: #fab387; }"
            )
            self._append_log("Auto-run started (every 4 hours)")
            # Run immediately
            self._auto_run_cycle()

    def _auto_run_cycle(self):
        """Triggered by auto-run timer."""
        if self.worker and self.worker.isRunning():
            return  # Skip if already running
        self._start_auto_trade()

    def _start_check_signals(self):
        """Check sell signals for current positions without auto-executing."""
        self.check_signals_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log.clear()
        self.status_label.setText("Checking sell signals...")
        self.status_label.setStyleSheet("color: #89b4fa;")
        engine = self._get_engine()

        def _run(progress_callback=None):
            engine.refresh_prices(progress_callback=progress_callback)
            return engine.check_sell_signals()

        self.worker = Worker(_run)
        self.worker.progress.connect(self._append_log)
        self.worker.finished.connect(self._check_signals_done)
        self.worker.start()

    def _check_signals_done(self, sells):
        self.progress_bar.setVisible(False)
        self.check_signals_btn.setEnabled(True)
        self._pending_sells = sells or []
        if sells:
            self._append_log(f"\n{len(sells)} sell signal(s) triggered:")
            for pos, reason, desc in sells:
                entry = pos["entry_price"]
                current = pos.get("current_price", entry)
                ret = ((current - entry) / entry) * 100
                self._append_log(f"  SELL {pos['ticker']}: {desc} ({ret:+.1f}%)")
            self.exec_sells_btn.setEnabled(True)
            self.status_label.setText(f"{len(sells)} sell signal(s) — click Execute to sell")
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
        else:
            self.exec_sells_btn.setEnabled(False)
            self._append_log("No sell signals triggered.")
            self.status_label.setText("No sell signals")
            self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self._load_and_display()

    def _execute_pending_sells(self):
        """Execute the sell signals found by _start_check_signals."""
        sells = getattr(self, "_pending_sells", [])
        if not sells:
            return
        engine = self._get_engine()
        engine.execute_sells(sells, progress_callback=self._append_log)
        tickers = [pos["ticker"] for pos, _, _ in sells]
        self._append_log(f"\nExecuted sells: {', '.join(tickers)}")
        self._pending_sells = []
        self.exec_sells_btn.setEnabled(False)
        self.status_label.setText(f"Sold {len(tickers)} position(s)")
        self.status_label.setStyleSheet("color: #fab387; font-weight: bold;")
        self._load_and_display()

    def _reset_portfolio(self):
        engine = self._get_engine()
        if engine.state.get("trade_history"):
            self._append_log("Resetting portfolio to $5,000...")
        engine.reset()
        self.status_label.setText("Portfolio reset to $5,000")
        self.status_label.setStyleSheet("color: #fab387; font-weight: bold;")
        self._load_and_display()



# ═══════════════════════════════════════════════════════════════════
# TAB 4: MY PORTFOLIO (Real Holdings + Alerts)
# ═══════════════════════════════════════════════════════════════════

class MyPortfolioTab(QWidget):
    """Track real stock holdings. Alerts are tied to this portfolio."""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.manager = None
        layout = QVBoxLayout()

        header = QLabel("My Portfolio  --  Real Holdings")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #f5c2e7; padding: 10px;")
        layout.addWidget(header)

        desc = QLabel(
            "Add stocks you've actually purchased. Email alerts (buy/sell signals) are tied to this portfolio.\n"
            "Buy signals come from highly recommended picks. Sell signals trigger on stop-loss, take-profit, etc."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 0 10px 5px 10px;")
        layout.addWidget(desc)

        # ── Portfolio Summary Cards ───────────────────────────
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            "QFrame { background-color: #181825; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 10px; }"
        )
        self.summary_grid = QGridLayout()
        self.summary_frame.setLayout(self.summary_grid)
        layout.addWidget(self.summary_frame)

        self._summary_labels = {}
        self._build_summary_cards()

        # ── Add Position Row ──────────────────────────────────
        add_frame = QFrame()
        add_frame.setStyleSheet(
            "QFrame { background-color: #181825; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 8px; margin: 4px 0; }"
        )
        add_row = QHBoxLayout()

        add_label = QLabel("Add Position:")
        add_label.setStyleSheet("font-weight: bold; color: #f5c2e7; border: none;")
        add_row.addWidget(add_label)

        add_row.addWidget(self._make_label("Ticker:"))
        self.add_ticker = QLineEdit()
        self.add_ticker.setPlaceholderText("e.g. AAPL")
        self.add_ticker.setMaximumWidth(100)
        add_row.addWidget(self.add_ticker)

        add_row.addWidget(self._make_label("Shares:"))
        self.add_shares = QSpinBox()
        self.add_shares.setRange(1, 999999)
        self.add_shares.setValue(100)
        self.add_shares.setMaximumWidth(100)
        add_row.addWidget(self.add_shares)

        add_row.addWidget(self._make_label("Entry $:"))
        self.add_price = QDoubleSpinBox()
        self.add_price.setRange(0.01, 999.99)
        self.add_price.setValue(1.00)
        self.add_price.setDecimals(4)
        self.add_price.setMaximumWidth(120)
        add_row.addWidget(self.add_price)

        self.add_btn = QPushButton("Add")
        self.add_btn.setStyleSheet(
            "QPushButton { background-color: #a6e3a1; color: #1e1e2e; "
            "padding: 8px 20px; font-weight: bold; }"
        )
        self.add_btn.clicked.connect(self._add_position)
        add_row.addWidget(self.add_btn)

        add_row.addStretch()
        add_frame.setLayout(add_row)
        layout.addWidget(add_frame)

        # ── Button Row ────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh Prices")
        self.refresh_btn.setStyleSheet(
            "QPushButton { background-color: #89b4fa; padding: 10px 20px; }"
            "QPushButton:hover { background-color: #74c7ec; }"
            "QPushButton:disabled { background-color: #585b70; color: #6c7086; }"
        )
        self.refresh_btn.clicked.connect(self._start_refresh)
        btn_row.addWidget(self.refresh_btn)

        self.check_signals_btn = QPushButton("Check Sell Signals")
        self.check_signals_btn.setStyleSheet(
            "QPushButton { background-color: #f9e2af; color: #1e1e2e; "
            "padding: 10px 20px; font-weight: bold; }"
            "QPushButton:hover { background-color: #fab387; }"
        )
        self.check_signals_btn.clicked.connect(self._check_sell_signals)
        btn_row.addWidget(self.check_signals_btn)

        self.status_label = QLabel("")
        btn_row.addWidget(self.status_label)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ── Positions Table ───────────────────────────────────
        pos_label = QLabel("Holdings")
        pos_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f5c2e7; padding: 8px 10px 2px 10px;")
        layout.addWidget(pos_label)

        self.pos_table = QTableWidget()
        self.pos_table.setColumnCount(9)
        self.pos_table.setHorizontalHeaderLabels([
            "Ticker", "Shares", "Entry $", "Current $",
            "P&L $", "P&L %", "Days Held", "Trail Active", "Action",
        ])
        self.pos_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.pos_table)

        # ── Trade History ─────────────────────────────────────
        hist_label = QLabel("Trade History")
        hist_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f5c2e7; padding: 8px 10px 2px 10px;")
        layout.addWidget(hist_label)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels([
            "Date", "Action", "Ticker", "Shares", "Price", "P&L %", "Reason",
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setMaximumHeight(200)
        layout.addWidget(self.history_table)

        # ── Activity Log ──────────────────────────────────────
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)

        self.setLayout(layout)
        QTimer.singleShot(100, self._load_and_display)

    def _get_manager(self):
        if self.manager is None:
            from pennystock.portfolio.manager import PortfolioManager
            self.manager = PortfolioManager()
        else:
            self.manager.state = self.manager._load_state()
        return self.manager

    def _make_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; border: none;")
        return lbl

    def _build_summary_cards(self):
        cards = [
            ("total_value", "Portfolio Value", "$0.00", "#cdd6f4"),
            ("cost_basis", "Cost Basis", "$0.00", "#89b4fa"),
            ("unrealized_pnl", "Unrealized P&L", "$0.00", "#cdd6f4"),
            ("realized_pnl", "Realized P&L", "$0.00", "#cdd6f4"),
            ("total_pnl", "Total P&L", "$0.00", "#cdd6f4"),
            ("win_rate", "Win Rate", "0.0%", "#6c7086"),
            ("num_positions", "Positions", "0", "#6c7086"),
        ]
        for col, (key, label, default, color) in enumerate(cards):
            card = QFrame()
            card.setStyleSheet(
                "QFrame { background-color: #1e1e2e; border: 1px solid #45475a; "
                "border-radius: 4px; padding: 4px; margin: 2px; }"
            )
            card_layout = QVBoxLayout()
            card_layout.setSpacing(2)
            card_layout.setContentsMargins(6, 4, 6, 4)

            val_label = QLabel(default)
            val_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold; border: none;")
            val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(val_label)

            name_label = QLabel(label)
            name_label.setStyleSheet("color: #6c7086; font-size: 10px; border: none;")
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(name_label)

            card.setLayout(card_layout)
            self.summary_grid.addWidget(card, 0, col)
            self._summary_labels[key] = val_label

    def _load_and_display(self):
        mgr = self._get_manager()
        self._update_summary(mgr.get_portfolio_summary())
        self._update_positions(mgr.state.get("positions", []))
        self._update_history(mgr.state.get("trade_history", []))

    def _update_summary(self, summary):
        self._summary_labels["total_value"].setText(f"${summary['total_value']:,.2f}")
        self._summary_labels["cost_basis"].setText(f"${summary['cost_basis']:,.2f}")

        ur = summary["unrealized_pnl"]
        self._summary_labels["unrealized_pnl"].setText(f"${ur:+,.2f}")
        ur_color = "#a6e3a1" if ur >= 0 else "#f38ba8"
        self._summary_labels["unrealized_pnl"].setStyleSheet(
            f"color: {ur_color}; font-size: 14px; font-weight: bold; border: none;"
        )

        rr = summary["realized_pnl"]
        self._summary_labels["realized_pnl"].setText(f"${rr:+,.2f}")
        rr_color = "#a6e3a1" if rr >= 0 else "#f38ba8"
        self._summary_labels["realized_pnl"].setStyleSheet(
            f"color: {rr_color}; font-size: 14px; font-weight: bold; border: none;"
        )

        tp = summary["total_pnl"]
        self._summary_labels["total_pnl"].setText(f"${tp:+,.2f}")
        tp_color = "#a6e3a1" if tp >= 0 else "#f38ba8"
        self._summary_labels["total_pnl"].setStyleSheet(
            f"color: {tp_color}; font-size: 14px; font-weight: bold; border: none;"
        )

        wr = summary["win_rate"]
        wr_color = "#a6e3a1" if wr > 50 else "#f9e2af" if wr >= 40 else "#f38ba8" if summary["total_trades"] > 0 else "#6c7086"
        self._summary_labels["win_rate"].setText(f"{wr:.0f}%")
        self._summary_labels["win_rate"].setStyleSheet(
            f"color: {wr_color}; font-size: 14px; font-weight: bold; border: none;"
        )

        self._summary_labels["num_positions"].setText(str(summary["num_positions"]))

    def _update_positions(self, positions):
        self.pos_table.setRowCount(len(positions))
        from datetime import datetime
        for row, pos in enumerate(positions):
            entry = pos["entry_price"]
            current = pos.get("current_price", entry)
            pnl = (current - entry) * pos["shares"]
            pnl_pct = ((current - entry) / entry) * 100 if entry > 0 else 0
            try:
                entry_date = datetime.fromisoformat(pos["entry_date"])
                days_held = (datetime.now() - entry_date).days
            except Exception:
                days_held = 0
            trail = "YES" if pos.get("trailing_stop_active") else ""
            pnl_color = QColor("#a6e3a1") if pnl >= 0 else QColor("#f38ba8")
            items = [
                (pos["ticker"], None), (str(pos["shares"]), None),
                (f"${entry:.4f}", None), (f"${current:.4f}", None),
                (f"${pnl:+.2f}", pnl_color), (f"{pnl_pct:+.1f}%", pnl_color),
                (str(days_held), None), (trail, None),
            ]
            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(color)
                self.pos_table.setItem(row, col, item)

            # Sell button in last column
            sell_btn = QPushButton("Sell")
            sell_btn.setStyleSheet(
                "QPushButton { background-color: #f38ba8; color: #1e1e2e; "
                "padding: 4px 10px; font-size: 11px; font-weight: bold; }"
            )
            sell_btn.clicked.connect(lambda checked, t=pos["ticker"]: self._sell_position(t))
            self.pos_table.setCellWidget(row, 8, sell_btn)

    def _update_history(self, trade_history):
        trades = list(reversed(trade_history[-30:]))
        self.history_table.setRowCount(len(trades))
        for row, t in enumerate(trades):
            is_buy = t["action"] == "BUY"
            action_color = QColor("#89b4fa") if is_buy else (
                QColor("#a6e3a1") if t.get("return_pct", 0) > 0 else QColor("#f38ba8")
            )
            pnl_str = f"{t.get('return_pct', 0):+.1f}%" if not is_buy else ""
            reason = t.get("sell_reason", "")
            items = [
                (t.get("date", "")[:16], None), (t["action"], action_color),
                (t["ticker"], None), (str(t["shares"]), None),
                (f"${t['price']:.4f}", None),
                (pnl_str, action_color if not is_buy else None),
                (reason, None),
            ]
            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(color)
                self.history_table.setItem(row, col, item)

    def _append_log(self, msg):
        self.log.append(msg)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _add_position(self):
        ticker = self.add_ticker.text().strip().upper()
        shares = self.add_shares.value()
        price = self.add_price.value()

        if not ticker:
            self.status_label.setText("Enter a ticker")
            self.status_label.setStyleSheet("color: #f38ba8;")
            return

        mgr = self._get_manager()
        mgr.add_position(ticker, shares, price, progress_callback=self._append_log)
        self.add_ticker.clear()
        self.status_label.setText(f"Added {ticker}")
        self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self._load_and_display()

    def _sell_position(self, ticker):
        mgr = self._get_manager()
        # Get current price for the sale
        for pos in mgr.state["positions"]:
            if pos["ticker"] == ticker:
                sell_price = pos.get("current_price", pos["entry_price"])
                mgr.remove_position(ticker, sell_price, reason="manual_sell",
                                    progress_callback=self._append_log)
                break
        self._load_and_display()

    def _start_refresh(self):
        self.refresh_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log.clear()
        self.status_label.setText("Refreshing prices...")
        self.status_label.setStyleSheet("color: #89b4fa;")
        mgr = self._get_manager()
        self.worker = Worker(mgr.refresh_prices)
        self.worker.progress.connect(self._append_log)
        self.worker.finished.connect(self._refresh_done)
        self.worker.start()

    def _refresh_done(self, _result):
        self.progress_bar.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.status_label.setText("Prices updated")
        self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self._load_and_display()

    def _check_sell_signals(self):
        self.check_signals_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log.clear()
        self.status_label.setText("Checking sell signals...")
        self.status_label.setStyleSheet("color: #89b4fa;")
        mgr = self._get_manager()

        def _run(progress_callback=None):
            mgr.refresh_prices(progress_callback=progress_callback)
            return mgr.check_sell_signals()

        self.worker = Worker(_run)
        self.worker.progress.connect(self._append_log)
        self.worker.finished.connect(self._check_signals_done)
        self.worker.start()

    def _check_signals_done(self, sells):
        self.progress_bar.setVisible(False)
        self.check_signals_btn.setEnabled(True)
        if sells:
            self._append_log(f"\n{len(sells)} sell signal(s) triggered:")
            for pos, reason, desc in sells:
                entry = pos["entry_price"]
                current = pos.get("current_price", entry)
                ret = ((current - entry) / entry) * 100
                self._append_log(f"  SELL {pos['ticker']}: {desc} ({ret:+.1f}%)")
            self.status_label.setText(f"{len(sells)} sell signal(s)!")
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
        else:
            self._append_log("No sell signals triggered.")
            self.status_label.setText("No sell signals")
            self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self._load_and_display()



# ═══════════════════════════════════════════════════════════════════
# TAB 5: ANALYZE STOCK (kept as-is)
# ═══════════════════════════════════════════════════════════════════

class AnalyzeStockTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_result = None
        layout = QVBoxLayout()

        header = QLabel("Analyze Stock  --  Deep Dive")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #f5c2e7; padding: 10px;")
        layout.addWidget(header)

        desc = QLabel(
            "Enter a ticker for comprehensive analysis: price action, fundamentals, "
            "pre-pump signals, technicals, sentiment, news, and full algorithm scoring."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 0 10px 5px 10px;")
        layout.addWidget(desc)

        input_row = QHBoxLayout()
        ticker_label = QLabel("Ticker:")
        ticker_label.setStyleSheet("font-size: 14px; font-weight: bold; padding-left: 10px;")
        input_row.addWidget(ticker_label)

        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("e.g. GETY")
        self.ticker_input.setMaximumWidth(150)
        self.ticker_input.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.ticker_input.returnPressed.connect(self._start_analyze)
        input_row.addWidget(self.ticker_input)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self._start_analyze)
        input_row.addWidget(self.analyze_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setStyleSheet(
            "QPushButton { background-color: #a6e3a1; padding: 12px 20px; }"
            "QPushButton:hover { background-color: #94e2d5; }"
            "QPushButton:disabled { background-color: #585b70; color: #6c7086; }"
        )
        self.refresh_btn.clicked.connect(self._start_analyze)
        self.refresh_btn.setVisible(False)
        input_row.addWidget(self.refresh_btn)

        self.status_label = QLabel("")
        input_row.addWidget(self.status_label)
        input_row.addStretch()
        layout.addLayout(input_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            "QFrame { background-color: #181825; border: 1px solid #45475a; border-radius: 6px; padding: 10px; }"
        )
        self.summary_frame.setVisible(False)
        self.summary_layout = QGridLayout()
        self.summary_frame.setLayout(self.summary_layout)
        layout.addWidget(self.summary_frame)

        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setStyleSheet(
            "QTextEdit { background-color: #11111b; color: #cdd6f4; "
            "border: 1px solid #45475a; font-family: 'Consolas', 'Courier New', monospace; "
            "font-size: 12px; padding: 8px; }"
        )
        layout.addWidget(self.report)

        self.setLayout(layout)

    def _start_analyze(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            self.status_label.setText("Enter a ticker symbol")
            self.status_label.setStyleSheet("color: #f38ba8;")
            return

        self.ticker_input.setText(ticker)
        self.analyze_btn.setEnabled(False)
        self.refresh_btn.setVisible(False)
        self.progress_bar.setVisible(True)
        self.summary_frame.setVisible(False)
        self.report.clear()
        self.status_label.setText(f"Analyzing {ticker}...")
        self.status_label.setStyleSheet("color: #89b4fa;")

        from pennystock.analysis.deep_dive import run_deep_dive
        self.worker = Worker(run_deep_dive, ticker)
        self.worker.progress.connect(self._append_report)
        self.worker.finished.connect(self._analyze_done)
        self.worker.start()

    def _append_report(self, msg):
        self.report.append(msg)
        scrollbar = self.report.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _analyze_done(self, result):
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.refresh_btn.setVisible(True)
        self.current_result = result

        if not result:
            self.status_label.setText("Analysis failed - check log")
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
            return

        ticker = result.get("ticker", "")
        score = result.get("final_score", 0)
        confidence = result.get("confidence", "LOW")
        killed = result.get("killed", False)

        if killed:
            self.status_label.setText(f"{ticker}: KILLED by quality filter")
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
        elif confidence == "HIGH":
            self.status_label.setText(f"{ticker}: {score:.1f} pts -- HIGH confidence")
            self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        elif confidence == "MEDIUM":
            self.status_label.setText(f"{ticker}: {score:.1f} pts -- MEDIUM confidence")
            self.status_label.setStyleSheet("color: #f9e2af; font-weight: bold;")
        else:
            self.status_label.setText(f"{ticker}: {score:.1f} pts -- LOW confidence")
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")

        self._build_summary(result)

    def _build_summary(self, result):
        while self.summary_layout.count():
            child = self.summary_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        ss = result.get("sub_scores", {})
        pp = result.get("pre_pump", {})
        info = result.get("info", {})
        movements = result.get("price_movements", {})
        gate = result.get("quality_gate", {})

        price = result.get("price", 0)
        score = result.get("final_score", 0)
        confidence = result.get("confidence", "LOW")
        company = result.get("company", "")

        headline = QLabel(f"  {result.get('ticker', '')}  --  ${price:.4f}  --  Score: {score:.1f} ({confidence})")
        headline.setStyleSheet(
            f"font-size: 16px; font-weight: bold; padding: 5px; "
            f"color: {'#a6e3a1' if confidence == 'HIGH' else '#f9e2af' if confidence == 'MEDIUM' else '#f38ba8'};"
        )
        self.summary_layout.addWidget(headline, 0, 0, 1, 6)

        company_label = QLabel(f"  {company}")
        company_label.setStyleSheet("color: #a6adc8; font-size: 12px; padding-left: 5px;")
        self.summary_layout.addWidget(company_label, 1, 0, 1, 6)

        movements_text = ""
        for label, key in [("1D", "1d"), ("1W", "1w"), ("1M", "1m"), ("3M", "3m"), ("6M", "6m"), ("1Y", "1y")]:
            val = movements.get(key)
            if val is not None:
                color = "#a6e3a1" if val >= 0 else "#f38ba8"
                movements_text += f'<span style="color:#a6adc8">{label}:</span> <span style="color:{color}">{val:+.1f}%</span>  '
        if movements_text:
            mv_label = QLabel(f"  {movements_text}")
            mv_label.setTextFormat(Qt.TextFormat.RichText)
            mv_label.setStyleSheet("font-size: 12px; padding: 3px 5px;")
            self.summary_layout.addWidget(mv_label, 2, 0, 1, 6)

        col = 0
        for name, key in [("Setup", "setup"), ("Technical", "technical"), ("Pre-Pump", "pre_pump"),
                           ("Fundamental", "fundamental"), ("Catalyst", "catalyst")]:
            val = ss.get(key, 0)
            card = self._make_score_card(name, val)
            self.summary_layout.addWidget(card, 3, col)
            col += 1

        penalty = gate.get("total_penalty", 0)
        if penalty > 0:
            pen_card = self._make_card("Penalties", f"-{penalty}pts", "#f38ba8")
            self.summary_layout.addWidget(pen_card, 3, col)

        col = 0
        float_shares = info.get("float_shares", 0) or 0
        insider = (info.get("insider_percent_held", 0) or 0) * 100
        si_pct = (info.get("short_percent_of_float", 0) or 0) * 100

        for label, value in [
            ("Float", f"{float_shares/1e6:.1f}M" if float_shares > 1e6 else f"{float_shares:,.0f}"),
            ("Insider", f"{insider:.0f}%"),
            ("SI%", f"{si_pct:.1f}%"),
            ("Pre-Pump", f"{pp.get('confluence_count', 0)}/7 {pp.get('confidence', 'LOW')}"),
        ]:
            stat = self._make_card(label, value, "#cdd6f4")
            self.summary_layout.addWidget(stat, 4, col)
            col += 1

        self.summary_frame.setVisible(True)

    def _make_score_card(self, label, score):
        if score >= 70:
            color = "#a6e3a1"
        elif score >= 50:
            color = "#f9e2af"
        else:
            color = "#f38ba8"
        return self._make_card(label, f"{score:.0f}", color)

    @staticmethod
    def _make_card(label, value, color):
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background-color: #1e1e2e; border: 1px solid #45475a; "
            "border-radius: 4px; padding: 4px; margin: 2px; }"
        )
        card_layout = QVBoxLayout()
        card_layout.setSpacing(2)
        card_layout.setContentsMargins(6, 4, 6, 4)

        val_label = QLabel(str(value))
        val_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold; border: none;")
        val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(val_label)

        name_label = QLabel(label)
        name_label.setStyleSheet("color: #6c7086; font-size: 10px; border: none;")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(name_label)

        card.setLayout(card_layout)
        return card



# ═══════════════════════════════════════════════════════════════════
# TAB 6: ALERTS (rewired to Portfolio)
# ═══════════════════════════════════════════════════════════════════

class AlertsTab(QWidget):
    """Email alerts tied to My Portfolio (not Simulation)."""

    def __init__(self):
        super().__init__()
        self.monitor = None
        layout = QVBoxLayout()

        header = QLabel("Email Alerts  --  Portfolio Buy/Sell Notifications")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #f5c2e7; padding: 10px;")
        layout.addWidget(header)

        desc = QLabel(
            "Monitors YOUR PORTFOLIO positions during market hours. Sends email alerts when:\n"
            "  - Sell triggers hit (stop-loss, take-profit, trailing stop, max hold)\n"
            "  - Highly recommended buy signals found (HIGH confidence picks only)\n"
            "  - Daily portfolio summary at market close"
        )
        desc.setStyleSheet("color: #a6adc8; padding: 0 10px 5px 10px;")
        layout.addWidget(desc)

        # ── Config Summary ──────────────────────────────────
        self.config_frame = QFrame()
        self.config_frame.setStyleSheet(
            "QFrame { background-color: #181825; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 10px; }"
        )
        config_grid = QGridLayout()

        config_grid.addWidget(self._label("SMTP Server:"), 0, 0)
        self.smtp_input = QLineEdit()
        self.smtp_input.setPlaceholderText("smtp.gmail.com")
        config_grid.addWidget(self.smtp_input, 0, 1)

        config_grid.addWidget(self._label("Port:"), 0, 2)
        self.port_input = QLineEdit()
        self.port_input.setPlaceholderText("587")
        self.port_input.setMaximumWidth(80)
        config_grid.addWidget(self.port_input, 0, 3)

        config_grid.addWidget(self._label("Sender Email:"), 1, 0)
        self.sender_input = QLineEdit()
        self.sender_input.setPlaceholderText("you@gmail.com")
        config_grid.addWidget(self.sender_input, 1, 1)

        config_grid.addWidget(self._label("App Password:"), 1, 2)
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("xxxx xxxx xxxx xxxx")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        config_grid.addWidget(self.password_input, 1, 3)

        config_grid.addWidget(self._label("Recipient:"), 2, 0)
        self.recipient_input = QLineEdit()
        self.recipient_input.setPlaceholderText("you@gmail.com (can be same as sender)")
        config_grid.addWidget(self.recipient_input, 2, 1)

        config_grid.addWidget(self._label("Check (min):"), 2, 2)
        self.interval_input = QLineEdit()
        self.interval_input.setPlaceholderText("15")
        self.interval_input.setMaximumWidth(80)
        config_grid.addWidget(self.interval_input, 2, 3)

        self.config_frame.setLayout(config_grid)
        layout.addWidget(self.config_frame)

        # ── Button Row ──────────────────────────────────────
        btn_row = QHBoxLayout()

        self.save_btn = QPushButton("Save Config")
        self.save_btn.setStyleSheet(
            "QPushButton { background-color: #89b4fa; padding: 10px 20px; }"
            "QPushButton:hover { background-color: #74c7ec; }"
        )
        self.save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(self.save_btn)

        self.test_btn = QPushButton("Send Test Email")
        self.test_btn.setStyleSheet(
            "QPushButton { background-color: #f9e2af; color: #1e1e2e; padding: 10px 20px; font-weight: bold; }"
            "QPushButton:hover { background-color: #fab387; }"
        )
        self.test_btn.clicked.connect(self._send_test)
        btn_row.addWidget(self.test_btn)

        self.start_btn = QPushButton("Start Monitor")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #a6e3a1; color: #1e1e2e; "
            "padding: 10px 20px; font-weight: bold; }"
            "QPushButton:hover { background-color: #94e2d5; }"
        )
        self.start_btn.clicked.connect(self._toggle_monitor)
        btn_row.addWidget(self.start_btn)

        self.check_btn = QPushButton("Check Now")
        self.check_btn.setStyleSheet(
            "QPushButton { background-color: #cba6f7; color: #1e1e2e; padding: 10px 20px; font-weight: bold; }"
            "QPushButton:hover { background-color: #b4befe; }"
        )
        self.check_btn.clicked.connect(self._check_now)
        btn_row.addWidget(self.check_btn)

        self.status_label = QLabel("")
        btn_row.addWidget(self.status_label)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Status Cards ────────────────────────────────────
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet(
            "QFrame { background-color: #181825; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 10px; }"
        )
        self.status_grid = QGridLayout()
        self._status_labels = {}
        self._build_status_cards()
        self.status_frame.setLayout(self.status_grid)
        layout.addWidget(self.status_frame)

        # ── Alert Log ───────────────────────────────────────
        log_label = QLabel("Alert Log")
        log_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f5c2e7; padding: 8px 10px 2px 10px;")
        layout.addWidget(log_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "QTextEdit { background-color: #11111b; border: 1px solid #45475a; "
            "border-radius: 4px; padding: 8px; font-family: monospace; font-size: 12px; }"
        )
        layout.addWidget(self.log_output)

        # ── Alert History Table ─────────────────────────────
        hist_label = QLabel("Alert History")
        hist_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f5c2e7; padding: 8px 10px 2px 10px;")
        layout.addWidget(hist_label)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(3)
        self.history_table.setHorizontalHeaderLabels(["Time", "Type", "Detail"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setMaximumHeight(180)
        layout.addWidget(self.history_table)

        self.setLayout(layout)

        self._load_config_fields()
        self._refresh_status()

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #a6adc8; font-weight: bold;")
        return lbl

    def _build_status_cards(self):
        cards = [
            ("monitor", "Monitor", "STOPPED"),
            ("alerts_sent", "Total Alerts", "0"),
            ("buy_alerts", "Buy Alerts", "0"),
            ("sell_alerts", "Sell Alerts", "0"),
            ("last_check", "Last Check", "Never"),
            ("last_scan", "Last Scan", "Never"),
        ]
        for i, (key, title, default) in enumerate(cards):
            title_lbl = QLabel(title)
            title_lbl.setStyleSheet("color: #585b70; font-size: 10px;")
            val_lbl = QLabel(default)
            val_lbl.setStyleSheet("color: #cdd6f4; font-size: 16px; font-weight: bold;")
            self.status_grid.addWidget(title_lbl, 0, i)
            self.status_grid.addWidget(val_lbl, 1, i)
            self._status_labels[key] = val_lbl

    def _load_config_fields(self):
        from pennystock.config import (
            ALERT_EMAIL_SMTP_SERVER, ALERT_EMAIL_SMTP_PORT,
            ALERT_EMAIL_SENDER, ALERT_EMAIL_PASSWORD,
            ALERT_EMAIL_RECIPIENT, ALERT_PRICE_CHECK_MINUTES,
        )
        self.smtp_input.setText(ALERT_EMAIL_SMTP_SERVER)
        self.port_input.setText(str(ALERT_EMAIL_SMTP_PORT))
        self.sender_input.setText(ALERT_EMAIL_SENDER)
        self.password_input.setText(ALERT_EMAIL_PASSWORD)
        self.recipient_input.setText(ALERT_EMAIL_RECIPIENT)
        self.interval_input.setText(str(ALERT_PRICE_CHECK_MINUTES))

    def _save_config(self):
        import json
        config = {
            "ALERT_ENABLED": True,
            "ALERT_EMAIL_SMTP_SERVER": self.smtp_input.text().strip(),
            "ALERT_EMAIL_SMTP_PORT": int(self.port_input.text().strip() or "587"),
            "ALERT_EMAIL_SENDER": self.sender_input.text().strip(),
            "ALERT_EMAIL_PASSWORD": self.password_input.text().strip(),
            "ALERT_EMAIL_RECIPIENT": self.recipient_input.text().strip(),
            "ALERT_PRICE_CHECK_MINUTES": int(self.interval_input.text().strip() or "15"),
        }

        import pennystock.config as cfg
        for key, val in config.items():
            setattr(cfg, key, val)

        config_path = os.path.join(os.path.dirname(__file__), "..", "alert_config.json")
        config_path = os.path.normpath(config_path)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        self.status_label.setText("Config saved!")
        self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self._append_log("Email configuration saved to alert_config.json")

    def _send_test(self):
        self._apply_config_to_module()
        from pennystock.alerts.email_sender import send_test_email
        self._append_log("Sending test email...")
        if send_test_email():
            self.status_label.setText("Test email sent!")
            self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            self._append_log("Test email sent successfully!")
        else:
            self.status_label.setText("Send failed -- check log")
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
            self._append_log("Test email FAILED. Check SMTP settings and app password.")

    def _apply_config_to_module(self):
        import pennystock.config as cfg
        cfg.ALERT_ENABLED = True
        cfg.ALERT_EMAIL_SMTP_SERVER = self.smtp_input.text().strip()
        cfg.ALERT_EMAIL_SMTP_PORT = int(self.port_input.text().strip() or "587")
        cfg.ALERT_EMAIL_SENDER = self.sender_input.text().strip()
        cfg.ALERT_EMAIL_PASSWORD = self.password_input.text().strip()
        cfg.ALERT_EMAIL_RECIPIENT = self.recipient_input.text().strip()
        cfg.ALERT_PRICE_CHECK_MINUTES = int(self.interval_input.text().strip() or "15")

    def _toggle_monitor(self):
        if self.monitor and self.monitor.is_running:
            self.monitor.stop()
            self.start_btn.setText("Start Monitor")
            self.start_btn.setStyleSheet(
                "QPushButton { background-color: #a6e3a1; color: #1e1e2e; "
                "padding: 10px 20px; font-weight: bold; }"
            )
            self._status_labels["monitor"].setText("STOPPED")
            self._status_labels["monitor"].setStyleSheet("color: #f38ba8; font-size: 16px; font-weight: bold;")
            self._append_log("Monitor stopped")
        else:
            self._apply_config_to_module()
            sender = self.sender_input.text().strip()
            recipient = self.recipient_input.text().strip()
            password = self.password_input.text().strip()
            if not sender or not recipient or not password:
                self.status_label.setText("Configure email first!")
                self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
                return
            from pennystock.alerts.monitor import AlertMonitor
            self.monitor = AlertMonitor()
            self.monitor.start(log_callback=self._append_log)
            self.start_btn.setText("Stop Monitor")
            self.start_btn.setStyleSheet(
                "QPushButton { background-color: #f38ba8; color: #1e1e2e; "
                "padding: 10px 20px; font-weight: bold; }"
            )
            self._status_labels["monitor"].setText("RUNNING")
            self._status_labels["monitor"].setStyleSheet("color: #a6e3a1; font-size: 16px; font-weight: bold;")
            self._append_log("Monitor started -- checking every "
                           f"{self.interval_input.text().strip() or '15'} min")

            if not hasattr(self, '_status_timer'):
                self._status_timer = QTimer()
                self._status_timer.timeout.connect(self._refresh_status)
                self._status_timer.start(30000)

    def _check_now(self):
        self._apply_config_to_module()
        from pennystock.alerts.monitor import AlertMonitor
        monitor = self.monitor or AlertMonitor()
        monitor._log_callback = self._append_log
        self._append_log("Running manual check...")
        try:
            monitor.run_once()
            self._append_log("Manual check complete")
        except Exception as e:
            self._append_log(f"Check failed: {e}")
        self._refresh_status()

    def _refresh_status(self):
        from pennystock.alerts.monitor import AlertMonitor
        monitor = self.monitor or AlertMonitor()
        status = monitor.get_status()

        running = self.monitor.is_running if self.monitor else False
        self._status_labels["monitor"].setText("RUNNING" if running else "STOPPED")
        self._status_labels["monitor"].setStyleSheet(
            f"color: {'#a6e3a1' if running else '#f38ba8'}; font-size: 16px; font-weight: bold;"
        )
        self._status_labels["alerts_sent"].setText(str(status["alerts_sent"]))
        self._status_labels["buy_alerts"].setText(str(status["buy_alerts"]))
        self._status_labels["sell_alerts"].setText(str(status["sell_alerts"]))
        self._status_labels["last_check"].setText(
            status["last_price_check"][:16] if status["last_price_check"] else "Never"
        )
        self._status_labels["last_scan"].setText(
            status["last_scan"][:16] if status["last_scan"] else "Never"
        )

        history = status.get("history", [])
        self.history_table.setRowCount(len(history))
        for row, h in enumerate(reversed(history)):
            self.history_table.setItem(row, 0, QTableWidgetItem(h["time"][:16]))
            type_item = QTableWidgetItem(h["type"])
            if h["type"] == "BUY":
                type_item.setForeground(QColor("#a6e3a1"))
            elif h["type"] == "SELL":
                type_item.setForeground(QColor("#f38ba8"))
            else:
                type_item.setForeground(QColor("#f9e2af"))
            self.history_table.setItem(row, 1, type_item)
            self.history_table.setItem(row, 2, QTableWidgetItem(h["detail"]))

    def _append_log(self, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {text}")


# ═══════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════

class PennyStockGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Stock Analyzer v{ALGORITHM_VERSION}")
        self.setMinimumSize(1100, 800)

        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        tabs = QTabWidget()
        tabs.addTab(AlgorithmBuilderTab(), "  Algorithm Builder  ")
        tabs.addTab(StockPickerTab(), "  Stock Picker  ")
        tabs.addTab(SimulationTab(), "  Simulation  ")
        tabs.addTab(MyPortfolioTab(), "  My Portfolio  ")
        tabs.addTab(AnalyzeStockTab(), "  Analyze Stock  ")
        tabs.addTab(AlertsTab(), "  Alerts  ")
        main_layout.addWidget(tabs)

        version_bar = QLabel(
            f"  Stock Analyzer v{ALGORITHM_VERSION}  |  $2-$5  |  "
            f"Market & Sector Sentiment Integrated  |  12 kill filters  |  7 pre-pump signals"
        )
        version_bar.setStyleSheet(
            "background-color: #11111b; color: #585b70; font-size: 11px; "
            "padding: 4px 8px; border-top: 1px solid #313244;"
        )
        version_bar.setFixedHeight(24)
        main_layout.addWidget(version_bar)

        central.setLayout(main_layout)
        self.setCentralWidget(central)


def launch_gui():
    """Launch the PyQt6 GUI application."""
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = PennyStockGUI()
    window.show()
    sys.exit(app.exec())
