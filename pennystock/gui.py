"""
PyQt6 GUI for the Penny Stock Analyzer.

Five tabs:
  Tab 1 - Build Algorithm: Learn from recent winners vs losers
  Tab 2 - Pick Stocks: Apply algorithm to find today's top 5
  Tab 3 - Analyze Stock: Comprehensive deep dive on a single ticker
  Tab 4 - Backtest: Run the algorithm on a past date and check results
  Tab 5 - Backtest Algorithm: Full 3-year optimization of sell strategy + weights
"""

import sys
import threading
from datetime import date

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QProgressBar, QLabel, QHeaderView, QLineEdit, QScrollArea, QFrame,
    QGridLayout, QSplitter, QSizePolicy, QDateEdit, QComboBox,
)
from PyQt6.QtGui import QFont, QColor

from pennystock import __version__
from pennystock.config import ALGORITHM_VERSION
from pennystock.algorithm import build_algorithm, pick_stocks, load_algorithm


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


# ── Tab 1: Build Algorithm ──────────────────────────────────────────
class BuildAlgorithmTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        layout = QVBoxLayout()

        # Header
        header = QLabel("Build Algorithm from Recent Winners")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #f5c2e7; padding: 10px;")
        layout.addWidget(header)

        desc = QLabel(
            "Analyzes all penny stocks ($0.05-$1.00) over the past 3 months.\n"
            "Finds stocks that gained steadily over 2-4+ weeks (not pump & dumps).\n"
            "Compares winners vs losers on technical, sentiment, and fundamental factors.\n"
            "Builds ONE unified algorithm with kill filters + weighted scoring."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 0 10px 10px 10px;")
        layout.addWidget(desc)

        # Button + Progress
        btn_row = QHBoxLayout()
        self.build_btn = QPushButton("Build Algorithm")
        self.build_btn.clicked.connect(self._start_build)
        btn_row.addWidget(self.build_btn)

        self.status_label = QLabel("")
        btn_row.addWidget(self.status_label)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Log output
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def _start_build(self):
        self.build_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log.clear()
        self._append_log("Starting algorithm build...")

        self.worker = Worker(build_algorithm)
        self.worker.progress.connect(self._append_log)
        self.worker.finished.connect(self._build_done)
        self.worker.start()

    def _append_log(self, msg):
        self.log.append(msg)
        # Auto-scroll to bottom
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _build_done(self, result):
        self.progress_bar.setVisible(False)
        self.build_btn.setEnabled(True)

        if result:
            n_factors = len(result.get("factors", []))
            n_winners = result.get("training_summary", {}).get("winners", 0)
            self.status_label.setText(
                f"Done! {n_winners} winners, {n_factors} factors learned."
            )
            self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            self._append_log("\nAlgorithm saved to algorithm.json")
        else:
            self.status_label.setText("Failed - check log for details.")
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")


# ── Tab 2: Pick Stocks ─────────────────────────────────────────────
class PickStocksTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        layout = QVBoxLayout()

        header = QLabel("Pick Top Penny Stocks")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #f5c2e7; padding: 10px;")
        layout.addWidget(header)

        # Check if algorithm exists
        algo = load_algorithm()
        if algo:
            algo_info = QLabel(
                f"Algorithm loaded (built {algo.get('built_date', 'unknown')}, "
                f"{len(algo.get('factors', []))} factors)"
            )
            algo_info.setStyleSheet("color: #a6e3a1; padding: 0 10px;")
        else:
            algo_info = QLabel("No algorithm found. Build one first using Tab 1.")
            algo_info.setStyleSheet("color: #fab387; padding: 0 10px;")
        layout.addWidget(algo_info)
        self.algo_info = algo_info

        # Button
        btn_row = QHBoxLayout()
        self.pick_btn = QPushButton("Find Top 5 Stocks")
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
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Rank", "Ticker", "Price", "Score",
            "Setup", "Technical", "Fundamental", "Catalyst", "Key Info"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setVisible(False)
        layout.addWidget(self.table)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def _start_pick(self):
        # Re-check algorithm
        algo = load_algorithm()
        if not algo:
            self.algo_info.setText("No algorithm found. Build one first using Tab 1.")
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
        self._append_log("Starting stock picking...")

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

        # Populate table
        self.table.setRowCount(len(picks))
        for row, pick in enumerate(picks):
            ss = pick.get("sub_scores", {})
            ki = pick.get("key_indicators", {})

            # Build key info string
            float_val = ki.get("float_shares", 0) or 0
            float_str = f"{float_val/1e6:.1f}M" if float_val > 0 else "N/A"
            insider_val = (ki.get("insider_pct") or 0) * 100
            rsi_val = ki.get("rsi")
            rsi_str = f"{rsi_val:.0f}" if rsi_val is not None else "N/A"
            key_info = f"Float:{float_str} Ins:{insider_val:.0f}% RSI:{rsi_str}"

            items = [
                (f"#{row + 1}", None),
                (pick["ticker"], None),
                (f"${pick['price']:.2f}", None),
                (f"{pick['final_score']:.1f}", self._score_color(pick["final_score"])),
                (f"{ss.get('setup', 0):.0f}", self._score_color(ss.get("setup", 0))),
                (f"{ss.get('technical', 0):.0f}", self._score_color(ss.get("technical", 0))),
                (f"{ss.get('fundamental', 0):.0f}", self._score_color(ss.get("fundamental", 0))),
                (f"{ss.get('catalyst', 0):.0f}", self._score_color(ss.get("catalyst", 0))),
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
            return QColor("#a6e3a1")   # Green
        elif score >= 50:
            return QColor("#f9e2af")   # Yellow
        else:
            return QColor("#f38ba8")   # Red


# ── Tab 3: Analyze Stock (Deep Dive) ──────────────────────────────
class AnalyzeStockTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_result = None
        layout = QVBoxLayout()

        # Header
        header = QLabel("Analyze Stock  —  Deep Dive")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #f5c2e7; padding: 10px;")
        layout.addWidget(header)

        desc = QLabel(
            "Enter a ticker for comprehensive analysis: price action, fundamentals, "
            "pre-pump signals, technicals, sentiment, news, and full algorithm scoring."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 0 10px 5px 10px;")
        layout.addWidget(desc)

        # Ticker input row
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

        # Main content area: summary cards on top, full report below
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            "QFrame { background-color: #181825; border: 1px solid #45475a; border-radius: 6px; padding: 10px; }"
        )
        self.summary_frame.setVisible(False)
        self.summary_layout = QGridLayout()
        self.summary_frame.setLayout(self.summary_layout)
        layout.addWidget(self.summary_frame)

        # Full report text area
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
            self.status_label.setText(f"{ticker}: {score:.1f} pts — HIGH confidence")
            self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        elif confidence == "MEDIUM":
            self.status_label.setText(f"{ticker}: {score:.1f} pts — MEDIUM confidence")
            self.status_label.setStyleSheet("color: #f9e2af; font-weight: bold;")
        else:
            self.status_label.setText(f"{ticker}: {score:.1f} pts — LOW confidence")
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")

        # Build summary cards
        self._build_summary(result)

    def _build_summary(self, result):
        """Build the summary card grid at the top."""
        # Clear existing widgets
        while self.summary_layout.count():
            child = self.summary_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        ss = result.get("sub_scores", {})
        pp = result.get("pre_pump", {})
        info = result.get("info", {})
        movements = result.get("price_movements", {})
        gate = result.get("quality_gate", {})

        # Row 0: Price and score headline
        price = result.get("price", 0)
        score = result.get("final_score", 0)
        confidence = result.get("confidence", "LOW")
        company = result.get("company", "")

        headline = QLabel(f"  {result.get('ticker', '')}  —  ${price:.4f}  —  Score: {score:.1f} ({confidence})")
        headline.setStyleSheet(
            f"font-size: 16px; font-weight: bold; padding: 5px; "
            f"color: {'#a6e3a1' if confidence == 'HIGH' else '#f9e2af' if confidence == 'MEDIUM' else '#f38ba8'};"
        )
        self.summary_layout.addWidget(headline, 0, 0, 1, 6)

        company_label = QLabel(f"  {company}")
        company_label.setStyleSheet("color: #a6adc8; font-size: 12px; padding-left: 5px;")
        self.summary_layout.addWidget(company_label, 1, 0, 1, 6)

        # Row 2: Price movements
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

        # Row 3: Sub-score cards
        col = 0
        for name, key in [("Setup", "setup"), ("Technical", "technical"), ("Pre-Pump", "pre_pump"),
                           ("Fundamental", "fundamental"), ("Catalyst", "catalyst")]:
            val = ss.get(key, 0)
            card = self._make_score_card(name, val)
            self.summary_layout.addWidget(card, 3, col)
            col += 1

        # Add penalty card
        penalty = gate.get("total_penalty", 0)
        if penalty > 0:
            pen_card = self._make_card("Penalties", f"-{penalty}pts", "#f38ba8")
            self.summary_layout.addWidget(pen_card, 3, col)

        # Row 4: Key stats
        col = 0
        float_shares = info.get("float_shares", 0) or 0
        insider = (info.get("insider_percent_held", 0) or 0) * 100
        si_pct = (info.get("short_percent_of_float", 0) or 0) * 100

        for label, value in [
            ("Float", f"{float_shares/1e6:.1f}M" if float_shares > 1e6 else f"{float_shares:,.0f}"),
            ("Insider", f"{insider:.0f}%"),
            ("SI%", f"{si_pct:.1f}%"),
            ("Pre-Pump", f"{pp.get('confluence_count', 0)}/7 {pp.get('confidence', 'LOW')}"),
            ("52w Pos", f"{(movements.get('1y_low', 0) or 0):.2f} - {(movements.get('1y_high', 0) or 0):.2f}"),
        ]:
            stat = self._make_card(label, value, "#cdd6f4")
            self.summary_layout.addWidget(stat, 4, col)
            col += 1

        self.summary_frame.setVisible(True)

    def _make_score_card(self, label, score):
        """Create a colored score card widget."""
        if score >= 70:
            color = "#a6e3a1"
        elif score >= 50:
            color = "#f9e2af"
        else:
            color = "#f38ba8"
        return self._make_card(label, f"{score:.0f}", color)

    @staticmethod
    def _make_card(label, value, color):
        """Create a small info card widget."""
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


# ── Tab 4: Backtest ────────────────────────────────────────────────
class BacktestTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_result = None
        layout = QVBoxLayout()

        # Header
        header = QLabel("Backtest  —  Historical Validation")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #f5c2e7; padding: 10px;")
        layout.addWidget(header)

        desc = QLabel(
            "Run the algorithm on a past date and see what would have happened.\n"
            "Picks are scored using historical price data; forward returns show actual results.\n"
            "Note: fundamentals (insider%, float, SI) use current data as proxy."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 0 10px 5px 10px;")
        layout.addWidget(desc)

        # Input row: date picker + run button
        input_row = QHBoxLayout()

        date_label = QLabel("Target Date:")
        date_label.setStyleSheet("font-size: 14px; font-weight: bold; padding-left: 10px;")
        input_row.addWidget(date_label)

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setDate(QDate(2025, 8, 1))
        self.date_edit.setMaximumDate(QDate.currentDate().addDays(-15))
        self.date_edit.setMinimumDate(QDate(2024, 1, 1))
        self.date_edit.setStyleSheet(
            "QDateEdit { background-color: #11111b; color: #cdd6f4; "
            "border: 1px solid #45475a; border-radius: 4px; padding: 8px 12px; "
            "font-size: 14px; font-weight: bold; }"
            "QDateEdit:focus { border: 2px solid #89b4fa; }"
        )
        self.date_edit.setMaximumWidth(180)
        input_row.addWidget(self.date_edit)

        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.clicked.connect(self._start_backtest)
        input_row.addWidget(self.run_btn)

        self.status_label = QLabel("")
        input_row.addWidget(self.status_label)
        input_row.addStretch()
        layout.addLayout(input_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Results table (columns built dynamically based on hold_days)
        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setVisible(False)
        layout.addWidget(self.table)

        # Summary frame
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            "QFrame { background-color: #181825; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 10px; }"
        )
        self.summary_frame.setVisible(False)
        self.summary_layout = QVBoxLayout()
        self.summary_frame.setLayout(self.summary_layout)
        layout.addWidget(self.summary_frame)

        # Full log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def _start_backtest(self):
        target_date = self.date_edit.date().toString("yyyy-MM-dd")

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.table.setVisible(False)
        self.summary_frame.setVisible(False)
        self.log.clear()
        self.status_label.setText(f"Running backtest for {target_date}...")
        self.status_label.setStyleSheet("color: #89b4fa;")

        from pennystock.backtest.historical import run_historical_backtest
        self.worker = Worker(run_historical_backtest, target_date)
        self.worker.progress.connect(self._append_log)
        self.worker.finished.connect(self._backtest_done)
        self.worker.start()

    def _append_log(self, msg):
        self.log.append(msg)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _backtest_done(self, result):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.current_result = result

        if not result or not result.get("picks"):
            self.status_label.setText("Backtest failed or no picks found")
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
            return

        picks = result["picks"]
        target = result["target_date"]
        summary = result.get("summary", {})
        hold_days = result.get("hold_days", [3, 5, 7, 10, 14])
        hold_keys = [f"{d}d" for d in hold_days]

        # Status: show best horizon win rate
        best_wr_key = max(hold_keys, key=lambda k: summary.get(f"top_picks_{k}", {}).get("win_rate", 0))
        best_wr = summary.get(f"top_picks_{best_wr_key}", {}).get("win_rate", 0)
        self.status_label.setText(
            f"{target}: {len(picks)} picks | Best: {best_wr:.0f}% win @ {best_wr_key}"
        )
        color = "#a6e3a1" if best_wr > 50 else "#f9e2af" if best_wr >= 40 else "#f38ba8"
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        # Build dynamic table columns
        base_headers = ["Rank", "Ticker", "Entry $", "Score", "Setup", "Tech", "PrePump"]
        ret_headers = [f"{k} Ret" for k in hold_keys]
        peak_headers = [f"{k} Peak" for k in hold_keys]
        all_headers = base_headers + ret_headers + peak_headers

        self.table.setColumnCount(len(all_headers))
        self.table.setHorizontalHeaderLabels(all_headers)
        self.table.setRowCount(len(picks))

        for row, p in enumerate(picks):
            ss = p.get("sub_scores", {})
            fr = p.get("forward_returns", {})

            items = [
                (f"#{row+1}", None),
                (p["ticker"], None),
                (f"${p['entry_price']:.4f}", None),
                (f"{p['score']:.1f}", self._score_color(p["score"])),
                (f"{ss.get('setup', 0):.0f}", None),
                (f"{ss.get('technical', 0):.0f}", None),
                (f"{ss.get('pre_pump', 0):.0f}", None),
            ]
            # Return columns
            for key in hold_keys:
                r = fr.get(key, {}).get("return_pct")
                sl = fr.get(key, {}).get("stop_loss_triggered", False)
                if r is not None:
                    text = f"{r:+.1f}%{'*' if sl else ''}"
                    c = QColor("#a6e3a1") if r > 0 else QColor("#f38ba8")
                    items.append((text, c))
                else:
                    items.append(("N/A", None))
            # Peak columns
            for key in hold_keys:
                pk = fr.get(key, {}).get("peak_return_pct")
                if pk is not None:
                    text = f"{pk:+.1f}%"
                    c = QColor("#89b4fa") if pk > 0 else QColor("#f38ba8")
                    items.append((text, c))
                else:
                    items.append(("N/A", None))

            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(color)
                self.table.setItem(row, col, item)

        self.table.setVisible(True)

        # Build summary
        self._build_summary(summary, hold_keys)

    def _build_summary(self, summary, hold_keys=None):
        """Build summary statistics display with stop-loss and peak data."""
        hold_keys = hold_keys or ["3d", "5d", "7d", "10d", "14d"]

        while self.summary_layout.count():
            child = self.summary_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for group, label in [("top_picks", "Top Picks"), ("all_scored", "All Scored")]:
            # Group header
            hdr = QLabel(f"<b style='color:#f5c2e7'>{label}</b>")
            hdr.setTextFormat(Qt.TextFormat.RichText)
            hdr.setStyleSheet("font-size: 13px; padding: 4px 5px 0 5px; border: none;")
            self.summary_layout.addWidget(hdr)

            for horizon in hold_keys:
                key = f"{group}_{horizon}"
                s = summary.get(key, {})
                if s.get("count", 0) == 0:
                    continue

                wr = s["win_rate"]
                wr_color = "#a6e3a1" if wr > 50 else "#f9e2af" if wr >= 40 else "#f38ba8"
                sl_wr = s.get("sl_win_rate", 0)
                sl_color = "#a6e3a1" if sl_wr > 50 else "#f9e2af" if sl_wr >= 40 else "#f38ba8"
                avg_peak = s.get("avg_peak", 0)
                sl_avg = s.get("sl_avg_return", 0)
                sl_count = s.get("sl_triggered_count", 0)

                text = (
                    f"  <b>{horizon}</b>: "
                    f"<span style='color:{wr_color}'>{wr:.0f}%W</span> "
                    f"Avg:{s['avg_return']:+.1f}% "
                    f"Peak:{avg_peak:+.1f}% "
                    f"| SL: <span style='color:{sl_color}'>{sl_wr:.0f}%W</span> "
                    f"Avg:{sl_avg:+.1f}% "
                    f"({sl_count} hit) "
                    f"| n={s['count']}"
                )
                lbl = QLabel(text)
                lbl.setTextFormat(Qt.TextFormat.RichText)
                lbl.setStyleSheet("font-size: 11px; padding: 1px 5px 1px 15px; border: none;")
                self.summary_layout.addWidget(lbl)

        # Legend
        legend = QLabel(
            "<span style='color:#6c7086'>W=Win Rate | Peak=Avg best return during hold | "
            "SL=With stop-loss | * in table = stop-loss triggered</span>"
        )
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setStyleSheet("font-size: 10px; padding: 4px 5px; border: none;")
        self.summary_layout.addWidget(legend)

        self.summary_frame.setVisible(True)

    @staticmethod
    def _score_color(score):
        if score >= 65:
            return QColor("#a6e3a1")
        elif score >= 50:
            return QColor("#f9e2af")
        else:
            return QColor("#f38ba8")


# ── Tab 5: Backtest Algorithm (Optimizer) ─────────────────────────
class OptimizeAlgorithmTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_result = None
        layout = QVBoxLayout()

        # Header
        header = QLabel("Backtest Algorithm  —  Full Optimization")
        header.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #f5c2e7; padding: 10px;"
        )
        layout.addWidget(header)

        desc = QLabel(
            "Runs the algorithm on 1st & 15th of every month for ~3 years (~70 dates).\n"
            "Tests thousands of sell strategies (hold period, stop-loss, take-profit, trailing stop).\n"
            "Finds optimal scoring weights and sell parameters to maximize returns.\n"
            "This takes 15-30 minutes. Results are saved and can be applied to the live algorithm."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 0 10px 5px 10px;")
        layout.addWidget(desc)

        # Button row
        btn_row = QHBoxLayout()

        self.run_btn = QPushButton("Run Full Optimization")
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #f5c2e7; color: #1e1e2e; "
            "padding: 12px 30px; font-size: 14px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #f38ba8; }"
            "QPushButton:disabled { background-color: #585b70; color: #6c7086; }"
        )
        self.run_btn.clicked.connect(self._start_optimization)
        btn_row.addWidget(self.run_btn)

        self.apply_btn = QPushButton("Apply Recommendations")
        self.apply_btn.setStyleSheet(
            "QPushButton { background-color: #a6e3a1; color: #1e1e2e; "
            "padding: 12px 20px; font-size: 13px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #94e2d5; }"
            "QPushButton:disabled { background-color: #585b70; color: #6c7086; }"
        )
        self.apply_btn.clicked.connect(self._apply_recommendations)
        self.apply_btn.setVisible(False)
        btn_row.addWidget(self.apply_btn)

        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.setStyleSheet(
            "QPushButton { background-color: #fab387; color: #1e1e2e; "
            "padding: 12px 20px; font-size: 13px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #f38ba8; }"
        )
        self.reset_btn.clicked.connect(self._reset_config)
        self.reset_btn.setVisible(False)
        btn_row.addWidget(self.reset_btn)

        self.status_label = QLabel("")
        btn_row.addWidget(self.status_label)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Summary frame for recommendations
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            "QFrame { background-color: #181825; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 10px; }"
        )
        self.summary_frame.setVisible(False)
        self.summary_layout = QVBoxLayout()
        self.summary_frame.setLayout(self.summary_layout)
        layout.addWidget(self.summary_frame)

        # Top strategies table
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "#", "Top N", "Hold", "Stop-Loss", "Take-Profit",
            "Trail", "Win Rate", "Avg Ret", "Med Ret", "Trades",
        ])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setVisible(False)
        layout.addWidget(self.table)

        # Full log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.setLayout(layout)

        # Check if optimized_config.json exists
        self._check_optimized_config()

    def _check_optimized_config(self):
        """Show reset button if optimized config exists."""
        import os
        opt_path = os.path.join(
            os.path.dirname(__file__), "..", "optimized_config.json"
        )
        if os.path.exists(os.path.normpath(opt_path)):
            self.reset_btn.setVisible(True)
            self.status_label.setText("Optimized config active")
            self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")

    def _start_optimization(self):
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.apply_btn.setVisible(False)
        self.summary_frame.setVisible(False)
        self.table.setVisible(False)
        self.log.clear()
        self.status_label.setText("Running full optimization...")
        self.status_label.setStyleSheet("color: #89b4fa;")

        from pennystock.backtest.optimizer import run_algorithm_optimization
        self.worker = Worker(run_algorithm_optimization)
        self.worker.progress.connect(self._append_log)
        self.worker.finished.connect(self._optimization_done)
        self.worker.start()

    def _append_log(self, msg):
        self.log.append(msg)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _optimization_done(self, result):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.current_result = result

        if not result or "error" in result:
            self.status_label.setText("Optimization failed — check log")
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
            return

        recs = result.get("recommendations", {})
        total = result.get("total_picks", 0)
        dates = result.get("dates_tested", 0)

        sell = recs.get("sell_strategy", {})
        scoring = recs.get("scoring", {})
        sell_wr = sell.get("expected_win_rate", 0)
        sell_avg = sell.get("expected_avg_return", 0)

        self.status_label.setText(
            f"Done! {dates} dates, {total} picks | "
            f"Best: {sell_wr:.0f}% win, {sell_avg:+.1f}% avg"
        )
        color = "#a6e3a1" if sell_wr > 50 else "#f9e2af"
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        # Show Apply button
        self.apply_btn.setVisible(True)

        # Build summary
        self._build_summary(recs)

        # Build top strategies table
        self._build_strategies_table(result)

    def _build_summary(self, recs):
        """Build recommendation summary cards."""
        while self.summary_layout.count():
            child = self.summary_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Sell strategy
        sell = recs.get("sell_strategy", {})
        if sell:
            hdr = QLabel("<b style='color:#f5c2e7; font-size:14px;'>"
                         "Recommended Sell Strategy</b>")
            hdr.setTextFormat(Qt.TextFormat.RichText)
            hdr.setStyleSheet("border: none; padding: 4px 5px;")
            self.summary_layout.addWidget(hdr)

            hold = sell.get("SELL_MAX_HOLD_DAYS", "?")
            sl = sell.get("SELL_STOP_LOSS_PCT", 0)
            tp = sell.get("SELL_TAKE_PROFIT_PCT", 0)
            ta = sell.get("SELL_TRAILING_STOP_ACTIVATE", 0)
            td = sell.get("SELL_TRAILING_STOP_DISTANCE", 0)
            top_n = sell.get("STAGE2_RETURN_TOP_N", 5)
            wr = sell.get("expected_win_rate", 0)
            ar = sell.get("expected_avg_return", 0)

            sl_str = f"{sl}%" if sl != 0 else "disabled"
            tp_str = f"+{tp}%" if tp != 0 else "disabled"
            trail_str = (f"activate at +{ta}%, trail {td}% from peak"
                         if ta > 0 else "disabled")
            wr_color = "#a6e3a1" if wr > 50 else "#f9e2af" if wr >= 40 else "#f38ba8"

            text = (
                f"  Hold: <b>{hold}d</b> | "
                f"Stop-Loss: <b>{sl_str}</b> | "
                f"Take-Profit: <b>{tp_str}</b> | "
                f"Trailing: <b>{trail_str}</b><br>"
                f"  Top N: <b>{top_n}</b> | "
                f"Expected: <span style='color:{wr_color}'><b>{wr:.0f}% win, "
                f"{ar:+.1f}% avg return</b></span>"
            )
            lbl = QLabel(text)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("font-size: 12px; padding: 2px 15px; border: none;")
            self.summary_layout.addWidget(lbl)

        # Scoring weights
        scoring = recs.get("scoring", {})
        if scoring:
            hdr2 = QLabel("<b style='color:#f5c2e7; font-size:14px;'>"
                          "Recommended Scoring Weights</b>")
            hdr2.setTextFormat(Qt.TextFormat.RichText)
            hdr2.setStyleSheet("border: none; padding: 8px 5px 4px 5px;")
            self.summary_layout.addWidget(hdr2)

            from pennystock.config import WEIGHTS as CUR_WEIGHTS, MIN_RECOMMENDATION_SCORE
            new_w = scoring.get("weights", {})
            new_min = scoring.get("MIN_RECOMMENDATION_SCORE", 40)
            wr2 = scoring.get("expected_win_rate", 0)
            ar2 = scoring.get("expected_avg_return", 0)

            rows = ""
            for k in ["setup", "technical", "pre_pump", "fundamental", "catalyst"]:
                cur = CUR_WEIGHTS.get(k, 0) * 100
                rec = new_w.get(k, 0) * 100
                delta = rec - cur
                d_color = "#a6e3a1" if delta > 0 else "#f38ba8" if delta < 0 else "#6c7086"
                d_str = f"{delta:+.0f}%" if abs(delta) > 0.5 else "="
                rows += (
                    f"  {k}: {cur:.0f}% -> "
                    f"<span style='color:{d_color}'><b>{rec:.0f}%</b></span> "
                    f"({d_str}) | "
                )

            text2 = (
                f"{rows}<br>"
                f"  Min Score: {MIN_RECOMMENDATION_SCORE} -> <b>{new_min}</b> | "
                f"Expected: <b>{wr2:.0f}% win, {ar2:+.1f}% avg</b>"
            )
            lbl2 = QLabel(text2)
            lbl2.setTextFormat(Qt.TextFormat.RichText)
            lbl2.setStyleSheet("font-size: 11px; padding: 2px 15px; border: none;")
            lbl2.setWordWrap(True)
            self.summary_layout.addWidget(lbl2)

        # Correlations
        corrs = recs.get("correlations", {})
        if corrs:
            hdr3 = QLabel("<b style='color:#f5c2e7; font-size:14px;'>"
                          "Factor Predictiveness</b>")
            hdr3.setTextFormat(Qt.TextFormat.RichText)
            hdr3.setStyleSheet("border: none; padding: 8px 5px 4px 5px;")
            self.summary_layout.addWidget(hdr3)

            corr_text = ""
            for k, v in sorted(corrs.items(),
                                key=lambda x: abs(x[1]), reverse=True):
                bar_len = int(abs(v) * 30)
                color = "#a6e3a1" if v > 0 else "#f38ba8"
                bar = f"<span style='color:{color}'>{'|' * bar_len}</span>"
                corr_text += f"  {k}: r={v:+.4f} {bar}<br>"

            lbl3 = QLabel(corr_text)
            lbl3.setTextFormat(Qt.TextFormat.RichText)
            lbl3.setStyleSheet("font-size: 11px; padding: 2px 15px; border: none; "
                                "font-family: 'Consolas', monospace;")
            self.summary_layout.addWidget(lbl3)

        self.summary_frame.setVisible(True)

    def _build_strategies_table(self, result):
        """Populate the top 20 sell strategies table."""
        top_20 = result.get("sell_optimization", {}).get("top_20", [])
        if not top_20:
            return

        self.table.setRowCount(len(top_20))
        for row, r in enumerate(top_20):
            trail_str = (f"{r['trail_activate']}/{r['trail_distance']}"
                         if r['trail_activate'] > 0 else "off")
            sl_str = f"{r['stop_loss']}%" if r['stop_loss'] != 0 else "off"
            tp_str = f"+{r['take_profit']}%" if r['take_profit'] != 0 else "off"

            items = [
                (f"#{row + 1}", None),
                (str(r["top_n"]), None),
                (f"{r['hold_days']}d", None),
                (sl_str, None),
                (tp_str, None),
                (trail_str, None),
                (f"{r['win_rate']:.0f}%",
                 QColor("#a6e3a1") if r["win_rate"] > 50
                 else QColor("#f38ba8")),
                (f"{r['avg_return']:+.1f}%",
                 QColor("#a6e3a1") if r["avg_return"] > 0
                 else QColor("#f38ba8")),
                (f"{r['median_return']:+.1f}%",
                 QColor("#a6e3a1") if r["median_return"] > 0
                 else QColor("#f38ba8")),
                (str(r["total_trades"]), None),
            ]

            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(color)
                self.table.setItem(row, col, item)

        self.table.setVisible(True)

    def _apply_recommendations(self):
        """Save optimized config to JSON file."""
        if not self.current_result:
            return

        recs = self.current_result.get("recommendations", {})
        if not recs:
            return

        from pennystock.backtest.optimizer import apply_optimized_config
        path = apply_optimized_config(recs)

        self.status_label.setText(f"Applied! Saved to {path}")
        self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self.apply_btn.setVisible(False)
        self.reset_btn.setVisible(True)

    def _reset_config(self):
        """Delete optimized_config.json to revert to defaults."""
        import os
        opt_path = os.path.join(
            os.path.dirname(__file__), "..", "optimized_config.json"
        )
        opt_path = os.path.normpath(opt_path)
        if os.path.exists(opt_path):
            os.remove(opt_path)
            self.status_label.setText("Reset to defaults. Restart to take effect.")
            self.status_label.setStyleSheet("color: #fab387; font-weight: bold;")
            self.reset_btn.setVisible(False)


# ── Main Window ─────────────────────────────────────────────────────
class PennyStockGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Penny Stock Analyzer v{ALGORITHM_VERSION}")
        self.setMinimumSize(1100, 800)

        # Central widget with main layout
        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(BuildAlgorithmTab(), "  Build Algorithm  ")
        tabs.addTab(PickStocksTab(), "  Pick Stocks  ")
        tabs.addTab(AnalyzeStockTab(), "  Analyze Stock  ")
        tabs.addTab(BacktestTab(), "  Backtest  ")
        tabs.addTab(OptimizeAlgorithmTab(), "  Backtest Algorithm  ")
        main_layout.addWidget(tabs)

        # Version bar at the bottom
        version_bar = QLabel(
            f"  Penny Stock Analyzer v{ALGORITHM_VERSION}  |  "
            f"Weights: pre_pump {35}% setup {25}% tech {20}% fund {10}% cat {10}%  |  "
            f"12 kill filters  |  7 pre-pump signals"
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
