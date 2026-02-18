"""
PyQt6 GUI for the Penny Stock Analyzer.

Two tabs:
  Tab 1 - Build Algorithm: Learn from recent winners vs losers
  Tab 2 - Pick Stocks: Apply algorithm to find today's top 5
"""

import sys
import threading

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QProgressBar, QLabel, QHeaderView,
)
from PyQt6.QtGui import QFont, QColor

from pennystock import __version__
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
        result = self.func(*self.args, progress_callback=self.progress.emit, **self.kwargs)
        self.finished.emit(result)


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


# ── Main Window ─────────────────────────────────────────────────────
class PennyStockGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Penny Stock Analyzer v{__version__}")
        self.setMinimumSize(950, 700)

        tabs = QTabWidget()
        tabs.addTab(BuildAlgorithmTab(), "  Build Algorithm  ")
        tabs.addTab(PickStocksTab(), "  Pick Stocks  ")
        self.setCentralWidget(tabs)


def launch_gui():
    """Launch the PyQt6 GUI application."""
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = PennyStockGUI()
    window.show()
    sys.exit(app.exec())
