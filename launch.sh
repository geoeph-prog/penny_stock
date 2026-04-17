#!/bin/bash
echo "============================================"
echo "  Penny Stock Analyzer - Starting..."
echo "============================================"
echo

cd "$(dirname "$0")"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install Python 3.9+"
    exit 1
fi

VENV_DIR=".venv"

# Set up venv on first run
if [ ! -d "$VENV_DIR" ]; then
    echo "First run — setting up virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install multitasking==0.0.11
    pip install -r requirements.txt
    echo
else
    source "$VENV_DIR/bin/activate"
fi

# Launch GUI
echo "Launching GUI..."
python main.py
