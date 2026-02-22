#!/bin/bash
echo "============================================"
echo "  Penny Stock Analyzer - Starting..."
echo "============================================"
echo

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install Python 3.8+"
    exit 1
fi

# Install dependencies if needed
if ! python3 -c "import finvizfinance" 2>/dev/null; then
    echo "Installing dependencies (first run only)..."
    pip3 install multitasking==0.0.11
    pip3 install -r requirements.txt
    echo
fi

# Launch GUI
echo "Launching GUI..."
python3 main.py
