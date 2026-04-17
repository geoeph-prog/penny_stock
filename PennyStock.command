#!/bin/bash
# Double-click this file in Finder to launch Penny Stock Analyzer.
# On first run it creates a venv and installs dependencies automatically.

set -e

cd "$(dirname "$0")"

VENV_DIR=".venv"
PYTHON=""

# ── Find the best available Python (prefer 3.10+ for SSL support) ──
# Check Homebrew versioned binaries first, then generic python3.
for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
        major=$(echo "$ver" | tr -d '(),' | awk '{print $1}')
        minor=$(echo "$ver" | tr -d '(),' | awk '{print $2}')
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ] 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "  Python 3.9+ is required but wasn't found."
    echo ""
    echo "  Install it with Homebrew:"
    echo "    /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "    brew install python@3.12"
    echo ""
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
fi

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "============================================"
echo "  Penny Stock Analyzer  (Python $PY_VER)"
echo "============================================"
echo ""

# ── Set up virtual environment on first run ─────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "First run — setting up environment..."
    "$PYTHON" -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install multitasking==0.0.11
    pip install -r requirements.txt
    echo "Setup complete."
    echo ""
else
    source "$VENV_DIR/bin/activate"
fi

# ── Launch ──────────────────────────────────────────────────────────
echo "Launching..."
python main.py
