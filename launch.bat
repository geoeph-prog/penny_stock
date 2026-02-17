@echo off
title Penny Stock Analyzer
echo ============================================
echo   Penny Stock Analyzer - Starting...
echo ============================================
echo.

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.8+ from python.org
    pause
    exit /b 1
)

:: Install dependencies if needed
pip show finvizfinance >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies (first run only)...
    pip install multitasking==0.0.11
    pip install -r requirements.txt
    echo.
)

:: Launch GUI
echo Launching GUI...
python main.py

if errorlevel 1 (
    echo.
    echo Something went wrong. Check the log above.
    pause
)
