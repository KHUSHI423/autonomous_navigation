@echo off
title Autonomous Modes Visualization - Quick Start
color 0B

echo ====================================================================
echo           AUTONOMOUS ROBOT MODES - 3D VISUALIZATION
echo                      Quick Start Launcher
echo ====================================================================
echo.
echo  This will launch the 3D interactive visualization system for all
echo  12 autonomous robot modes.
echo.
echo  Modes Include:
echo    - Delivery, Ambulance, Crash Response
echo    - Follow-Me, Summon, Auto-Park, Escort
echo    - Elder Assist, Guidance, Medical Delivery, Hospital Assist
echo.
echo ====================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo.
    echo Please install Python from https://www.python.org/
    echo OR open index.html directly in your browser.
    echo.
    pause
    exit /b 1
)

echo [OK] Python detected
echo.

REM Get the current directory
set MODES_DIR=%~dp0

echo Starting HTTP Server...
echo.
echo ====================================================================
echo   Visualization Hub will open at:
echo   http://localhost:8080
echo.
echo   Press Ctrl+C to stop the server
echo ====================================================================
echo.

REM Start Python HTTP server
cd /d "%MODES_DIR%"
start http://localhost:8080
python -m http.server 8080

pause
