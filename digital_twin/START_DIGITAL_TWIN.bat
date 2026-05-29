@echo off
title DIGITAL TWIN - Robot Car Simulation
color 0B

echo ============================================================================
echo   DIGITAL TWIN - Autonomous Robot Car Visualization System
echo ============================================================================
echo.
echo   Starting Digital Twin Servers...
echo.
echo   This will launch:
echo   - Main Digital Twin Server (ws://localhost:8765)
echo   - Hardware Sync Bridge (ws://localhost:8766)
echo   - HTTP Server (http://localhost:8080)
echo.
echo   After startup, your browser should open automatically.
echo   If not, navigate to: http://localhost:8080
echo.
echo   Controls:
echo   - Click 'Control' panel for manual control (WASD or buttons)
echo   - Click 'Hardware' panel for real robot sync
echo   - Click 'Simulation' panel for route testing
echo.
echo ============================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if required packages are installed
echo [CHECK] Verifying dependencies...
python -c "import websockets" >nul 2>&1
if errorlevel 1 (
    echo [INSTALL] Installing websockets package...
    pip install websockets
)

python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo [INSTALL] Installing numpy package...
    pip install numpy
)

echo.
echo [START] Launching Digital Twin Servers...
echo.

REM Start the main Digital Twin server
start "Digital Twin - Main Server" cmd /k "python digital_twin_server.py"
timeout /t 1 /nobreak >nul

REM Start the Hardware Sync Bridge
start "Digital Twin - Hardware Sync" cmd /k "python hardware_sync_bridge.py"
timeout /t 1 /nobreak >nul

REM Wait for servers to start
echo [WAIT] Servers starting...
timeout /t 3 /nobreak >nul

REM Open browser
echo [OPEN] Launching web interface...
start http://localhost:8080

echo.
echo ============================================================================
echo   Digital Twin is now running!
echo.
echo   Server consoles are in separate windows
echo   Web interface opened in your default browser
echo.
echo   Quick Start:
echo   1. Click 'Control' button - Use WASD or arrow buttons to drive
echo   2. Click 'Hardware' button - Enable hardware sync for real robot
echo   3. Click 'Simulation' button - Test autonomous routes
echo.
echo   Keyboard Controls:
echo   W - Forward    A - Left    S - Stop    D - Right
echo.
echo   Press any key to exit this window...
echo ============================================================================
pause >nul
