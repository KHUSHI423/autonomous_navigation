@echo off
title DIGITAL TWIN - DIAGNOSTIC TOOL
color 0B

echo ============================================================================
echo   DIGITAL TWIN - REAL-TIME UPDATE DIAGNOSTIC
echo ============================================================================
echo.
echo This tool will check if all components are working correctly.
echo.
echo Press Ctrl+C at any time to stop
echo.
echo ============================================================================
echo.

:MENU
echo.
echo Choose diagnostic test:
echo.
echo 1. Test Hardware Sync Bridge (port 8767)
echo 2. Test Robot Car Detection Server (port 5002)
echo 3. Test Web Server (will find free port)
echo 4. Kill all Python processes
echo 5. Exit
echo.
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto TEST_HARDWARE
if "%choice%"=="2" goto TEST_DETECTION
if "%choice%"=="3" goto TEST_WEBSERVER
if "%choice%"=="4" goto KILL_PYTHON
if "%choice%"=="5" goto END

echo Invalid choice!
goto MENU

:TEST_HARDWARE
echo.
echo ============================================================================
echo   TESTING HARDWARE SYNC BRIDGE
echo ============================================================================
echo.
echo Checking if hardware_sync_bridge.py is running...
echo.

for /f "tokens=5" %%a in ('netstat -aon ^| find ":8767" ^| find "LISTENING"') do (
    echo ✅ Port 8767 is in use by process %%a
    goto HARDWARE_RUNNING
)

echo ❌ Port 8767 is NOT in use!
echo    hardware_sync_bridge.py is not running!
echo.
echo    Start it with:
echo    cd hardware_setup\digital_twin
echo    python hardware_sync_bridge.py
echo.
pause
goto MENU

:HARDWARE_RUNNING
echo.
echo Testing sensor data endpoint...
echo.

python -c "import requests; r=requests.get('http://localhost:8767/sensors', timeout=2); print('HTTP Status:', r.status_code); print('Data:', r.json())" 2>nul
if errorlevel 1 (
    echo.
    echo ❌ Cannot connect to http://localhost:8767/sensors
    echo    hardware_sync_bridge.py may be running but not responding
) else (
    echo.
    echo ✅ Hardware Sync Bridge is working!
)
echo.
pause
goto MENU

:TEST_DETECTION
echo.
echo ============================================================================
echo   TESTING ROBOT CAR DETECTION SERVER
echo ============================================================================
echo.
echo Checking if robot_car_v2_complete.py is running...
echo.

for /f "tokens=5" %%a in ('netstat -aon ^| find ":5002" ^| find "LISTENING"') do (
    echo ✅ Port 5002 is in use by process %%a
    goto DETECTION_RUNNING
)

echo ❌ Port 5002 is NOT in use!
echo    robot_car_v2_complete.py is not running!
echo.
echo    Start it with:
echo    cd hardware_setup\robot_car_and_pi_v2_2esp
echo    python robot_car_v2_complete.py
echo.
pause
goto MENU

:DETECTION_RUNNING
echo.
echo Testing detection endpoint...
echo.

python -c "import requests; r=requests.get('http://localhost:5002/detections', timeout=2); print('HTTP Status:', r.status_code); print('Data:', r.json())" 2>nul
if errorlevel 1 (
    echo.
    echo ❌ Cannot connect to http://localhost:5002/detections
) else (
    echo.
    echo ✅ Detection Server is working!
)
echo.
pause
goto MENU

:TEST_WEBSERVER
echo.
echo ============================================================================
echo   STARTING WEB SERVER
echo ============================================================================
echo.
echo Finding free port...
echo.

for %%p in (9090 9091 9092 9093 9094 9095) do (
    netstat -aon | find "%%p" | find "LISTENING" >nul
    if errorlevel 1 (
        echo Port %%p is FREE
        echo.
        echo Starting web server on port %%p...
        echo.
        echo ================================================================
        echo   WEB SERVER STARTED
        echo   Open: http://localhost:%%p
        echo ================================================================
        echo.
        python -m http.server %%p
        goto MENU
    )
)

echo Could not find free port in range 9090-9095
pause
goto MENU

:KILL_PYTHON
echo.
echo ============================================================================
echo   KILLING ALL PYTHON PROCESSES
echo ============================================================================
echo.

taskkill /F /IM python.exe 2>nul
if errorlevel 1 (
    echo No Python processes found
) else (
    echo Successfully killed all Python processes
)
echo.
pause
goto MENU

:END
echo.
echo Goodbye!
