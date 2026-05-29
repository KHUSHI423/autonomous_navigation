@echo off
echo ============================================================================
echo   KILLING ALL OLD PYTHON PROCESSES
echo ============================================================================
echo.

taskkill /F /IM python.exe 2>nul
if errorlevel 1 (
    echo No Python processes found or killed
) else (
    echo Successfully killed all Python processes
)

echo.
echo ============================================================================
echo   DONE! You can now start the system fresh.
echo ============================================================================
echo.
echo START IN THIS ORDER:
echo.
echo 1. TERMINAL 1 - Hardware Sync Bridge:
echo    cd hardware_setup\digital_twin
echo    python hardware_sync_bridge.py
echo.
echo 2. TERMINAL 2 - Robot Car (wait 2 seconds after Terminal 1):
echo    cd hardware_setup\robot_car_and_pi_v2_2esp
echo    python robot_car_v2_complete.py
echo.
echo 3. TERMINAL 3 - Web Server (wait 2 seconds after Terminal 2):
echo    cd hardware_setup\digital_twin
echo    python -m http.server 8080
echo.
echo 4. BROWSER: http://localhost:8080
echo.
echo ============================================================================
pause
