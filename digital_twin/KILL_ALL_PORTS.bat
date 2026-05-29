@echo off
echo ============================================================================
echo   KILLING PROCESSES USING PORTS 8080, 8765, 8766, 8767
echo ============================================================================
echo.

for /f "tokens=5" %%a in ('netstat -aon ^| find ":8080" ^| find "LISTENING"') do (
    echo Killing process %%a using port 8080
    taskkill /F /PID %%a 2>nul
)

for /f "tokens=5" %%a in ('netstat -aon ^| find ":8765" ^| find "LISTENING"') do (
    echo Killing process %%a using port 8765
    taskkill /F /PID %%a 2>nul
)

for /f "tokens=5" %%a in ('netstat -aon ^| find ":8766" ^| find "LISTENING"') do (
    echo Killing process %%a using port 8766
    taskkill /F /PID %%a 2>nul
)

for /f "tokens=5" %%a in ('netstat -aon ^| find ":8767" ^| find "LISTENING"') do (
    echo Killing process %%a using port 8767
    taskkill /F /PID %%a 2>nul
)

echo.
echo Done! Ports should now be free.
echo.
pause
