@echo off
REM =============================================================================
REM EdgeDrive3D Telegram Bot - Windows Launcher
REM =============================================================================

echo =============================================================================
echo   EDGEDRIVE3D TELEGRAM BOT - LAUNCHER
echo =============================================================================
echo.

REM Check if .env exists
if not exist .env (
    echo [!] .env file not found!
    echo.
    echo Creating .env from template...
    copy .env.example .env
    echo.
    echo =============================================================================
    echo IMPORTANT: Configure .env file before running!
    echo.
    echo 1. Open .env in a text editor
    echo 2. Set TELEGRAM_BOT_TOKEN to your bot token
    echo 3. Set TELEGRAM_ADMIN_IDS to your user ID
    echo 4. Update ESP32_IP if needed
    echo =============================================================================
    echo.
    pause
    exit /b 1
)

REM Load environment variables from .env
for /f "delims=" %%a in (.env) do (
    if not "%%a"=="" (
        if not "%%a"=="" (
            if not "%%a:~0,1"=="#" (
                set %%a
            )
        )
    )
)

echo [?] Checking configuration...
echo.

REM Check if token is set
if "%TELEGRAM_BOT_TOKEN%"=="" (
    echo [ERROR] TELEGRAM_BOT_TOKEN not set!
    echo Please edit .env and set your bot token
    pause
    exit /b 1
)

REM Check if admin IDs are set
if "%TELEGRAM_ADMIN_IDS%"=="" (
    echo [ERROR] TELEGRAM_ADMIN_IDS not set!
    echo Please edit .env and set your user ID
    pause
    exit /b 1
)

echo [OK] Configuration loaded
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Check if dependencies are installed
echo [?] Checking dependencies...
python -c "import telegram" >nul 2>&1
if errorlevel 1 (
    echo [!] Dependencies not installed. Installing now...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies!
        pause
        exit /b 1
    )
) else (
    echo [OK] Dependencies installed
)
echo.

REM Create logs directory
if not exist logs mkdir logs
if not exist charts mkdir charts

echo =============================================================================
echo Starting Telegram Bot...
echo =============================================================================
echo.
echo Bot Token: %TELEGRAM_BOT_TOKEN:~0,10%...
echo Admin IDs: %TELEGRAM_ADMIN_IDS%
echo ESP32 IP:  %ESP32_IP%
echo.
echo Press Ctrl+C to stop
echo.
echo =============================================================================
echo.

REM Run the bot
python telegram_bot_main.py

pause
