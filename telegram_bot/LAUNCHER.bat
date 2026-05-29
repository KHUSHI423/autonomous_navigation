@echo off
REM =============================================================================
REM EdgeDrive3D Telegram Bot - Complete Launcher with Setup Check
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
    copy .env.example .env >nul
    echo [OK] .env file created
    echo.
    echo =============================================================================
    echo IMPORTANT: Configure .env file before running!
    echo =============================================================================
    echo.
    echo 1. Open .env in a text editor (e.g., Notepad)
    echo 2. Find these lines:
    echo    TELEGRAM_BOT_TOKEN=your_bot_token_here
    echo    TELEGRAM_ADMIN_IDS=your_user_id_here
    echo.
    echo 3. Replace with your actual values:
    echo    TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
    echo    TELEGRAM_ADMIN_IDS=123456789
    echo.
    echo 4. Save the file and run this script again
    echo.
    echo =============================================================================
    echo.
    echo GET YOUR BOT TOKEN:
    echo 1. Open Telegram and search for @BotFather
    echo 2. Send /newbot
    echo 3. Follow the prompts
    echo 4. Copy the token you receive
    echo.
    echo GET YOUR USER ID:
    echo 1. Search for @userinfobot on Telegram
    echo 2. Press Start
    echo 3. Copy your User ID
    echo.
    echo =============================================================================
    pause
    exit /b 1
)

REM Load environment variables from .env
echo [?] Loading configuration from .env...
for /f "tokens=*" %%a in (.env) do (
    if not "%%a"=="" (
        if not "%%a:~0,1"=="#" (
            set %%a
        )
    )
)

REM Validate configuration
echo [?] Validating configuration...

if "%TELEGRAM_BOT_TOKEN%"=="" (
    echo [ERROR] TELEGRAM_BOT_TOKEN not set!
    echo Please edit .env and set your bot token from @BotFather
    pause
    exit /b 1
)

if "%TELEGRAM_BOT_TOKEN%"=="YOUR_BOT_TOKEN_HERE" (
    echo [ERROR] Please replace YOUR_BOT_TOKEN_HERE with your actual token!
    echo Get token from @BotFather on Telegram
    pause
    exit /b 1
)

if "%TELEGRAM_ADMIN_IDS%"=="" (
    echo [ERROR] TELEGRAM_ADMIN_IDS not set!
    echo Please edit .env and set your Telegram user ID
    echo Get your ID from @userinfobot
    pause
    exit /b 1
)

if "%TELEGRAM_ADMIN_IDS%"=="your_user_id_here" (
    echo [ERROR] Please replace your_user_id_here with your actual user ID!
    echo Get your ID from @userinfobot on Telegram
    pause
    exit /b 1
)

echo [OK] Configuration validated
echo.

REM Display configuration
echo =============================================================================
echo   CONFIGURATION
echo =============================================================================
echo Bot Token:    %TELEGRAM_BOT_TOKEN:~0,15%... (hidden for security)
echo Admin IDs:    %TELEGRAM_ADMIN_IDS%
echo ESP32 IP:     %ESP32_IP%
echo Autonomous:   %AUTONOMOUS_ENABLED%
echo Geofencing:   %GEOFENCE_ENABLED%
echo =============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

python --version
echo [OK] Python found
echo.

REM Check if dependencies are installed
echo [?] Checking dependencies...
python -c "import telegram" >nul 2>&1
if errorlevel 1 (
    echo [!] Installing dependencies...
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

REM Create directories
if not exist logs mkdir logs
if not exist charts mkdir charts

echo =============================================================================
echo   STARTING TELEGRAM BOT
echo =============================================================================
echo.
echo Bot is starting... Press Ctrl+C to stop
echo.
echo Connecting to Telegram...
echo.

REM Run the bot
python telegram_bot_main.py

echo.
echo =============================================================================
echo   BOT STOPPED
echo =============================================================================
pause
