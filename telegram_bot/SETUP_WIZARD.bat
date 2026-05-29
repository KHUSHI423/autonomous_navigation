@echo off
REM =============================================================================
REM EdgeDrive3D Telegram Bot - Quick Setup Script
REM =============================================================================

echo =============================================================================
echo   EDGEDRIVE3D TELEGRAM BOT - QUICK SETUP
echo =============================================================================
echo.
echo This script will help you configure the bot in 2 minutes!
echo.
echo =============================================================================
echo STEP 1: CREATE YOUR TELEGRAM BOT
echo =============================================================================
echo.
echo 1. Open Telegram on your phone or computer
echo 2. Search for: @BotFather
echo 3. Send message: /newbot
echo 4. Choose a name (e.g., "My Robot Car")
echo 5. Choose a username (e.g., "edge_drive_bot")
echo 6. COPY THE TOKEN (looks like: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz)
echo.
echo Press any key when you have your token...
pause >nul
echo.

echo =============================================================================
echo STEP 2: GET YOUR USER ID
echo =============================================================================
echo.
echo 1. Search for: @userinfobot
echo 2. Press Start
echo 3. Copy your User ID (a number like: 123456789)
echo.
echo Press any key when you have your user ID...
pause >nul
echo.

echo =============================================================================
echo STEP 3: ENTER YOUR CONFIGURATION
echo =============================================================================
echo.

REM Get bot token
set /p BOT_TOKEN="Enter your BOT TOKEN: "
if "%BOT_TOKEN%"=="" (
    echo [ERROR] Token cannot be empty!
    pause
    exit /b 1
)

echo.
set /p USER_ID="Enter your USER ID: "
if "%USER_ID%"=="" (
    echo [ERROR] User ID cannot be empty!
    pause
    exit /b 1
)

echo.
echo ESP32 IP address (default: 10.17.122.207):
set /p ESP32_IP="Enter ESP32 IP: "
if "%ESP32_IP%"=="" set ESP32_IP=10.17.122.207

echo.
echo =============================================================================
echo STEP 4: SAVING CONFIGURATION
echo =============================================================================
echo.

REM Create .env file
(
echo # EdgeDrive3D Telegram Bot Configuration
echo # Generated on %DATE% at %TIME%
echo.
echo # Telegram Bot Settings
echo TELEGRAM_BOT_TOKEN=%BOT_TOKEN%
echo TELEGRAM_ADMIN_IDS=%USER_ID%
echo.
echo # Robot Car Settings
echo ESP32_IP=%ESP32_IP%
echo ESP32_COMMAND_PORT=9000
echo ESP32_STATUS_PORT=9001
echo UDP_VIDEO_PORT=5000
echo.
echo # Feature Settings
echo AUTONOMOUS_ENABLED=true
echo GEOFENCE_ENABLED=true
echo GEOFENCE_RADIUS=100.0
echo.
echo # GPS Settings
echo GPS_SIMULATE=true
echo GPS_PORT=auto
echo GPS_BAUDRATE=9600
echo.
echo # Update Intervals (milliseconds)
echo TELEMETRY_INTERVAL=5000
echo VIDEO_INTERVAL=2000
echo GPS_INTERVAL=3000
echo.
echo # Logging
echo LOG_LEVEL=INFO
) > .env

echo [OK] Configuration saved to .env
echo.

echo =============================================================================
echo STEP 5: INSTALL DEPENDENCIES
echo =============================================================================
echo.
echo Installing required Python packages...
echo.

pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo =============================================================================
echo SETUP COMPLETE!
echo =============================================================================
echo.
echo Your bot is ready to run!
echo.
echo NEXT STEPS:
echo 1. Run LAUNCHER.bat to start the bot
echo 2. Open Telegram and find your bot
echo 3. Press START or send /start
echo.
echo =============================================================================
echo.
echo Would you like to start the bot now? (Y/N)
set /p START_NOW="> "
if /i "%START_NOW%"=="Y" (
    echo.
    echo Starting bot...
    echo.
    python telegram_bot_main.py
) else (
    echo.
    echo You can start the bot anytime by running LAUNCHER.bat
)

pause
