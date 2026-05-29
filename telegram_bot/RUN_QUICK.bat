@echo off
REM Quick setup for Telegram Bot - Sets environment variables and runs

echo =============================================================================
echo   EDGEDRIVE3D TELEGRAM BOT - QUICK START
echo =============================================================================
echo.

REM Check if .env exists, if not create it
if not exist .env (
    echo [!] .env file not found!
    echo.
    echo ========================================================
    echo BEFORE RUNNING, YOU NEED TO:
    echo ========================================================
    echo.
    echo 1. Create a Telegram bot:
    echo    - Open Telegram and search for @BotFather
    echo    - Send /newbot
    echo    - Follow the prompts
    echo    - COPY THE TOKEN you receive
    echo.
    echo 2. Get your User ID:
    echo    - Search for @userinfobot on Telegram
    echo    - Press Start
    echo    - COPY YOUR USER ID (number)
    echo.
    echo 3. Edit .env file and set:
    echo    TELEGRAM_BOT_TOKEN=your_token_here
    echo    TELEGRAM_ADMIN_IDS=your_user_id
    echo.
    echo ========================================================
    echo.
    
    REM Create .env from example
    if exist .env.example (
        copy .env.example .env
        echo Created .env file. Please edit it with your token and user ID.
    )
    
    pause
    exit /b 1
)

REM Quick check if token is set
findstr /C:"TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE" .env >nul
if not errorlevel 1 (
    echo [ERROR] Please edit .env and set your actual TELEGRAM_BOT_TOKEN!
    pause
    exit /b 1
)

findstr /C:"TELEGRAM_ADMIN_IDS=your_user_id_here" .env >nul
if not errorlevel 1 (
    echo [ERROR] Please edit .env and set your actual TELEGRAM_ADMIN_IDS!
    pause
    exit /b 1
)

echo [OK] .env file found
echo.

REM Load environment and run
for /f "delims=" %%a in (.env) do (
    if not "%%a"=="" (
        if not "%%a:~0,1"=="#" (
            set %%a
        )
    )
)

echo Starting bot...
echo.
python telegram_bot_main.py

pause
