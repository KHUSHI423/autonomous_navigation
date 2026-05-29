@echo off
REM =============================================================================
REM EDGE DRIVE 3D - WINDOWS SETUP SCRIPT
REM =============================================================================

echo.
echo ============================================================================
echo     EdgeDrive3D - Automated Setup for Windows
echo ============================================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.8+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Create virtual environment
echo [STEP 1/4] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    echo Virtual environment created
)
echo.

REM Activate virtual environment
echo [STEP 2/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [STEP 3/4] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo.

REM Install requirements
echo [STEP 4/4] Installing dependencies (this may take 5-10 minutes)...
echo.
pip install numpy opencv-python Pillow PyYAML requests tqdm --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Some packages failed to install, continuing...
)

echo.
echo Installing PyTorch (this is the largest download)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
if %errorlevel% neq 0 (
    echo [INFO] CUDA PyTorch failed, installing CPU version...
    pip install torch torchvision --quiet
)

echo.
echo Installing Ultralytics YOLO...
pip install ultralytics --quiet

echo.
echo Installing visualization libraries...
pip install plotly open3d --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Open3D failed, 3D visualization may be limited
)

echo.
echo Installing Streamlit...
pip install streamlit --quiet

echo.
echo ============================================================================
echo     Setup Complete!
echo ============================================================================
echo.
echo Next steps:
echo   1. Activate environment: venv\Scripts\activate
echo   2. Test installation:    python main.py --help
echo   3. Run dashboard:        streamlit run dashboard\app.py
echo   4. Process image:        python main.py image test.jpg -o output/
echo.
echo For Raspberry Pi streaming:
echo   - On Pi: python hardware\pi_sender.py ^<LAPTOP_IP^> -p 5000
echo   - On Laptop: python main.py pi-stream --port 5000
echo.
echo For ESP32 motor control:
echo   - Flash hardware\hardware.ino to ESP32
echo   - Run: python main.py pi-stream --auto --esp32-ip 192.168.4.1
echo.
echo ============================================================================
echo.

pause
