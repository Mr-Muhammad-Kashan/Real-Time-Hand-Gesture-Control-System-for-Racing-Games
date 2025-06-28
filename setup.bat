@echo off
REM Real-Time Hand Gesture Control System Setup Script
REM Author: Muhammad Kashan Tariq
REM Automated setup for Windows systems

echo ============================================================
echo REAL-TIME HAND GESTURE CONTROL SYSTEM SETUP
echo ============================================================
echo Author: Muhammad Kashan Tariq ^| Version: 1.0.0
echo Optimized for Intel 13th or 14th Gen + NVIDIA RTX GPUs + 16GB RAM
echo ============================================================

echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✓ Python found
python --version

echo.
echo Creating virtual environment...
if exist "Python-3.10" (
    echo Virtual environment already exists. Removing old version...
    rmdir /s /q "Python-3.10"
)

python -m venv Python-3.10
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment created

echo.
echo Activating virtual environment...
call Python-3.10\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing required packages...
echo This may take 5-10 minutes depending on your internet connection...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install some packages
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ✓ All packages installed successfully!

echo.
echo Testing system components...
python -c "import cv2; print('✓ OpenCV:', cv2.__version__)" 2>nul
python -c "import mediapipe as mp; print('✓ MediaPipe:', mp.__version__)" 2>nul
python -c "import torch; print('✓ PyTorch:', torch.__version__); print('✓ CUDA Available:', torch.cuda.is_available())" 2>nul
python -c "import pynput; print('✓ PyInput: Ready')" 2>nul
python -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>nul

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo 1. Ensure your webcam is connected and working
echo 2. Close other applications for optimal performance  
echo 3. Run the system: python main.py
echo.
echo For detailed instructions, see README.md
echo For troubleshooting, check the README troubleshooting section
echo.
echo Press any key to exit setup...
pause >nul
