@echo off
REM Quick Setup and Run Script for PDF QA System (Windows)

echo.
echo 🚀 PDF Question Answering System - Quick Setup
echo ============================================== 
echo.

REM Check Python version
echo ✓ Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8 or higher
    exit /b 1
)

REM Create virtual environment
echo ✓ Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo ✓ Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate venv
    exit /b 1
)

REM Install dependencies
echo ✓ Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    exit /b 1
)
echo ✅ Dependencies installed

REM Download NLTK data
echo ✓ Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True)" >nul 2>&1
echo ✅ NLTK data downloaded

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo ✓ Creating .env file...
    copy .env.example .env >nul 2>&1
    echo ✅ .env file created (edit with your API keys if needed)
)

echo.
echo ✅ Setup complete!
echo.
echo To start the app, run:
echo   streamlit run app.py
echo.
pause
