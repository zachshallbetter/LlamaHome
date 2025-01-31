@echo off
setlocal enabledelayedexpansion

:: Colors for output
set "GREEN=[32m"
set "RED=[31m"
set "NC=[0m"

:: Function to print status messages
:print_status
echo %GREEN%[*]%NC% %~1
exit /b 0

:print_error
echo %RED%[x]%NC% %~1
exit /b 0

:: Check Python version
python --version 2>NUL | findstr /C:"Python 3.11" >NUL
if errorlevel 1 (
    call :print_error "Python 3.11 is required but not found."
    echo Please install Python 3.11 from python.org
    exit /b 1
)

:: Create virtual environment
call :print_status "Creating virtual environment..."
python -m venv .venv

:: Activate virtual environment
call :print_status "Activating virtual environment..."
call .venv\Scripts\activate.bat

:: Upgrade pip
call :print_status "Upgrading pip..."
python -m pip install --upgrade pip

:: Install dependencies
call :print_status "Installing dependencies..."
python -m pip install -r requirements.txt

:: Install development dependencies if in dev mode
if "%1"=="--dev" (
    call :print_status "Installing development dependencies..."
    python -m pip install -e .[dev]
)

:: Install test dependencies if in test mode
if "%1"=="--test" (
    call :print_status "Installing test dependencies..."
    python -m pip install -e .[test]
)

call :print_status "Virtual environment setup complete!"
exit /b 0 