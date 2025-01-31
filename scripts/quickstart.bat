@echo off

REM Check Python version
python --version 2>NUL | findstr /C:"Python 3.11" >NUL
if errorlevel 1 (
    echo Python 3.11 is required but not found.
    echo Please install Python 3.11 from python.org
    exit /b 1
)

REM Check Git
git --version >NUL 2>&1
if errorlevel 1 (
    echo Git is required but not found.
    echo Please install Git from git-scm.com
    exit /b 1
)

REM Run installation
echo Running installation...
python scripts/install.py

REM Setup configuration if not exists
if not exist .env (
    echo Setting up initial configuration...
    copy .env.example .env
    echo Please edit .env file with your preferred settings
)

REM Activate virtual environment and start application
echo Starting application...
call .venv\Scripts\activate
python -m src.interfaces.cli 