@echo off
setlocal enabledelayedexpansion

:: Colors for output
set "GREEN=[32m"
set "YELLOW=[33m"
set "RED=[31m"
set "NC=[0m"

:: Function to print status messages
:print_status
echo %GREEN%[*]%NC% %~1
exit /b 0

:print_warning
echo %YELLOW%[!]%NC% %~1
exit /b 0

:print_error
echo %RED%[x]%NC% %~1
exit /b 0

:: Check system requirements
:check_requirements
call :print_status "Checking system requirements..."

:: Check Python version
python --version 2>NUL | findstr /C:"Python 3.11" >NUL
if errorlevel 1 (
    call :print_error "Python 3.11 is required but not found."
    echo Please install Python 3.11 from python.org
    exit /b 1
)

:: Check Git
git --version >NUL 2>&1
if errorlevel 1 (
    call :print_error "Git is required but not found."
    echo Please install Git from git-scm.com
    exit /b 1
)

:: Check memory
wmic ComputerSystem get TotalPhysicalMemory | findstr /R /N "^" > memory.txt
for /f "skip=1" %%A in (memory.txt) do set "total_mem=%%A"
del memory.txt
set /a "total_mem_gb=!total_mem:~0,-9!/1024"
if !total_mem_gb! LSS 16 (
    call :print_warning "Less than 16GB RAM detected. Performance may be limited."
)

:: Check disk space
for /f "tokens=3" %%A in ('dir /-c 2^>nul ^| find "bytes free"') do set "free_space=%%A"
set /a "free_space_gb=!free_space:~0,-9!/1024"
if !free_space_gb! LSS 50 (
    call :print_warning "Less than 50GB free disk space. You may need more space for models."
)

exit /b 0

:: Setup environment
:setup_environment
call :print_status "Setting up environment..."

:: Create virtual environment if it doesn't exist
if not exist .venv (
    call :print_status "Creating virtual environment..."
    python -m venv .venv
    if errorlevel 1 (
        call :print_error "Failed to create virtual environment"
        exit /b 1
    )
)

:: Activate virtual environment
call :print_status "Activating virtual environment..."
call .venv\Scripts\activate
if errorlevel 1 (
    call :print_error "Failed to activate virtual environment"
    exit /b 1
)

exit /b 0

:: Install dependencies
:install_dependencies
call :print_status "Installing dependencies..."

python scripts/install.py
if errorlevel 1 (
    call :print_error "Installation failed"
    exit /b 1
)

exit /b 0

:: Configure application
:configure_application
call :print_status "Configuring application..."

:: Setup configuration if not exists
if not exist .env (
    call :print_status "Setting up initial configuration..."
    copy .env.example .env
    call :print_warning "Please edit .env file with your preferred settings"
)

:: Create necessary directories
mkdir .data\models .data\cache .data\training .data\metrics .logs 2>NUL

exit /b 0

:: Main execution
:main
call :print_status "Starting LlamaHome setup..."

:: Run checks and setup
call :check_requirements
if errorlevel 1 exit /b 1

call :setup_environment
if errorlevel 1 exit /b 1

call :install_dependencies
if errorlevel 1 exit /b 1

call :configure_application
if errorlevel 1 exit /b 1

call :print_status "Setup complete!"
call :print_status "Starting application..."

:: Start application
python -m src.interfaces.cli

exit /b 0

:: Run main function
call :main
if errorlevel 1 (
    call :print_error "Setup failed"
    exit /b 1
) 