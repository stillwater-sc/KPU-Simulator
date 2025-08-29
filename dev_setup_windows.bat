@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo KPU Simulator - Development Environment Setup
echo ========================================
echo.

REM Check prerequisites
echo Checking prerequisites...

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+ from python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✓ Python found: %PYTHON_VERSION%

REM Check if we have a supported Python version (3.8+)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)
if %MAJOR% lss 3 (
    echo ERROR: Python 3.8+ required, found %PYTHON_VERSION%
    pause
    exit /b 1
)
if %MAJOR% equ 3 if %MINOR% lss 8 (
    echo ERROR: Python 3.8+ required, found %PYTHON_VERSION%
    pause
    exit /b 1
)

REM Check CMake
cmake --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake not found. Please install CMake from cmake.org
    pause
    exit /b 1
)

for /f "tokens=3" %%i in ('cmake --version ^| findstr "cmake version"') do set CMAKE_VERSION=%%i
echo ✓ CMake found: %CMAKE_VERSION%

REM Check Visual Studio Build Tools
where cl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: Visual Studio compiler not found in PATH
    echo You may need to run this from a "Developer Command Prompt"
    echo Or install Visual Studio Build Tools
)

echo.
echo Setting up development environment...

REM Get project directory
set PROJECT_DIR=%~dp0
set PROJECT_DIR=%PROJECT_DIR:~0,-1%

REM Clean up any existing broken venv
if exist "%PROJECT_DIR%\kpu_venv" (
    echo Removing existing virtual environment...
    rmdir /s /q "%PROJECT_DIR%\kpu_venv"
)

REM Create fresh virtual environment
echo Creating Python virtual environment...
python -m venv "%PROJECT_DIR%\kpu_venv"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call "%PROJECT_DIR%\kpu_venv\Scripts\activate.bat"

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
if exist "%PROJECT_DIR%\requirements.txt" (
    echo Installing Python dependencies from requirements.txt...
    pip install -r "%PROJECT_DIR%\requirements.txt"
) else (
    echo Installing basic Python dependencies...
    pip install "pybind11[global]" numpy matplotlib seaborn jupyter ipywidgets
)

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

REM Verify critical packages
echo Verifying installation...
python -c "import pybind11; print('✓ pybind11:', pybind11.__version__)" || (
    echo ERROR: pybind11 verification failed
    pause
    exit /b 1
)

python -c "import numpy; print('✓ numpy:', numpy.__version__)" || (
    echo ERROR: numpy verification failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Development Environment Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Configure and build the C++ project:
echo    - Run: cmake_gui_setup.bat
echo    - Or use CMake GUI manually
echo    - Or use command line: cmake -S . -B build -G "Visual Studio 17 2022"
echo.
echo 2. Build the project:
echo    - In Visual Studio: Build Solution
echo    - Or command line: cmake --build build --config Release
echo.
echo 3. Test Python bindings:
echo    - Activate environment: kpu_venv\Scripts\activate
echo    - Run: python tests\test_python.py
echo.
echo 4. Start Jupyter for interactive development:
echo    - With environment activated: jupyter notebook
echo.
echo Environment details:
echo   Project: %PROJECT_DIR%
echo   Virtual env: %PROJECT_DIR%\kpu_venv
echo   Python: %PYTHON_VERSION%
echo   CMake: %CMAKE_VERSION%
echo.
pause
