@echo off
setlocal EnableDelayedExpansion

echo Setting up CMake GUI with Python virtual environment...
echo.

REM Get absolute path to current directory
set PROJECT_DIR=%~dp0
set PROJECT_DIR=%PROJECT_DIR:~0,-1%

echo Project directory: %PROJECT_DIR%

REM Check if virtual environment exists and is working
echo Checking virtual environment...
if not exist "%PROJECT_DIR%\kpu_venv\Scripts\python.exe" (
    echo Virtual environment not found. Creating fresh environment...
    goto CREATE_VENV
)

REM Test if the existing venv works
"%PROJECT_DIR%\kpu_venv\Scripts\python.exe" -c "import sys; print('Testing venv...')" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Existing virtual environment appears corrupted. Recreating...
    rmdir /s /q "%PROJECT_DIR%\kpu_venv"
    goto CREATE_VENV
)

echo Virtual environment found and working.
goto ACTIVATE_VENV

:CREATE_VENV
echo Creating new virtual environment...
python -m venv "%PROJECT_DIR%\kpu_venv"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python is installed and accessible
    pause
    exit /b 1
)

:ACTIVATE_VENV
REM Activate virtual environment by setting up the environment
set VIRTUAL_ENV=%PROJECT_DIR%\kpu_venv
set PATH=%PROJECT_DIR%\kpu_venv\Scripts;%PATH%
set PYTHONHOME=
set PYTHONPATH=

echo Virtual environment activated.

REM Install/upgrade dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip --quiet
python -m pip install "pybind11[global]" numpy matplotlib seaborn jupyter ipywidgets --quiet

REM Verify installation
echo Verifying Python installation...
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import pybind11; print('pybind11 version:', pybind11.__version__)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: pybind11 verification failed
    pause
    exit /b 1
)

REM Get pybind11 CMake directory
echo Detecting pybind11 CMake configuration...
for /f "tokens=*" %%i in ('python -m pybind11 --cmake 2^>nul') do set PYBIND11_DIR=%%i

if "%PYBIND11_DIR%"=="" (
    echo ERROR: Could not detect pybind11 CMake directory
    echo Try running manually: python -m pybind11 --cmake
    pause
    exit /b 1
)

echo pybind11 CMake directory: %PYBIND11_DIR%

REM Set up CMake environment variables
set Python_ROOT_DIR=%PROJECT_DIR%\kpu_venv
set Python_EXECUTABLE=%PROJECT_DIR%\kpu_venv\Scripts\python.exe
set Python_FIND_VIRTUALENV=FIRST
set pybind11_DIR=%PYBIND11_DIR%

REM Create build directory if it doesn't exist
if not exist "%PROJECT_DIR%\build" mkdir "%PROJECT_DIR%\build"

REM Clear any existing CMake cache to avoid conflicts
if exist "%PROJECT_DIR%\build\CMakeCache.txt" (
    echo Clearing existing CMake cache...
    del "%PROJECT_DIR%\build\CMakeCache.txt"
)

REM Create a CMake initial cache file to pre-configure variables
echo Creating CMake initial cache...
(
echo # CMake initial cache for KPU Simulator with Python venv
echo set^(Python_ROOT_DIR "%Python_ROOT_DIR%" CACHE PATH "Python virtual environment root"^)
echo set^(Python_EXECUTABLE "%Python_EXECUTABLE%" CACHE FILEPATH "Python executable"^)
echo set^(Python_FIND_VIRTUALENV "FIRST" CACHE STRING "Find virtual env first"^)
echo set^(pybind11_DIR "%pybind11_DIR%" CACHE PATH "pybind11 CMake directory"^)
echo set^(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type"^)
) > "%PROJECT_DIR%\build\CMakeInitialCache.cmake"

REM Display configuration
echo.
echo ========== CMake Configuration ==========
echo Project directory: %PROJECT_DIR%
echo Build directory: %PROJECT_DIR%\build
echo Python executable: %Python_EXECUTABLE%
echo Python root: %Python_ROOT_DIR%
echo pybind11 directory: %pybind11_DIR%
echo Initial cache: %PROJECT_DIR%\build\CMakeInitialCache.cmake
echo =========================================
echo.

echo Launching CMake GUI...
echo.
echo The CMake GUI will open with the project pre-configured.
echo Source directory: %PROJECT_DIR%
echo Build directory: %PROJECT_DIR%\build
echo.
echo If the GUI opens a different project:
echo 1. Click "Browse Source..." and select: %PROJECT_DIR%
echo 2. Click "Browse Build..." and select: %PROJECT_DIR%\build
echo 3. Click "Configure" and select "Visual Studio 17 2022"
echo.

REM Launch CMake GUI with explicit source and build directories
cmake-gui "%PROJECT_DIR%" -B "%PROJECT_DIR%\build" -C "%PROJECT_DIR%\build\CMakeInitialCache.cmake"

echo.
echo CMake GUI session ended.
echo.
echo If you want to build from command line instead:
echo   cmake --build build --config Release
echo.
echo Virtual environment remains active in this terminal.
pause
