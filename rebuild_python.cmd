@echo off
REM Quick rebuild script for Python bindings after code changes

echo ======================================================================
echo Rebuilding Python Bindings
echo ======================================================================
echo.

cd /d %~dp0build_msvc

echo Building stillwater_kpu...
cmake --build . --config Release --target stillwater_kpu
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    exit /b 1
)

echo.
echo ======================================================================
echo Build Complete!
echo ======================================================================
echo.
echo Python module location:
echo   %~dp0build_msvc\src\bindings\python\Release\stillwater_kpu.pyd
echo.
echo To test, run:
echo   python test_dma_python.py
echo.
