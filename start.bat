@echo off
setlocal enabledelayedexpansion
title Pulmonary Nodule AI Diagnostic System
cd /d "%~dp0"

echo ============================================================
echo   PULMONARY NODULE DIAGNOSTIC INTELLIGENCE (CADx)
echo ============================================================
echo.

:: Detect Python environment
set "PY_BIN=python"
if exist ".venv\Scripts\python.exe" (
    set "PY_BIN=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PY_BIN=venv\Scripts\python.exe"
)

:: Verify Python is accessible
"%PY_BIN%" --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found on your system PATH.
    echo Please install Python 3.10+ or configure your environment.
    pause
    exit /b 1
)

:: Quick menu
echo Select launch mode:
echo   [1] Launch PACS Desktop Application (Default)
echo   [2] Launch in Demo Mode (Simulated UI)
echo   [3] Launch FastAPI Microservice Server
echo   [4] Run Pipeline Verification & Diagnostics
echo   [5] Create Desktop Shortcut Icon
echo.
set /p choice="Enter option [1-5] (Press Enter for 1): "

if "%choice%"=="" set choice=1
if "%choice%"=="1" goto launch_gui
if "%choice%"=="2" goto launch_demo
if "%choice%"=="3" goto launch_api
if "%choice%"=="4" goto launch_verify
if "%choice%"=="5" goto create_shortcut

:launch_gui
echo.
echo Launching PACS Desktop Application...
"%PY_BIN%" gui.py
goto end

:launch_demo
echo.
echo Launching in Clinical Demo Mode (NODULE_DEMO=1)...
set "NODULE_DEMO=1"
"%PY_BIN%" gui.py
goto end

:launch_api
echo.
echo Launching FastAPI Microservice on http://localhost:8000...
"%PY_BIN%" -m uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload
goto end

:launch_verify
echo.
echo Running Full Pipeline Verification...
"%PY_BIN%" scripts\verify_e2e.py
echo.
pause
goto end

:create_shortcut
echo.
echo Creating Desktop Shortcut with Medical Icon...
"%PY_BIN%" scripts\create_shortcut.py
echo.
pause
goto end

:end

