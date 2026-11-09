@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0launcher.py"
    if errorlevel 1 pause
    exit /b
)

where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0launcher.py"
    if errorlevel 1 pause
    exit /b
)

echo Python 3 was not found. Install Python 3 and try again.
pause
