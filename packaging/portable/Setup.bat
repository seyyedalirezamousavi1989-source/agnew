@echo off
setlocal
cd /d "%~dp0"
echo ==============================================
echo       LocalChat 2.0 - Portable preparation
echo ==============================================
if not exist "%~dp0data" mkdir "%~dp0data"
if not exist "%~dp0data\voice" mkdir "%~dp0data\voice"
if not exist "%~dp0data\logs" mkdir "%~dp0data\logs"
if exist "%~dp0LocalChat.exe" (
  echo [OK] Application files found.
) else (
  echo [ERROR] LocalChat.exe is missing. Extract the complete ZIP first.
  pause
  exit /b 1
)
where ollama >nul 2>nul
if errorlevel 1 (
  echo [INFO] Ollama was not detected. You can install it later from https://ollama.com/download
) else (
  echo [OK] Ollama command found.
)
echo.
echo Preparation complete. Run Start.bat to launch LocalChat.
pause
