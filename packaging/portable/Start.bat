@echo off
setlocal
cd /d "%~dp0"
set "LOCALCHAT_DATA_DIR=%~dp0data"
if not exist "%LOCALCHAT_DATA_DIR%" mkdir "%LOCALCHAT_DATA_DIR%"
for %%D in (conversations settings voice uploads logs projects) do if not exist "%LOCALCHAT_DATA_DIR%\%%D" mkdir "%LOCALCHAT_DATA_DIR%\%%D"
if not exist "%~dp0LocalChat.exe" (
  echo LocalChat.exe was not found. Please extract the complete Portable ZIP.
  pause
  exit /b 1
)
start "" "%~dp0LocalChat.exe"
