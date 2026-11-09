@echo off
setlocal
cd /d "%~dp0"
set "LOCALCHAT_DATA_DIR=%~dp0data"
for %%D in (conversations settings voice uploads logs projects) do if not exist "%LOCALCHAT_DATA_DIR%\%%D" mkdir "%LOCALCHAT_DATA_DIR%\%%D"
if not exist "%~dp0LocalChat.exe" (
  echo [ERROR] LocalChat.exe is missing. Re-extract the Portable package.
  pause
  exit /b 1
)
echo Portable folders checked. User data and downloaded models were not deleted.
echo If the app files are damaged, extract a fresh release into a new folder and copy the data folder.
pause
