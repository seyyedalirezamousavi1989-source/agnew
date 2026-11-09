@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PY="
rem Python is required only on the BUILD computer, not on computers receiving Setup.exe.
where py >nul 2>nul
if not errorlevel 1 (
  py -3.10 -c "import sys; assert sys.version_info >= (3,10) and sys.version_info < (3,11)" >nul 2>nul && set "PY=py -3.10"
  if not defined PY py -3.11 -c "import sys" >nul 2>nul && set "PY=py -3.11"
  if not defined PY py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
)
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 python -c "import sys; assert sys.version_info >= (3,10) and sys.version_info < (3,13)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python 3.10, 3.11 or 3.12 is required on the BUILD computer.
  pause & exit /b 1
)
%PY% -c "import sys; print('Build Python:', sys.version); assert sys.maxsize > 2**32, '64-bit Python is required for MetaTrader5'"
if errorlevel 1 (echo [ERROR] A 64-bit Python interpreter is required.& pause & exit /b 1)

echo [1/5] Preparing test runner...
%PY% -m pip --version >nul 2>nul
if errorlevel 1 (
  %PY% -m ensurepip --upgrade
  if errorlevel 1 (echo [ERROR] pip could not be initialized.& pause & exit /b 1)
)
%PY% -m pip show pytest >nul 2>nul
if errorlevel 1 (
  %PY% -m pip install "pytest>=7,<9"
  if errorlevel 1 (echo [ERROR] Could not install pytest.& pause & exit /b 1)
)

echo [2/5] Running regression tests...
%PY% -m pytest -q local_chat
if errorlevel 1 (echo [ERROR] Regression tests failed. Installer was NOT created.& pause & exit /b 1)

echo [3/5] Installing build dependencies...
rem Keep NumPy on the stable 1.x ABI for the bundled MT5 extension. NumPy
rem must be included in the frozen app; excluding it breaks MT5 at runtime.
%PY% -m pip install --upgrade --force-reinstall "numpy>=1.26.4,<2.0"
if errorlevel 1 (echo [ERROR] Could not install the supported NumPy runtime.& pause & exit /b 1)
%PY% -m pip install --upgrade "pyinstaller>=6,<7" pywebview pypdf Pillow "MetaTrader5>=5.0.45"
if errorlevel 1 (echo [ERROR] Build dependency installation failed.& pause & exit /b 1)
%PY% -c "import numpy as np; from numpy.core import multiarray; print('NumPy:', np.__version__, np.__file__); print('NumPy multiarray:', multiarray.__file__); import MetaTrader5 as mt5; print('MetaTrader5 package:', mt5.__file__); print('MetaTrader5 version:', getattr(mt5, '__version__', 'unknown'))"
if errorlevel 1 (echo [ERROR] NumPy multiarray or MetaTrader5 cannot be imported by the selected build Python.& pause & exit /b 1)

echo [4/5] Building application payload...
if exist build rmdir /s /q build
rem Windows MAX_PATH (260) workaround: the project folder path is long, and
rem numpy.libs\libopenblas64__*.dll pushes the COLLECT destination past 260
rem characters. Build in a short folder instead of inside the project tree.
set "LCB=%LOCALAPPDATA%\LCB"
if exist "%LCB%" rmdir /s /q "%LCB%"
mkdir "%LCB%"
%PY% -m PyInstaller packaging\localchat.spec --noconfirm --clean --distpath "%LCB%\payload" --workpath "%LCB%\work"
if errorlevel 1 (echo [ERROR] PyInstaller failed.& pause & exit /b 1)

set "APPDIR=%LCB%\payload\LocalChat"
if not exist "%APPDIR%\LocalChat.exe" (echo [ERROR] PyInstaller output LocalChat.exe was not found.& pause & exit /b 1)
%PY% packaging\verify_bundle.py "%APPDIR%"
if errorlevel 1 (echo [ERROR] Built app folder is missing MT5/NumPy files.& pause & exit /b 1)

rem Locate Inno Setup 7 first (user may already have installed it), then 6.
rem Inno Setup 7 can compile the existing .iss script; no downgrade is needed.
set "ISCC="
where ISCC.exe >nul 2>nul && set "ISCC=ISCC.exe"
where iscc >nul 2>nul && if not defined ISCC set "ISCC=iscc"

rem Per-user and machine-wide default install locations.
for %%V in (7 6) do (
  if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup %%V\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup %%V\ISCC.exe"
  if not defined ISCC if exist "%ProgramFiles%\Inno Setup %%V\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup %%V\ISCC.exe"
  if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup %%V\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup %%V\ISCC.exe"
)

if not defined ISCC (
  echo [ERROR] Inno Setup 7 or 6 is not detected.
  echo Install Inno Setup 7.1.1 x64, then rerun this script.
  echo Expected compiler: "%%ProgramFiles%%\Inno Setup 7\ISCC.exe" or "%%ProgramFiles(x86)%%\Inno Setup 7\ISCC.exe".
  pause & exit /b 1
)

echo Using Inno compiler: %ISCC%

echo [5/6] Preparing bundled Persian installer translation...
rem Always fetch a fresh copy to a temporary file. This prevents a stale,
rem truncated, or HTML error response from a previous run being reused.
set "FARSI_TMP=packaging\Farsi.download.isl"
if exist "%FARSI_TMP%" del /q "%FARSI_TMP%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/jrsoftware/issrc/main/Files/Languages/Unofficial/Farsi.isl' -OutFile $env:FARSI_TMP"
if errorlevel 1 (echo [ERROR] Could not download the Farsi translation from the Inno Setup source repository.& pause & exit /b 1)
%PY% -c "from pathlib import Path; import re,sys,unicodedata; p=Path(r'packaging/Farsi.download.isl'); b=p.read_bytes(); s=b.decode('utf-8-sig'); s=unicodedata.normalize('NFC',s); checks={'size':len(b)>10000,'LangOptions':'[LangOptions]' in s,'Messages':'[Messages]' in s,'Farsi language ID':re.search(r'(?im)^LanguageID\s*=\s*\$0429\s*$',s) is not None,'RTL':re.search(r'(?im)^RightToLeft\s*=\s*yes\s*$',s) is not None,'Persian text':any('\u0600'<=c<='\u06ff' for c in s)}; print('Farsi translation validation:', checks); sys.exit(0 if all(checks.values()) else 1)"
if errorlevel 1 (del /q "%FARSI_TMP%" 2>nul & echo [ERROR] Downloaded Farsi translation failed validation. Existing translation was not overwritten.& pause & exit /b 1)
move /y "%FARSI_TMP%" "packaging\Farsi.isl" >nul
if errorlevel 1 (echo [ERROR] Could not install the validated Farsi translation file.& pause & exit /b 1)
echo Farsi translation validated and staged successfully.

echo [6/6] Creating ONE Windows Setup EXE...
"%ISCC%" /Qp "/DPayloadDir=%APPDIR%" packaging\installer.iss
if errorlevel 1 (echo [ERROR] Inno Setup compilation failed.& pause & exit /b 1)
if not exist build\installer\LocalChat-Setup-2.0.5.exe (echo [ERROR] Setup EXE was not created.& pause & exit /b 1)

echo.
echo BUILD COMPLETE
 echo Installer: %CD%\build\installer\LocalChat-Setup-2.0.5.exe
 echo This is the single-file installer to distribute to other Windows PCs.
echo No portable ZIP is produced by this build script.
pause
