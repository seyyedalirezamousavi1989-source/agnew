$ErrorActionPreference = "Stop"
$AppData = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME "AppData\Local" }
$Python = Join-Path $AppData "LocalChat\voice\.venv-parsvoice\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "محیط صوتی پیدا نشد. ابتدا setup_voice.ps1 را از همین پوشه اجرا کنید." }
Write-Host "Using audio Python: $Python"
& $Python -m pip install --upgrade faster-whisper
if ($LASTEXITCODE -ne 0) { throw "نصب faster-whisper ناموفق بود." }
& $Python -c "from faster_whisper import WhisperModel; print('faster-whisper import OK')"
if ($LASTEXITCODE -ne 0) { throw "faster-whisper نصب شد اما import ناموفق بود." }
Write-Host "ورودی گفتار فارسی آماده است. مدل small در اولین اجرا دانلود می‌شود."
