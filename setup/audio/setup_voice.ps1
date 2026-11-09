$ErrorActionPreference = "Stop"

$AppData = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME "AppData\Local" }
$VoiceRoot = Join-Path $AppData "LocalChat\voice"
$Venv = Join-Path $VoiceRoot ".venv-parsvoice"
$Python = Join-Path $Venv "Scripts\python.exe"
New-Item -ItemType Directory -Force -Path $VoiceRoot | Out-Null

if (-not (Test-Path $Python)) {
    Write-Host "Creating dedicated ParsVoice environment at $Venv ..."
    py -3.10 -m venv $Venv
}
if (-not (Test-Path $Python)) { throw "Python 3.10 was not found. Install Python 3.10 and retry." }

Write-Host "Updating pip build tools..."
& $Python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Could not update pip/setuptools/wheel." }

Write-Host "Installing Coqui TTS..."
& $Python -m pip install TTS
if ($LASTEXITCODE -ne 0) { throw "TTS installation failed. Check Microsoft C++ Build Tools and Python 3.10 compatibility." }

Write-Host "Installing Edge TTS online Persian voices..."
& $Python -m pip install --upgrade edge-tts
if ($LASTEXITCODE -ne 0) { throw "edge-tts installation failed." }

Write-Host "Installing Hugging Face CLI..."
& $Python -m pip install huggingface_hub
if ($LASTEXITCODE -ne 0) { throw "Could not install huggingface_hub." }

& $Python -c "from TTS.api import TTS; print('Coqui TTS import OK')"
if ($LASTEXITCODE -ne 0) { throw "TTS was installed but could not be imported." }

Write-Host ""
Write-Host "Voice environment is ready: $Venv"
Write-Host "Model directory: $(Join-Path $VoiceRoot 'models\ParsVoice-XTTS')"
Write-Host "Accept model access at https://huggingface.co/MohammadJRanjbar/ParsVoice-XTTS"
Write-Host "To install STT, run setup_stt.ps1 from this folder."
Write-Host "Set AIRESEARCH_TTS_PYTHON only if you want to override the default interpreter."
