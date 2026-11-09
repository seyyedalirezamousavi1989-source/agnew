# Audio migration for LocalChat 1.9.2

- Restored isolated local worker scripts from Stage11: `stt_worker.py`, `tts_worker.py`; Persian text normalization is in `persian_speech.py`.
- Added local microphone recording + faster-whisper transcription endpoint `/api/voice/transcribe`.
- Added per-assistant-message audio playback and auto-play toggle.
- Added persistent TTS settings (`edge_tts` or `local_parsvoice`), Persian Dilara/Farid voices, speech rate, model directory, optional local fallback.
- Edge TTS requires internet and `pip install -r requirements-audio.txt`; it is an online dependency, not an offline guarantee.
- STT requires `faster-whisper` in the audio Python environment. ParsVoice requires its own Coqui TTS environment, model files, and reference WAV.
- Default paths use `%LOCALAPPDATA%\LocalChat\voice`; set `AIRESEARCH_TTS_PYTHON` / `AIRESEARCH_STT_PYTHON` to point to the correct Python executables if using separate environments.

This integration was syntax-checked in the build environment only. Full model inference, Windows microphone permissions, CUDA, and real Edge service calls must be verified on the target PC.
