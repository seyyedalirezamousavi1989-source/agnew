import json
from pathlib import Path

import pytest

from local_chat import settings
from local_chat.persian_speech import normalize_for_speech, split_for_speech


def test_tts_settings_persist_and_switch_provider(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "settings_path", lambda: path)
    saved = settings.set_tts_settings({
        "provider": "edge_tts", "voice": "fa-IR-FaridNeural",
        "rate": "+10%", "fallback_local": True,
    })
    assert saved["voice"] == "fa-IR-FaridNeural"
    assert json.loads(path.read_text(encoding="utf-8"))["tts"]["fallback_local"] is True
    switched = settings.set_tts_settings({"provider": "local_parsvoice"})
    assert switched["voice"] == "reference"


def test_tts_settings_reject_invalid_voice_and_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "settings_path", lambda: tmp_path / "settings.json")
    with pytest.raises(ValueError):
        settings.set_tts_settings({"provider": "edge_tts", "voice": "not-a-voice"})
    with pytest.raises(ValueError):
        settings.set_tts_settings({"provider": "edge_tts", "voice": "fa-IR-DilaraNeural", "rate": "+99%"})


def test_persian_speech_normalization_and_chunking():
    spoken = normalize_for_speech("سلام، نسخه 1.9.2 و API آماده است.")
    assert "سلام" in spoken and "یک" in spoken
    chunks = split_for_speech(spoken)
    assert chunks and all(isinstance(chunk, str) for chunk in chunks)


def test_audio_workers_are_present():
    package = Path(__file__).parent
    assert (package / "tts_worker.py").is_file()
    assert (package / "stt_worker.py").is_file()


def test_parsvoice_model_dir_resolves_configured_flat_folder(tmp_path, monkeypatch):
    from local_chat import tts_worker

    model_dir = tmp_path / "ParsVoice-XTTS"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.pth").write_bytes(b"checkpoint")
    monkeypatch.setenv("AIRESEARCH_PARVOICE_DIR", str(model_dir))
    assert tts_worker._resolve_model_dir() == model_dir.resolve()


def test_parsvoice_model_dir_resolves_nested_snapshot(tmp_path, monkeypatch):
    from local_chat import tts_worker

    snapshot = tmp_path / "cache" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.pth").write_bytes(b"checkpoint")
    monkeypatch.setenv("AIRESEARCH_PARVOICE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(tts_worker, "DEFAULT_MODEL_DIR", tmp_path / "missing-default")
    assert tts_worker._resolve_model_dir() == snapshot.resolve()



def test_parsvoice_model_dir_is_derived_from_dedicated_venv(tmp_path, monkeypatch):
    from local_chat import tts_worker

    repo = tmp_path / "agentgptversion"
    venv_python = repo / ".venv-parsvoice" / "Scripts" / "python.exe"
    model_dir = repo / "models" / "ParsVoice-XTTS"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.pth").write_bytes(b"checkpoint")
    monkeypatch.setenv("AIRESEARCH_PARVOICE_DIR", str(tmp_path / "missing-configured-path"))
    monkeypatch.setattr(tts_worker, "DEFAULT_MODEL_DIR", tmp_path / "missing-default")
    monkeypatch.setattr(tts_worker.sys, "executable", str(venv_python))

    assert tts_worker._resolve_model_dir() == model_dir.resolve()

def test_parsvoice_model_dir_error_lists_required_files(tmp_path, monkeypatch):
    from local_chat import tts_worker

    monkeypatch.setenv("AIRESEARCH_PARVOICE_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(tts_worker, "DEFAULT_MODEL_DIR", tmp_path / "missing-default")
    with pytest.raises(FileNotFoundError, match="config.json و model.pth"):
        tts_worker._resolve_model_dir()


def test_tts_worker_health_rejects_legacy_protocol(monkeypatch):
    from local_chat import server

    monkeypatch.setattr(server, "_worker_health", lambda _port: {
        "ok": True, "engine": "parsvoice-local-xtts-fa-v2",
        "protocol_version": 2, "requested_model_dir": "C:/models/ParsVoice-XTTS",
    })
    assert server._worker_ok(8775, "parsvoice-local-xtts-fa-v2", "C:/models/ParsVoice-XTTS") is False


def test_tts_worker_health_rejects_stale_model_path(monkeypatch):
    from local_chat import server

    monkeypatch.setattr(server, "_worker_health", lambda _port: {
        "ok": True, "engine": "parsvoice-local-xtts-fa-v2",
        "protocol_version": server.TTS_WORKER_PROTOCOL,
        "requested_model_dir": "C:/old/ParsVoice-XTTS",
    })
    assert server._worker_ok(8775, "parsvoice-local-xtts-fa-v2", "G:/models/ParsVoice-XTTS") is False


def test_tts_worker_health_accepts_current_protocol_and_path(monkeypatch):
    from local_chat import server

    monkeypatch.setattr(server, "_worker_health", lambda _port: {
        "ok": True, "engine": "parsvoice-local-xtts-fa-v2",
        "protocol_version": server.TTS_WORKER_PROTOCOL,
        "requested_model_dir": "G:/models/ParsVoice-XTTS",
    })
    assert server._worker_ok(8775, "parsvoice-local-xtts-fa-v2", "G:/models/ParsVoice-XTTS") is True


def test_launcher_version_matches_chat_server_version():
    from local_chat import launcher, server

    assert launcher.APP_VERSION == server.APP_VERSION


def test_voice_python_candidates_include_configured_and_existing_user_venv(monkeypatch):
    from local_chat import server

    monkeypatch.setenv("AIRESEARCH_TTS_PYTHON", r"D:\custom\.venv-parsvoice\Scripts\python.exe")
    candidates = [str(path) for path in server._voice_python_candidates("AIRESEARCH_TTS_PYTHON")]
    assert candidates[0].endswith(r"D:\custom\.venv-parsvoice\Scripts\python.exe")
    assert any(path.endswith(r"G:\agentgptversion\.venv-parsvoice\Scripts\python.exe") for path in candidates)


def test_parsvoice_wav_validator_accepts_pcm_wav():
    import io
    import wave
    from local_chat.tts_worker import validate_wav_bytes

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 240)
    info = validate_wav_bytes(buffer.getvalue())
    assert info["sample_rate"] == 24000 and info["frames"] == 240


def test_parsvoice_wav_validator_rejects_truncated_payload():
    from local_chat.tts_worker import validate_wav_bytes

    with pytest.raises(ValueError, match="WAV"):
        validate_wav_bytes(b"RIFF\x00\x00\x00\x00WAVE")
