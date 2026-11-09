"""Local-only ParsVoice XTTS inference worker.

Loads the user's local Hugging Face ParsVoice checkpoint directly instead of
using Coqui's model registry (ParsVoice is not registered there). The worker
is isolated in the dedicated audio virtual environment and binds to loopback.
"""
import argparse
import io
import json
import os
import sys
import threading
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import gcd
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import persian_speech  # noqa: E402  (shared, standard-library-only text preparation)

MODEL_ID = "MohammadJRanjbar/parsvoice-xtts-v2"
WORKER_ENGINE = "parsvoice-local-xtts-fa-v2"
WORKER_PROTOCOL_VERSION = 3
LATENT_CACHE = {}
CHUNK_PAUSE_SECONDS = 0.18


def _float_env(name, default, low, high):
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return min(max(value, low), high)
MODEL = None
MODEL_LOCK = threading.Lock()
DATA_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "LocalChat"
DEFAULT_MODEL_DIR = DATA_DIR / "voice" / "models" / "ParsVoice-XTTS"


def _compatible_vocab(model_dir: Path) -> Path:
    """Convert the repository's BPE merge pairs to tokenizers' accepted form.

    The source vocab.json is left untouched. ParsVoice's repository currently
    stores each BPE merge as a two-item JSON array, while tokenizers 0.19.x
    expects a space-separated string.
    """
    source = model_dir / "vocab.json"
    compatible = model_dir / "vocab_compatible.json"
    if not source.is_file():
        raise FileNotFoundError(f"فایل واژگان مدل پیدا نشد: {source}")
    if compatible.is_file() and compatible.stat().st_mtime >= source.stat().st_mtime:
        return compatible

    import json as _json
    from tokenizers import Tokenizer

    with source.open("r", encoding="utf-8") as handle:
        payload = _json.load(handle)
    model_data = payload.get("model")
    if not isinstance(model_data, dict) or not isinstance(model_data.get("merges"), list):
        raise ValueError("ساختار BPE فایل vocab.json قابل تشخیص نیست.")
    model_data["merges"] = [
        " ".join(item) if isinstance(item, list) and len(item) == 2 else item
        for item in model_data["merges"]
    ]
    # Validate before writing so an invalid converted file is never left behind.
    Tokenizer.from_str(_json.dumps(payload, ensure_ascii=False))
    temporary = compatible.with_suffix(".json.tmp")
    temporary.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(compatible)
    return compatible


def _install_persian_tokenizer_support():
    """Add the missing Persian text-cleaning branch to Coqui XTTS 0.22.0.

    ParsVoice's config and BPE vocabulary contain Persian, but the upstream
    VoiceBpeTokenizer.preprocess_text implementation in Coqui 0.22.0 does not
    route `fa` to any cleaner and raises NotImplementedError. The model's
    vocabulary accepts raw Persian Unicode, so use basic_cleaners (which
    preserves Persian code points) for this language only.
    """
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
    from TTS.tts.utils.text.cleaners import basic_cleaners

    current = VoiceBpeTokenizer.preprocess_text
    if getattr(current, "_localchat_persian_enabled", False):
        return

    def preprocess_text_with_persian(self, txt, lang):
        normalized_lang = lang.split("-")[0]
        if normalized_lang == "fa":
            return basic_cleaners(txt)
        return current(self, txt, lang)

    preprocess_text_with_persian._localchat_persian_enabled = True
    VoiceBpeTokenizer.preprocess_text = preprocess_text_with_persian


def _resolve_model_dir() -> Path:
    """Find a complete local ParsVoice model without downloading or moving files."""
    configured = os.environ.get("AIRESEARCH_PARVOICE_DIR", "").strip().strip('"')
    project_root = Path(__file__).resolve().parent.parent
    roots = []
    if configured:
        roots.append(Path(os.path.expandvars(configured)).expanduser())
    roots.extend([
        DEFAULT_MODEL_DIR,
        project_root / "models" / "ParsVoice-XTTS",
        project_root / "models" / "parsvoice-xtts-v2",
    ])
    # The dedicated ParsVoice venv may live beside the model repository rather
    # than beside the extracted LocalChat app. For example:
    # G:\agentgptversion\.venv-parsvoice\Scripts\python.exe
    # G:\agentgptversion\models\ParsVoice-XTTS\{config.json,model.pth}
    # Derive the repository root from the interpreter actually running this
    # worker; do not confuse the venv directory with the model directory.
    try:
        interpreter = Path(sys.executable).resolve()
        for parent in interpreter.parents:
            if parent.name.casefold() in {".venv-parsvoice", "venv-parsvoice"}:
                repository_root = parent.parent
                roots.extend([
                    repository_root / "models" / "ParsVoice-XTTS",
                    repository_root / "models" / "parsvoice-xtts-v2",
                ])
                break
    except OSError:
        pass
    roots.append(Path.home() / ".cache" / "huggingface" / "hub" / "models--MohammadJRanjbar--parsvoice-xtts-v2")
    seen, checked = set(), []
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        if str(root).casefold() in seen:
            continue
        seen.add(str(root).casefold())
        checked.append(root)
        if (root / "config.json").is_file() and (root / "model.pth").is_file():
            return root
        # Support extracted archives and Hugging Face snapshot directories.
        if root.is_dir():
            try:
                for pattern in ("*/config.json", "*/*/config.json"):
                    for config in root.glob(pattern):
                        if (config.parent / "model.pth").is_file():
                            return config.parent
            except OSError:
                pass
    locations = "\n".join(f"  - {root}" for root in checked)
    raise FileNotFoundError(
        "پوشه معتبر مدل ParsVoice پیدا نشد؛ هر دو فایل config.json و model.pth باید در پوشه مدل باشند. "
        "مسیرهای بررسی‌شده:\n" + locations +
        "\nدر تنظیمات صوتی، مسیر پوشه‌ای را وارد کنید که این دو فایل مستقیماً داخل آن هستند."
    )


def _load_local_model(device: str):
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    _install_persian_tokenizer_support()

    model_dir = _resolve_model_dir()
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "model.pth"
    vocab_path = _compatible_vocab(model_dir)
    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(model_dir),
        checkpoint_path=str(checkpoint_path),
        vocab_path=str(vocab_path),
        eval=True,
        use_deepspeed=False,
    )
    model.to(device)
    return model


def load_model():
    global MODEL
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL
        import torch

        device_setting = os.environ.get("AIRESEARCH_TTS_DEVICE", "auto").strip().lower()
        if device_setting not in {"auto", "cuda", "cpu"}:
            device_setting = "auto"
        if device_setting == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("AIRESEARCH_TTS_DEVICE=cuda تنظیم شده اما CUDA در دسترس نیست.")
        device = "cuda" if device_setting == "cuda" or (
            device_setting == "auto" and torch.cuda.is_available()
        ) else "cpu"
        try:
            MODEL = _load_local_model(device)
        except RuntimeError as exc:
            is_oom = any(marker in str(exc).casefold() for marker in (
                "out of memory", "cuda error: out of memory", "cublas_status_alloc_failed"
            ))
            if device != "cuda" or not is_oom:
                raise
            torch.cuda.empty_cache()
            MODEL = _load_local_model("cpu")
        return MODEL


def _safe_load_audio(audiopath, sampling_rate):
    """Read and resample reference audio without torchaudio/torchcodec."""
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    audio, original_rate = sf.read(str(audiopath), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError("فایل صدای مرجع خالی است.")
    audio = audio.mean(axis=1)
    if original_rate != sampling_rate:
        divisor = gcd(int(original_rate), int(sampling_rate))
        audio = resample_poly(
            audio,
            int(sampling_rate) // divisor,
            int(original_rate) // divisor,
        )
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    return torch.from_numpy(audio).unsqueeze(0)


def _conditioning(model, speaker_wav):
    """Reference-voice latents are expensive; compute them once per reference file."""
    path = Path(speaker_wav)
    key = (str(path), path.stat().st_mtime_ns)
    cached = LATENT_CACHE.get(key)
    if cached is None:
        LATENT_CACHE.clear()
        cached = model.get_conditioning_latents(
            audio_path=[str(path)],
            gpt_cond_len=model.config.gpt_cond_len,
            max_ref_length=model.config.max_ref_len,
            sound_norm_refs=model.config.sound_norm_refs,
        )
        LATENT_CACHE[key] = cached
    return cached


def _synthesize_piece(model, text, gpt_cond_latent, speaker_embedding):
    import numpy as np

    result = model.inference(
        text=text,
        language="fa",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        # Keep sampling conservative for Persian phoneme stability. Raise only
        # through an explicit environment override after listening tests.
        temperature=_float_env("AIRESEARCH_TTS_TEMPERATURE", 0.25, 0.05, 0.65),
        length_penalty=1.0,
        repetition_penalty=_float_env("AIRESEARCH_TTS_REPETITION_PENALTY", 5.0, 1.0, 12.0),
        top_k=30,
        top_p=0.8,
    )
    waveform = result.get("wav")
    if waveform is None:
        raise RuntimeError("مدل ParsVoice خروجی صوتی برنگرداند.")
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().float().cpu().numpy()
    audio = np.asarray(waveform, dtype=np.float32).squeeze()
    if audio.size == 0:
        raise RuntimeError("مدل صوتی خروجی خالی تولید کرد.")
    return audio


def validate_wav_bytes(audio: bytes) -> dict:
    """Reject truncated or non-PCM WAV payloads before returning them to browsers."""
    if not isinstance(audio, (bytes, bytearray)) or len(audio) < 44:
        raise ValueError("خروجی WAV خالی یا کوتاه‌تر از هدر استاندارد است.")
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            info = {
                "channels": wav.getnchannels(),
                "sample_width": wav.getsampwidth(),
                "sample_rate": wav.getframerate(),
                "frames": wav.getnframes(),
                "compression": wav.getcomptype(),
            }
            expected = wav.getnframes() * wav.getnchannels() * wav.getsampwidth()
            if info["compression"] != "NONE":
                raise ValueError("کدک WAV فشرده پشتیبانی نمی‌شود.")
            if info["channels"] not in (1, 2) or info["sample_width"] not in (1, 2, 3, 4):
                raise ValueError("ساختار کانال یا عمق نمونه WAV معتبر نیست.")
            if not 8000 <= info["sample_rate"] <= 96000 or info["frames"] <= 0:
                raise ValueError("نرخ نمونه‌برداری یا تعداد فریم‌های WAV معتبر نیست.")
            if expected <= 0 or len(audio) < 44 + expected:
                raise ValueError("داده صوتی WAV ناقص یا بریده شده است.")
            return info
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"هدر WAV قابل خواندن نیست: {exc}") from exc


def synthesize(text, speaker_wav):
    import numpy as np
    import soundfile as sf
    import TTS.tts.models.xtts as xtts_module

    model = load_model()
    # XTTS 0.22.0 routes audio loading through torchaudio, which in the
    # installed torchaudio build requires torchcodec. Use soundfile instead.
    xtts_module.load_audio = _safe_load_audio

    gpt_cond_latent, speaker_embedding = _conditioning(model, speaker_wav)
    sample_rate = int(getattr(model.config.audio, "output_sample_rate", 24000))
    # Normalize mixed Persian/Latin chat text before XTTS tokenization. Raw
    # Markdown, digits and unknown Latin tokens are a frequent source of garbled speech.
    spoken_text = persian_speech.normalize_for_speech(text)
    if not spoken_text:
        raise ValueError("متن پس از آماده‌سازی برای گفتار خالی شد.")
    # Sentence-sized chunks improve prosody and reduce pronunciation drift.
    pieces = persian_speech.split_for_speech(spoken_text) or [spoken_text]
    pause = np.zeros(int(sample_rate * CHUNK_PAUSE_SECONDS), dtype=np.float32)
    parts = []
    for piece in pieces:
        parts.append(_synthesize_piece(model, piece, gpt_cond_latent, speaker_embedding))
        parts.append(pause)
    audio = np.concatenate(parts)
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalChatTTS/1.2"

    def log_message(self, *_args):
        return

    def send_json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlsplit(self.path).path == "/health":
            requested_dir = os.environ.get("AIRESEARCH_PARVOICE_DIR", "").strip().strip('"')
            resolved_dir = None
            model_files_ok = False
            try:
                resolved = _resolve_model_dir()
                resolved_dir = str(resolved)
                model_files_ok = True
            except Exception:
                pass
            self.send_json(200, {
                "ok": True, "model": MODEL_ID, "engine": WORKER_ENGINE,
                "protocol_version": WORKER_PROTOCOL_VERSION,
                "requested_model_dir": requested_dir,
                "resolved_model_dir": resolved_dir,
                "model_files_ok": model_files_ok,
                "loaded": MODEL is not None,
            })
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if urlsplit(self.path).path != "/synthesize":
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 24_000:
                self.send_json(413, {"error": "درخواست صوتی نامعتبر یا بیش از حد بزرگ است."})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = payload.get("text")
            speaker = payload.get("speaker_wav")
            if not isinstance(text, str) or not text.strip() or len(text) > 6000:
                self.send_json(400, {"error": "متن برای تبدیل به صدا معتبر نیست."})
                return
            if not isinstance(speaker, str) or not Path(speaker).is_file():
                self.send_json(400, {"error": "فایل صدای مرجع پیدا نشد."})
                return
            audio = synthesize(text.strip(), speaker)
            validate_wav_bytes(audio)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(audio)
        except Exception as exc:
            traceback.print_exc()
            self.send_json(500, {"error": f"{type(exc).__name__}: {str(exc)[:800]}"})

    def handle_error(self):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"ParsVoice worker listening on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
