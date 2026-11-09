"""Loopback-only local speech-to-text worker using faster-whisper."""
import argparse
import base64
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = None
MODEL_LOCK = threading.Lock()
MODEL_SIZE = os.environ.get("LOCALCHAT_STT_MODEL", "small").strip() or "small"
DEVICE_SETTING = os.environ.get("LOCALCHAT_STT_DEVICE", "auto").strip().lower()


def _resolve_device():
    if DEVICE_SETTING in {"cpu", "cuda"}:
        return DEVICE_SETTING
    try:
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def load_model():
    global MODEL
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL
        from faster_whisper import WhisperModel
        device = _resolve_device()
        compute_type = "float16" if device == "cuda" else "int8"
        try:
            MODEL = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
        except Exception:
            if device != "cuda" or DEVICE_SETTING == "cuda":
                raise
            MODEL = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        return MODEL


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalChatSTT/1.0"

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
        if self.path.split("?", 1)[0] == "/health":
            try:
                import faster_whisper  # noqa: F401
                installed = True
            except Exception:
                installed = False
            self.send_json(200, {
                "ok": installed,
                "model": MODEL_SIZE,
                "loaded": MODEL is not None,
                "device": _resolve_device(),
            })
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/transcribe":
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 22 * 1024 * 1024:
                self.send_json(413, {"error": "درخواست صوتی نامعتبر یا بیش از حد بزرگ است."})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            encoded = payload.get("data")
            extension = payload.get("extension", ".webm")
            if extension not in {".webm", ".wav", ".m4a", ".mp3"}:
                extension = ".webm"
            if not isinstance(encoded, str) or not encoded:
                self.send_json(400, {"error": "دادهٔ صوتی ارسال نشده است."})
                return
            raw = base64.b64decode(encoded, validate=True)
            if not raw or len(raw) > 15 * 1024 * 1024:
                self.send_json(413, {"error": "حجم فایل صوتی نامعتبر است."})
                return
            model = load_model()
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp:
                temp.write(raw)
                audio_path = temp.name
            try:
                segments, info = model.transcribe(
                    audio_path,
                    language="fa",
                    task="transcribe",
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
            finally:
                try:
                    Path(audio_path).unlink(missing_ok=True)
                except OSError:
                    pass
            self.send_json(200, {
                "text": text,
                "language": getattr(info, "language", "fa"),
                "model": MODEL_SIZE,
            })
        except Exception as exc:
            self.send_json(500, {"error": f"{type(exc).__name__}: {str(exc)[:700]}"})

    def handle_error(self):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Local STT worker listening on 127.0.0.1:{args.port}; model={MODEL_SIZE}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
