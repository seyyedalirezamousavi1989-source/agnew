"""HTTP-facing service for long tasks: creates/controls tasks and builds the model callable.

Kept out of ``server.py`` on purpose (that file is already huge); the server only routes to this class.
It owns no state beyond the ``TaskManager`` and takes every Ollama interaction as an injected callable.
"""
import re
import threading
import time
from urllib.parse import quote

try:
    from . import context_manager, docengine, docwriter, pipeline, tasks
except ImportError:                     # pragma: no cover
    import context_manager, docengine, docwriter, pipeline, tasks

KINDS = ("translate", "summarize")
MAX_TASK_UPLOAD = 128 * 1024 * 1024
MAX_PARAM_TEXT = 600
DEFAULT_TASK_CTX = 16384
MAX_WORKERS = 2                         # a local GPU rarely gains from more parallel requests


class ServiceError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _clean(value, limit=MAX_PARAM_TEXT):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value or "")).strip()[:limit]


class TaskService:
    def __init__(self, root_getter, ollama_post, capacity_getter, thinking_getter=lambda model: False):
        self._root_getter = root_getter
        self._ollama_post = ollama_post
        self._capacity = capacity_getter
        self._thinking = thinking_getter
        self._manager = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ plumbing
    @property
    def manager(self):
        with self._lock:
            if self._manager is None:
                self._manager = tasks.TaskManager(self._root_getter())
            return self._manager

    def stop_all(self, timeout=5):
        if self._manager is not None:
            self._manager.stop_all(timeout)

    def generator(self, model, num_ctx, num_predict, temperature=0.2):
        """``generate(messages) -> str`` bound to one model and window (no streaming: results are stored whole)."""
        thinking = bool(self._thinking(model))

        def generate(messages):
            body = {"model": model, "stream": False, "keep_alive": "20m", "messages": messages,
                    "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict}}
            if thinking:
                body["think"] = False
            data = self._ollama_post("/api/chat", body, timeout=900)
            return ((data.get("message") or {}).get("content") or "").strip()
        return generate

    def window(self, params):
        """(num_ctx, num_predict, chunk_tokens) for the chosen model – the plan follows the model's real capacity."""
        capacity = self._capacity(params["model"])
        # default: use what the model really supports, but cap at 16k (VRAM and speed); the user can override
        requested = int(params.get("num_ctx") or min(capacity or 8192, DEFAULT_TASK_CTX))
        ctx = context_manager.effective_context(requested, capacity)
        predict = max(256, min(4096, int(ctx * 0.35)))
        chunk = docengine.plan_chunk_tokens(ctx, predict, task=params["kind"])
        return ctx, predict, chunk

    # ------------------------------------------------------------------ create / control
    def create(self, name, data, options):
        kind = _clean(options.get("kind"), 20) or "translate"
        model = _clean(options.get("model"), 200)
        if kind not in KINDS:
            raise ServiceError("نوع کار باید translate یا summarize باشد.")
        if not model:
            raise ServiceError("مدل مشخص نشده است.")
        if not data or len(data) > MAX_TASK_UPLOAD:
            raise ServiceError("فایل خالی یا بزرگ‌تر از حد مجاز (۱۲۸ مگابایت) است.", 413)
        try:                                              # fail early, with a clear reason, before storing anything
            probe = docengine.load_document(bytes(data), name)
        except docengine.DocumentError as exc:
            raise ServiceError(str(exc), 415)
        if not probe.blocks:
            raise ServiceError("در این فایل متنی برای پردازش پیدا نشد.", 422)
        params = {"kind": kind, "model": model, "target": _clean(options.get("target"), 8) or "fa",
                  "style": _clean(options.get("style")), "num_ctx": int(options.get("num_ctx") or 0),
                  "workers": max(1, min(MAX_WORKERS, int(options.get("workers") or 1)))}
        ctx, predict, chunk = self.window(params)
        params.update({"num_ctx": ctx, "num_predict": predict, "max_tokens": chunk})
        task_id = self.manager.create(kind, name, bytes(data), params, title=_clean(options.get("title"), 120) or name)
        self.manager.prepare(task_id)                      # chunk now so the caller sees the plan immediately
        if options.get("autostart", True):
            self.start(task_id)
        return self.manager.progress(task_id)

    def processor(self, task):
        params = task["params"]
        generate = self.generator(params["model"], params["num_ctx"], params["num_predict"])
        if task["kind"] == "summarize":
            return pipeline.make_summarize_processor(generate, params.get("target", "fa"))
        return pipeline.make_translate_processor(generate, params.get("target", "fa"), params.get("style", ""))

    def start(self, task_id):
        task = self.manager.get(task_id)
        if task is None:
            raise ServiceError("کار پیدا نشد.", 404)
        try:
            self.manager.start(task_id, self.processor(task), workers=task["params"].get("workers", 1))
        except tasks.TaskError as exc:
            raise ServiceError(str(exc), 409)

    def detail(self, task_id):
        progress = self.manager.progress(task_id)
        if progress is None:
            raise ServiceError("کار پیدا نشد.", 404)
        rows = self.manager.chunk_rows(task_id, with_output=False)
        issues = []
        for row in rows:
            try:
                import json
                for issue in (json.loads(row["meta"] or "{}").get("issues") or []):
                    issues.append({"chunk": row["idx"] + 1, **issue})
            except ValueError:
                pass
        progress["chunks"] = [{"index": r["idx"] + 1, "status": r["status"], "attempts": r["attempts"], "error": r["error"],
                               "tokens": r["tokens_in"]} for r in rows]
        progress["issues"] = issues
        progress["events"] = self.manager.events(task_id, 30)
        progress["storage"] = self.manager.storage_stats(task_id)
        progress["glossary"] = self.manager.glossary(task_id)
        return progress

    def action(self, task_id, action, payload=None):
        payload = payload or {}
        if self.manager.get(task_id) is None:
            raise ServiceError("کار پیدا نشد.", 404)
        try:
            if action in ("start", "resume"):
                self.start(task_id)
            elif action == "pause":
                self.manager.pause(task_id)
            elif action == "cancel":
                self.manager.cancel(task_id)
            elif action == "retry":
                self.manager.retry_failed(task_id, payload.get("chunks"))
                self.start(task_id)
            elif action == "regenerate":
                indexes = [int(i) - 1 for i in (payload.get("chunks") or [])]
                if not indexes:
                    raise ServiceError("شمارهٔ بخش‌ها مشخص نشده است.")
                self.manager.regenerate(task_id, indexes)
                self.start(task_id)
            elif action == "delete":
                self.manager.delete(task_id)
                return {"deleted": True}
            elif action == "glossary":
                pairs = payload.get("terms")
                if not isinstance(pairs, dict) or not pairs:
                    raise ServiceError("واژه‌نامه معتبر نیست.")
                self.manager.add_glossary(task_id, {_clean(k, 60): _clean(v, 60) for k, v in pairs.items()}, locked=True)
            else:
                raise ServiceError("عملیات ناشناخته است.", 404)
        except tasks.TaskError as exc:
            raise ServiceError(str(exc), 409)
        return self.detail(task_id)

    def download(self, task_id, fmt="docx", allow_partial=False):
        task = self.manager.get(task_id)
        if task is None:
            raise ServiceError("کار پیدا نشد.", 404)
        fmt = (fmt or "docx").lower()
        if fmt not in docwriter.FORMATS:
            raise ServiceError("قالب باید txt، md، docx یا pdf باشد.")
        try:
            data, report = pipeline.assemble(self.manager, task_id, fmt, allow_partial=allow_partial)
        except tasks.TaskError as exc:
            raise ServiceError(str(exc), 409)
        except docwriter.WriterUnavailable as exc:
            raise ServiceError(str(exc), 501)
        stem = re.sub(r"[^\w\-.]+", "_", task["title"].rsplit(".", 1)[0], flags=re.U).strip("_") or "output"
        return data, docwriter.FORMATS[fmt][1], f"{stem}.{fmt}", report
