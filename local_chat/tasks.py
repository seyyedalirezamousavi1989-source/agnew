"""Long-running task manager: persistent, pausable, resumable chunked jobs (book translation, large-document
analysis, codebase inspection).

What it guarantees
* Every finished chunk is committed to ``agent_tasks.sqlite3`` BEFORE the next one starts – a crash, a power cut
  or a closed app loses at most the chunks that were in flight.
* Pause / resume / cancel are cooperative and never discard completed chunks.
* After a restart a task that was running is marked ``interrupted`` and ``resume`` continues with only the
  chunks that are not done – nothing is reprocessed, nothing is skipped (the chunk plan is re-derived from the
  stored source file and checked against the stored chunk digests).
* A failed chunk is recorded (error, attempts) and does not stop the others; ``retry_failed`` /
  ``regenerate`` redo exactly the chosen chunks.
* Progress, stage, elapsed time (summed across runs), ETA and an event log are always available.

The model call itself is a plain callable ``processor(chunk, ctx) -> str | ChunkResult`` supplied by the
caller, so this module has no dependency on Ollama.
"""
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from . import docengine
    from .dbutil import Database
except ImportError:                     # pragma: no cover
    import docengine
    from dbutil import Database

log = logging.getLogger("localchat.tasks")

SCHEMA_VERSION = 1
DB_NAME = "agent_tasks.sqlite3"
MAX_WORKERS = 4
MAX_ATTEMPTS = 2

QUEUED, PREPARING, RUNNING, PAUSED = "queued", "preparing", "running", "paused"
CANCELLED, COMPLETED, PARTIAL, FAILED, INTERRUPTED = "cancelled", "completed", "partial", "failed", "interrupted"
ACTIVE = (PREPARING, RUNNING)
RESUMABLE = (PAUSED, INTERRUPTED, CANCELLED, PARTIAL, QUEUED, FAILED)


class TaskError(Exception):
    pass


class TaskCancelled(Exception):
    """Raised inside a processor when the task was cancelled/paused while a chunk was in flight."""


class ChunkResult:
    def __init__(self, output, summary="", meta=None):
        self.output, self.summary, self.meta = output, summary, meta or {}


class TaskContext:
    """What a processor may use besides the chunk itself."""

    def __init__(self, manager, task, chunk, stop_event):
        self.manager, self.task, self.chunk, self.stop = manager, task, chunk, stop_event
        self.params = task["params"]

    def check(self):
        if self.stop.is_set():
            raise TaskCancelled()

    def previous_summary(self):
        """Summary the previous chunk left behind (consistent terminology / story so far)."""
        return self.manager.carry_for(self.task["id"], self.chunk.index)

    def glossary(self):
        return self.manager.glossary(self.task["id"])

    def remember_terms(self, pairs):
        self.manager.add_glossary(self.task["id"], pairs)


_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
        stage TEXT NOT NULL DEFAULT '', params TEXT NOT NULL DEFAULT '{}', source_name TEXT NOT NULL DEFAULT '',
        source_path TEXT NOT NULL DEFAULT '', plan TEXT NOT NULL DEFAULT '{}',
        total_chunks INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '',
        elapsed_seconds REAL NOT NULL DEFAULT 0, run_started REAL, owner TEXT NOT NULL DEFAULT '',
        checkpoint TEXT NOT NULL DEFAULT '{}',
        created_at REAL NOT NULL, updated_at REAL NOT NULL, finished_at REAL)""",
    """CREATE TABLE IF NOT EXISTS chunks (
        task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE, idx INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', digest TEXT NOT NULL DEFAULT '', tokens_in INTEGER NOT NULL DEFAULT 0,
        output TEXT, summary TEXT NOT NULL DEFAULT '', meta TEXT NOT NULL DEFAULT '{}', previous_output TEXT,
        attempts INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '', seconds REAL NOT NULL DEFAULT 0,
        updated_at REAL NOT NULL DEFAULT 0, PRIMARY KEY (task_id, idx))""",
    """CREATE TABLE IF NOT EXISTS task_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
        ts REAL NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS glossary (
        task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE, term TEXT NOT NULL, translation TEXT NOT NULL,
        locked INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (task_id, term))""",
    "CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(task_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_events_task ON task_events(task_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, updated_at DESC)",
)


def _row(row):
    if row is None:
        return None
    data = dict(row)
    for key in ("params", "plan", "checkpoint"):
        try:
            data[key] = json.loads(data.get(key) or "{}")
        except ValueError:
            data[key] = {}
    return data


class TaskManager:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.files_dir = self.root / "tasks"
        self.db = Database(self.root / DB_NAME)
        self.boot_id = uuid.uuid4().hex
        self._runners = {}                # task id -> {"thread", "stop", "reason"}
        self._lock = threading.RLock()
        self._migrate()
        self.recover()

    # ------------------------------------------------------------------ schema / recovery
    def _migrate(self):
        existing = self.db.has_table("tasks")
        version = self.db.version() if existing else 0
        if existing and version < SCHEMA_VERSION:
            self.db.backup(f"v{version}")
        with self.db.tx() as db:
            for statement in _SCHEMA:
                db.execute(statement)
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def recover(self):
        """Tasks that were running when the app stopped become ``interrupted`` (resumable, nothing lost)."""
        now = time.time()
        with self.db.tx() as db:
            stale = db.execute("SELECT id, run_started FROM tasks WHERE status IN (?, ?, ?) AND owner != ?",
                               (PREPARING, RUNNING, "pausing", self.boot_id)).fetchall()
            for row in stale:
                spent = max(0.0, now - row["run_started"]) if row["run_started"] else 0.0
                db.execute("UPDATE tasks SET status=?, stage=?, elapsed_seconds=elapsed_seconds+?, run_started=NULL, "
                           "updated_at=? WHERE id=?", (INTERRUPTED, "interrupted", spent, now, row["id"]))
                db.execute("UPDATE chunks SET status='pending', error='interrupted' WHERE task_id=? AND status='running'",
                           (row["id"],))
                db.execute("INSERT INTO task_events (task_id, ts, level, message) VALUES (?,?,?,?)",
                           (row["id"], now, "warn", "برنامه بسته یا متوقف شد؛ کار قابل ادامه است."))
        return [row["id"] for row in stale]

    # ------------------------------------------------------------------ small helpers
    def _event(self, task_id, level, message, db=None):
        args = (task_id, time.time(), level, str(message)[:2000])
        if db is not None:
            db.execute("INSERT INTO task_events (task_id, ts, level, message) VALUES (?,?,?,?)", args)
            return
        with self.db.tx() as conn:
            conn.execute("INSERT INTO task_events (task_id, ts, level, message) VALUES (?,?,?,?)", args)

    def _set(self, task_id, **fields):
        fields["updated_at"] = time.time()
        for key in ("params", "plan", "checkpoint"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key], ensure_ascii=False)
        names = ", ".join(f"{k}=?" for k in fields)
        with self.db.tx() as db:
            db.execute(f"UPDATE tasks SET {names} WHERE id=?", [*fields.values(), task_id])

    def get(self, task_id):
        with self.db.read() as db:
            return _row(db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())

    def list(self, limit=50):
        with self.db.read() as db:
            rows = db.execute("SELECT id FROM tasks ORDER BY updated_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [self.progress(r["id"]) for r in rows]

    def events(self, task_id, limit=100):
        with self.db.read() as db:
            rows = db.execute("SELECT ts, level, message FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT ?",
                              (task_id, int(limit))).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------ create
    def create(self, kind, source_name, data, params=None, title=""):
        """Store the source file and register the task (nothing is processed yet)."""
        if not isinstance(data, (bytes, bytearray)):
            raise TaskError("داده‌ی فایل معتبر نیست.")
        task_id = uuid.uuid4().hex
        folder = self.files_dir / task_id
        folder.mkdir(parents=True, exist_ok=True)
        suffix = Path(source_name).suffix[:12] or ".bin"
        source = folder / f"source{re.sub(r'[^A-Za-z0-9.]', '', suffix)}"
        source.write_bytes(bytes(data))
        now = time.time()
        with self.db.tx() as db:
            db.execute("INSERT INTO tasks (id, kind, title, status, stage, params, source_name, source_path, "
                       "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (task_id, kind, title or source_name, QUEUED, "queued",
                        json.dumps(params or {}, ensure_ascii=False), source_name, str(source), now, now))
            self._event(task_id, "info", f"کار «{title or source_name}» ساخته شد.", db)
        return task_id

    # ------------------------------------------------------------------ plan (deterministic chunking)
    def _build_chunks(self, task):
        doc = docengine.load_document(task["source_path"], task["source_name"])
        chunks = docengine.chunk_document(doc, int(task["params"].get("max_tokens") or 1500))
        report = docengine.verify_chunks(doc, chunks)
        if not report["ok"]:
            raise TaskError("بررسی یکپارچگی چانک‌ها ناموفق بود: " + "; ".join(report["problems"][:3]))
        return doc, chunks

    def prepare(self, task_id):
        """Chunk the stored source once and store the plan; on a resume the plan is re-derived and compared."""
        task = self.get(task_id)
        if task is None:
            raise TaskError("کار پیدا نشد.")
        doc, chunks = self._build_chunks(task)
        digests = [c.digest for c in chunks]
        with self.db.read() as db:
            stored = db.execute("SELECT idx, digest FROM chunks WHERE task_id=? ORDER BY idx", (task_id,)).fetchall()
        if stored:
            if [r["digest"] for r in stored] != digests:
                with self.db.read() as db:
                    done = db.execute("SELECT COUNT(*) FROM chunks WHERE task_id=? AND status='done'", (task_id,)).fetchone()[0]
                if done:
                    raise TaskError("طرح چانک‌بندی با نسخهٔ ذخیره‌شده فرق می‌کند؛ برای حفظ نتایج، تنظیمات را تغییر نده.")
                with self.db.tx() as db:
                    db.execute("DELETE FROM chunks WHERE task_id=?", (task_id,))
                stored = []
        if not stored:
            now = time.time()
            with self.db.tx() as db:
                db.executemany("INSERT INTO chunks (task_id, idx, digest, tokens_in, updated_at) VALUES (?,?,?,?,?)",
                               [(task_id, c.index, c.digest, c.tokens, now) for c in chunks])
            self._set(task_id, total_chunks=len(chunks),
                      plan={"chunks": len(chunks), "outline": [list(o[:2]) for o in doc.outline()[:200]],
                            "stats": doc.stats(), "notes": doc.notes})
            self._event(task_id, "info", f"سند به {len(chunks)} بخش تقسیم شد ({doc.stats()['tokens']} توکن).")
        return doc, chunks

    # ------------------------------------------------------------------ run control
    def start(self, task_id, processor, workers=1):
        """Run (or continue) a task in a background thread.  Only chunks that are not done are processed."""
        with self._lock:
            if task_id in self._runners:
                raise TaskError("این کار هم‌اکنون در حال اجراست.")
            task = self.get(task_id)
            if task is None:
                raise TaskError("کار پیدا نشد.")
            if task["status"] == COMPLETED:
                raise TaskError("این کار تمام شده است؛ برای بازتولید یک بخش از regenerate استفاده کن.")
            stop = threading.Event()
            runner = {"stop": stop, "reason": None}
            thread = threading.Thread(target=self._run, args=(task_id, processor, max(1, min(MAX_WORKERS, int(workers))), runner),
                                      name=f"task-{task_id[:6]}", daemon=True)
            runner["thread"] = thread
            self._runners[task_id] = runner
            self._set(task_id, status=PREPARING, stage="preparing", owner=self.boot_id, run_started=time.time(), error="")
            thread.start()
        return task_id

    def wait(self, task_id, timeout=None):
        runner = self._runners.get(task_id)
        if runner:
            runner["thread"].join(timeout)
        return self.progress(task_id)

    def pause(self, task_id):
        return self._request_stop(task_id, "pause")

    def cancel(self, task_id):
        runner = self._runners.get(task_id)
        if runner:
            return self._request_stop(task_id, "cancel")
        task = self.get(task_id)
        if task and task["status"] in (QUEUED, PAUSED, INTERRUPTED, PARTIAL, FAILED):
            self._set(task_id, status=CANCELLED, stage="cancelled")
            self._event(task_id, "info", "کار لغو شد؛ نتایج کامل‌شده نگه داشته شد.")
        return self.progress(task_id)

    def _request_stop(self, task_id, reason):
        with self._lock:
            runner = self._runners.get(task_id)
            if runner is None:
                raise TaskError("این کار در حال اجرا نیست.")
            runner["reason"] = reason
            runner["stop"].set()
        return self.progress(task_id)

    def retry_failed(self, task_id, indexes=None):
        with self.db.tx() as db:
            marks, args = "", [task_id]
            if indexes:
                marks = " AND idx IN (%s)" % ",".join("?" for _ in indexes)
                args += [int(i) for i in indexes]
            n = db.execute(f"UPDATE chunks SET status='pending', attempts=0, error='' WHERE task_id=? AND status='failed'{marks}",
                           args).rowcount
            self._event(task_id, "info", f"{n} بخش ناموفق برای تلاش دوباره علامت خورد.", db)
        return n

    def regenerate(self, task_id, indexes):
        """Redo specific chunks even if they were done (the previous output is kept in ``previous_output``)."""
        indexes = [int(i) for i in indexes]
        with self.db.tx() as db:
            for i in indexes:
                db.execute("UPDATE chunks SET status='pending', previous_output=output, attempts=0, error='' "
                           "WHERE task_id=? AND idx=?", (task_id, i))
            self._event(task_id, "info", f"بازتولید بخش‌های {indexes} درخواست شد.", db)
        task = self.get(task_id)
        if task and task["status"] == COMPLETED:
            self._set(task_id, status=PARTIAL, stage="regenerate", finished_at=None)
        return len(indexes)

    # ------------------------------------------------------------------ the worker
    def _run(self, task_id, processor, workers, runner):
        stop = runner["stop"]
        ticker_done = threading.Event()

        def heartbeat():
            while not ticker_done.wait(5):
                try:
                    self._set(task_id, owner=self.boot_id)
                except Exception:
                    pass

        threading.Thread(target=heartbeat, daemon=True).start()
        final_status, error = None, ""
        try:
            doc, chunks = self.prepare(task_id)
            by_index = {c.index: c for c in chunks}
            with self.db.read() as db:
                pending = [r["idx"] for r in db.execute(
                    "SELECT idx FROM chunks WHERE task_id=? AND status IN ('pending','running') ORDER BY idx", (task_id,)).fetchall()]
            self._set(task_id, status=RUNNING, stage="processing")
            self._event(task_id, "info", f"پردازش شروع شد: {len(pending)} بخش باقی‌مانده، {workers} کارگر.")
            task = self.get(task_id)

            def work(index):
                for _attempt in range(MAX_ATTEMPTS):               # a bad chunk gets one immediate retry, then is marked failed
                    if stop.is_set():
                        return index, None
                    self._chunk_state(task_id, index, "running")
                    started = time.time()
                    try:
                        result = processor(by_index[index], TaskContext(self, task, by_index[index], stop))
                        if stop.is_set() and result is None:
                            raise TaskCancelled()
                        if not isinstance(result, ChunkResult):
                            result = ChunkResult(str(result))
                        self._chunk_done(task_id, index, result, time.time() - started)
                        return index, True
                    except TaskCancelled:
                        self._chunk_state(task_id, index, "pending")
                        return index, None
                    except Exception as exc:                              # a bad chunk never stops the others
                        log.exception("chunk %s of task %s failed", index, task_id)
                        if self._chunk_failed(task_id, index, exc, time.time() - started) == "failed":
                            return index, False
                return index, False

            if workers == 1:
                for index in pending:
                    if stop.is_set():
                        break
                    work(index)
            else:
                with ThreadPoolExecutor(workers) as pool:
                    futures, queue = [], list(pending)
                    while (queue or futures) and not stop.is_set():
                        while queue and len(futures) < workers and not stop.is_set():
                            futures.append(pool.submit(work, queue.pop(0)))
                        done = [f for f in futures if f.done()]
                        for f in done:
                            futures.remove(f)
                        if not done:
                            time.sleep(0.02)
                    for f in futures:
                        f.result()
            counts = self.chunk_counts(task_id)
            if runner["reason"] == "cancel":
                final_status = CANCELLED
            elif runner["reason"] == "pause":
                final_status = PAUSED
            elif counts.get("done", 0) == counts["total"]:
                final_status = COMPLETED
            elif counts.get("pending", 0) + counts.get("running", 0):
                final_status = PAUSED
            else:
                final_status = PARTIAL
        except Exception as exc:
            log.exception("task %s failed", task_id)
            final_status, error = FAILED, f"{type(exc).__name__}: {exc}"
            try:
                self._event(task_id, "error", error)
            except Exception:
                pass
        finally:
            ticker_done.set()
            task = self.get(task_id) or {}
            spent = max(0.0, time.time() - (task.get("run_started") or time.time()))
            counts = self.chunk_counts(task_id) if task else {}
            self._set(task_id, status=final_status or INTERRUPTED, stage=(final_status or INTERRUPTED),
                      elapsed_seconds=(task.get("elapsed_seconds") or 0.0) + spent, run_started=None, error=error,
                      finished_at=time.time() if final_status == COMPLETED else None)
            if final_status:
                self._event(task_id, "info" if final_status != FAILED else "error",
                            f"وضعیت: {final_status} — {counts.get('done', 0)}/{counts.get('total', 0)} بخش انجام شد"
                            + (f"، {counts.get('failed', 0)} ناموفق" if counts.get("failed") else ""))
            with self._lock:
                self._runners.pop(task_id, None)

    # ------------------------------------------------------------------ chunk persistence
    def _chunk_state(self, task_id, index, status):
        with self.db.tx() as db:
            db.execute("UPDATE chunks SET status=?, updated_at=? WHERE task_id=? AND idx=?",
                       (status, time.time(), task_id, index))

    def _chunk_done(self, task_id, index, result, seconds):
        with self.db.tx() as db:
            db.execute("UPDATE chunks SET status='done', output=?, summary=?, meta=?, error='', attempts=attempts+1, "
                       "seconds=?, updated_at=? WHERE task_id=? AND idx=?",
                       (result.output, result.summary or "", json.dumps(result.meta, ensure_ascii=False), seconds,
                        time.time(), task_id, index))

    def _chunk_failed(self, task_id, index, exc, seconds):
        message = f"{type(exc).__name__}: {exc}"[:1000]
        with self.db.tx() as db:
            attempts = db.execute("SELECT attempts FROM chunks WHERE task_id=? AND idx=?", (task_id, index)).fetchone()["attempts"] + 1
            status = "pending" if attempts < MAX_ATTEMPTS else "failed"
            db.execute("UPDATE chunks SET status=?, error=?, attempts=?, seconds=seconds+?, updated_at=? WHERE task_id=? AND idx=?",
                       (status, message, attempts, seconds, time.time(), task_id, index))
            self._event(task_id, "warn" if status == "pending" else "error",
                        f"بخش {index + 1}: {message}" + (" (تلاش دوباره)" if status == "pending" else ""), db)
        return status

    # ------------------------------------------------------------------ reading results
    def chunk_counts(self, task_id):
        with self.db.read() as db:
            rows = db.execute("SELECT status, COUNT(*) AS n FROM chunks WHERE task_id=? GROUP BY status", (task_id,)).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        counts["total"] = sum(counts.values())
        return counts

    def chunk_rows(self, task_id, with_output=True):
        cols = "idx, status, digest, tokens_in, summary, meta, attempts, error, seconds" + (", output, previous_output" if with_output else "")
        with self.db.read() as db:
            rows = db.execute(f"SELECT {cols} FROM chunks WHERE task_id=? ORDER BY idx", (task_id,)).fetchall()
        return [dict(r) for r in rows]

    def carry_for(self, task_id, index):
        """Summary of the nearest finished chunk before ``index`` (what the model should remember)."""
        with self.db.read() as db:
            row = db.execute("SELECT summary FROM chunks WHERE task_id=? AND idx<? AND status='done' AND summary != '' "
                             "ORDER BY idx DESC LIMIT 1", (task_id, index)).fetchone()
        return row["summary"] if row else ""

    def glossary(self, task_id):
        with self.db.read() as db:
            rows = db.execute("SELECT term, translation, locked FROM glossary WHERE task_id=? ORDER BY term", (task_id,)).fetchall()
        return {r["term"]: r["translation"] for r in rows}

    def add_glossary(self, task_id, pairs, locked=False):
        """First translation of a term wins unless the entry is replaced explicitly (locked entries never change)."""
        with self.db.tx() as db:
            for term, translation in dict(pairs).items():
                term, translation = str(term).strip(), str(translation).strip()
                if term and translation:
                    db.execute("INSERT INTO glossary (task_id, term, translation, locked) VALUES (?,?,?,?) "
                               "ON CONFLICT(task_id, term) DO UPDATE SET translation=excluded.translation, locked=excluded.locked "
                               "WHERE ?", (task_id, term, translation, int(locked), int(locked)))

    def progress(self, task_id):
        task = self.get(task_id)
        if task is None:
            return None
        counts = self.chunk_counts(task_id)
        total, done = counts.get("total", 0), counts.get("done", 0)
        running = task["status"] in ACTIVE
        elapsed = task["elapsed_seconds"] + (max(0.0, time.time() - task["run_started"]) if running and task["run_started"] else 0.0)
        with self.db.read() as db:
            seconds = db.execute("SELECT AVG(seconds) FROM chunks WHERE task_id=? AND status='done' AND seconds>0", (task_id,)).fetchone()[0]
            last_error = db.execute("SELECT idx, error FROM chunks WHERE task_id=? AND error != '' AND status='failed' "
                                    "ORDER BY updated_at DESC LIMIT 1", (task_id,)).fetchone()
        remaining = total - done
        return {
            "id": task_id, "kind": task["kind"], "title": task["title"], "status": task["status"], "stage": task["stage"],
            "total": total, "done": done, "failed": counts.get("failed", 0), "pending": counts.get("pending", 0),
            "running": counts.get("running", 0), "percent": round(100.0 * done / total, 1) if total else 0.0,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(seconds * remaining, 1) if seconds and remaining and running else None,
            "error": task["error"], "last_chunk_error": dict(last_error) if last_error else None,
            "resumable": (task["status"] in RESUMABLE and done < total) or task["status"] in (PARTIAL, INTERRUPTED),
            "created_at": task["created_at"], "updated_at": task["updated_at"],
        }

    def storage_stats(self, task_id):
        folder = self.files_dir / task_id
        files = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) if folder.exists() else 0
        with self.db.read() as db:
            row = db.execute("SELECT COALESCE(SUM(LENGTH(output)),0) AS out, COALESCE(SUM(LENGTH(previous_output)),0) AS prev, "
                             "COALESCE(SUM(LENGTH(summary)),0) AS summ FROM chunks WHERE task_id=?", (task_id,)).fetchone()
        return {"source_files_bytes": files, "outputs_bytes": row["out"], "previous_outputs_bytes": row["prev"],
                "summaries_bytes": row["summ"], "db_bytes": self.db.path.stat().st_size}

    def delete(self, task_id):
        if task_id in self._runners:
            raise TaskError("ابتدا کار را متوقف کن.")
        with self.db.tx() as db:
            db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        shutil.rmtree(self.files_dir / task_id, ignore_errors=True)

    def stop_all(self, timeout=10):
        """For a clean app shutdown: pause every runner and wait; state stays resumable."""
        for task_id in list(self._runners):
            try:
                self.pause(task_id)
            except TaskError:
                pass
        for runner in list(self._runners.values()):
            runner["thread"].join(timeout)
