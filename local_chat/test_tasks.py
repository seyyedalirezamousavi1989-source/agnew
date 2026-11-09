"""Long-running tasks: persistence, pause/resume/cancel, crash recovery (real SIGKILL), selective retry, order."""
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

from local_chat import docengine as de
from local_chat import tasks as tk
from local_chat.test_docengine import make_book

ROOT = Path(__file__).resolve().parent.parent
BOOK = make_book(chapters=6, sections=2, paragraphs=4).encode()
PARAMS = {"max_tokens": 500}


def echo(chunk, ctx):
    return tk.ChunkResult(f"T[{chunk.index}]::" + chunk.text, summary=f"s{chunk.index}")


def wait_for(predicate, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TaskTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = tk.TaskManager(self.tmp.name)

    def tearDown(self):
        self.manager.stop_all(5)
        self.tmp.cleanup()

    def new_task(self, data=BOOK, **params):
        return self.manager.create("translate", "book.txt", data, {**PARAMS, **params})

    def assert_complete_and_ordered(self, task_id):
        rows = self.manager.chunk_rows(task_id)
        self.assertEqual([r["idx"] for r in rows], list(range(len(rows))))
        self.assertTrue(all(r["status"] == "done" for r in rows))
        for row in rows:
            self.assertTrue(row["output"].startswith(f"T[{row['idx']}]::"))
        return rows


class LifecycleTests(TaskTestCase):
    def test_runs_to_completion_with_ordered_complete_output(self):
        task_id = self.new_task()
        self.manager.start(task_id, echo)
        progress = self.manager.wait(task_id, 30)
        self.assertEqual(progress["status"], tk.COMPLETED)
        self.assertEqual(progress["percent"], 100.0)
        rows = self.assert_complete_and_ordered(task_id)
        rebuilt = "\n\n".join(r["output"].split("::", 1)[1] for r in rows)
        doc = de.load_document(BOOK, "book.txt")
        chunks = de.chunk_document(doc, 500)
        self.assertEqual(rebuilt, "\n\n".join(c.text for c in chunks))
        self.assertEqual(len(rows), progress["total"])

    def test_progress_fields_and_events(self):
        task_id = self.new_task()
        gate = threading.Event()

        def slow(chunk, ctx):
            gate.wait(5)
            return "x"

        self.manager.start(task_id, slow)
        self.assertTrue(wait_for(lambda: self.manager.progress(task_id)["running"] >= 1))
        progress = self.manager.progress(task_id)
        self.assertIn(progress["status"], (tk.RUNNING,))
        self.assertEqual(progress["stage"], "processing")
        self.assertGreaterEqual(progress["elapsed_seconds"], 0)
        gate.set()
        self.manager.wait(task_id, 30)
        events = [e["message"] for e in self.manager.events(task_id)]
        self.assertTrue(any("پردازش شروع شد" in m for m in events))

    def test_pause_then_resume_never_reprocesses_done_chunks(self):
        task_id = self.new_task()
        calls = []
        gate = threading.Event()

        def processor(chunk, ctx):
            calls.append(chunk.index)
            if len(calls) == 3:
                self.manager.pause(task_id)
                gate.set()
            return tk.ChunkResult(f"T[{chunk.index}]::x", summary=f"s{chunk.index}")

        self.manager.start(task_id, processor)
        gate.wait(10)
        progress = self.manager.wait(task_id, 30)
        self.assertEqual(progress["status"], tk.PAUSED)
        done_before = progress["done"]
        self.assertGreaterEqual(done_before, 3)
        self.assertLess(done_before, progress["total"])
        calls.clear()
        self.manager.start(task_id, processor.__class__ and (lambda c, x: tk.ChunkResult(f"T[{c.index}]::y", summary="s") if calls.append(c.index) is None else None))
        progress = self.manager.wait(task_id, 30)
        self.assertEqual(progress["status"], tk.COMPLETED)
        self.assertEqual(len(calls), progress["total"] - done_before)          # only the remaining chunks
        self.assertEqual(sorted(calls), sorted(set(calls)))                    # none twice
        self.assert_complete_and_ordered(task_id)

    def test_cancel_keeps_completed_results_and_can_be_resumed(self):
        task_id = self.new_task()
        count = {"n": 0}

        def processor(chunk, ctx):
            count["n"] += 1
            if count["n"] == 2:
                self.manager.cancel(task_id)
            return tk.ChunkResult(f"T[{chunk.index}]::x")

        self.manager.start(task_id, processor)
        progress = self.manager.wait(task_id, 30)
        self.assertEqual(progress["status"], tk.CANCELLED)
        self.assertGreaterEqual(progress["done"], 2)
        self.assertTrue(progress["resumable"])
        kept = {r["idx"]: r["output"] for r in self.manager.chunk_rows(task_id) if r["status"] == "done"}
        self.manager.start(task_id, echo)
        self.manager.wait(task_id, 30)
        rows = {r["idx"]: r for r in self.manager.chunk_rows(task_id)}
        for index, output in kept.items():
            self.assertEqual(rows[index]["output"], output)                    # untouched by the resume
        self.assertEqual(self.manager.progress(task_id)["status"], tk.COMPLETED)

    def test_cancel_of_idle_task_and_double_start_are_safe(self):
        task_id = self.new_task()
        self.assertEqual(self.manager.cancel(task_id)["status"], tk.CANCELLED)
        gate = threading.Event()
        self.manager.start(task_id, lambda c, x: (gate.wait(5), "x")[1])
        with self.assertRaises(tk.TaskError):
            self.manager.start(task_id, echo)
        gate.set()
        self.manager.wait(task_id, 30)
        with self.assertRaises(tk.TaskError):
            self.manager.start(task_id, echo)                                  # completed: use regenerate


class FailureTests(TaskTestCase):
    def test_failed_chunk_does_not_stop_others_and_can_be_retried_selectively(self):
        task_id = self.new_task()
        bad = {3}

        def processor(chunk, ctx):
            if chunk.index in bad:
                raise RuntimeError("model crashed")
            return tk.ChunkResult(f"T[{chunk.index}]::ok")

        self.manager.start(task_id, processor)
        progress = self.manager.wait(task_id, 30)
        self.assertEqual(progress["status"], tk.PARTIAL)
        self.assertEqual(progress["failed"], 1)
        self.assertEqual(progress["done"], progress["total"] - 1)
        self.assertIn("model crashed", progress["last_chunk_error"]["error"])
        row = self.manager.chunk_rows(task_id)[3]
        self.assertEqual((row["status"], row["attempts"]), ("failed", tk.MAX_ATTEMPTS))
        bad.clear()
        self.assertEqual(self.manager.retry_failed(task_id), 1)
        self.manager.start(task_id, processor)
        self.assertEqual(self.manager.wait(task_id, 30)["status"], tk.COMPLETED)

    def test_transient_error_is_retried_inside_the_run(self):
        task_id = self.new_task()
        seen = {}

        def flaky(chunk, ctx):
            seen[chunk.index] = seen.get(chunk.index, 0) + 1
            if chunk.index == 1 and seen[1] == 1:
                raise TimeoutError("busy")
            return tk.ChunkResult("ok")

        self.manager.start(task_id, flaky)
        self.assertEqual(self.manager.wait(task_id, 30)["status"], tk.COMPLETED)
        self.assertEqual(seen[1], 2)

    def test_unreadable_source_marks_task_failed_with_reason(self):
        task_id = self.manager.create("translate", "x.docx", b"PK\x03\x04junk", PARAMS)
        self.manager.start(task_id, echo)
        progress = self.manager.wait(task_id, 15)
        self.assertEqual(progress["status"], tk.FAILED)
        self.assertTrue(progress["error"])

    def test_regenerate_redoes_only_chosen_chunks_and_keeps_old_output(self):
        task_id = self.new_task()
        self.manager.start(task_id, echo)
        self.manager.wait(task_id, 30)
        before = {r["idx"]: r["output"] for r in self.manager.chunk_rows(task_id)}
        redo = []
        self.manager.regenerate(task_id, [2, 4])
        self.assertEqual(self.manager.progress(task_id)["status"], tk.PARTIAL)
        self.manager.start(task_id, lambda c, x: (redo.append(c.index), tk.ChunkResult(f"T[{c.index}]::NEW"))[1])
        self.manager.wait(task_id, 30)
        self.assertEqual(sorted(redo), [2, 4])
        rows = {r["idx"]: r for r in self.manager.chunk_rows(task_id)}
        self.assertTrue(rows[2]["output"].endswith("NEW"))
        self.assertEqual(rows[2]["previous_output"], before[2])
        self.assertEqual(rows[0]["output"], before[0])


class ContextAndParallelTests(TaskTestCase):
    def test_summary_and_glossary_carry_between_chunks(self):
        task_id = self.new_task()
        seen = {}

        def processor(chunk, ctx):
            seen[chunk.index] = (ctx.previous_summary(), dict(ctx.glossary()))
            ctx.remember_terms({"Ollama": "اولاما"} if chunk.index == 0 else {"Ollama": "اُلاما"})
            return tk.ChunkResult(f"T[{chunk.index}]::x", summary=f"خلاصهٔ {chunk.index}")

        self.manager.start(task_id, processor)
        self.manager.wait(task_id, 30)
        self.assertEqual(seen[0], ("", {}))
        self.assertEqual(seen[1][0], "خلاصهٔ 0")
        self.assertEqual(seen[3][1], {"Ollama": "اولاما"})                    # first translation wins: terminology is stable
        self.manager.add_glossary(task_id, {"Ollama": "الاما"}, locked=True)
        self.assertEqual(self.manager.glossary(task_id)["Ollama"], "الاما")

    def test_parallel_workers_keep_order_and_completeness(self):
        task_id = self.new_task()

        def processor(chunk, ctx):
            time.sleep(0.01 * (chunk.index % 3))
            return tk.ChunkResult(f"T[{chunk.index}]::p")

        self.manager.start(task_id, processor, workers=4)
        self.assertEqual(self.manager.wait(task_id, 30)["status"], tk.COMPLETED)
        self.assert_complete_and_ordered(task_id)

    def test_changed_plan_after_completed_chunks_is_refused(self):
        task_id = self.new_task()
        self.manager.start(task_id, lambda c, x: tk.ChunkResult("x") if c.index < 2 else (_ for _ in ()).throw(tk.TaskCancelled()))
        self.manager.wait(task_id, 30)
        self.manager._set(task_id, params={"max_tokens": 2500})
        self.manager.start(task_id, echo)
        progress = self.manager.wait(task_id, 30)
        self.assertEqual(progress["status"], tk.FAILED)
        self.assertIn("طرح چانک‌بندی", progress["error"])


class CrashRecoveryTests(unittest.TestCase):
    def test_sigkill_mid_task_then_restart_resumes_without_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            script = textwrap.dedent(f"""
                import sys, time
                sys.path.insert(0, {str(ROOT)!r})
                from local_chat import tasks as tk
                m = tk.TaskManager({directory!r})
                data = open({directory!r} + "/book.txt", "rb").read()
                tid = m.create("translate", "book.txt", data, {{"max_tokens": 500}})
                open({directory!r} + "/tid", "w").write(tid)
                def slow(chunk, ctx):
                    time.sleep(0.15)
                    return tk.ChunkResult("T[%d]::old" % chunk.index, summary="s")
                m.start(tid, slow)
                time.sleep(120)
            """)
            Path(directory, "book.txt").write_bytes(BOOK)
            proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                self.assertTrue(wait_for(lambda: Path(directory, "tid").exists(), 20))
                task_id = Path(directory, "tid").read_text()
                db_path = Path(directory) / tk.DB_NAME

                def done_count():
                    import sqlite3
                    try:
                        db = sqlite3.connect(db_path, timeout=5)
                        n = db.execute("SELECT COUNT(*) FROM chunks WHERE status='done'").fetchone()[0]
                        db.close()
                        return n
                    except sqlite3.Error:
                        return 0

                self.assertTrue(wait_for(lambda: done_count() >= 3, 30))
                os.kill(proc.pid, signal.SIGKILL)                              # no clean shutdown, no finally blocks
                proc.wait(10)
            finally:
                if proc.poll() is None:
                    proc.kill()
            manager = tk.TaskManager(directory)                                # «restart»
            try:
                progress = manager.progress(task_id)
                self.assertEqual(progress["status"], tk.INTERRUPTED)
                self.assertTrue(progress["resumable"])
                survived = {r["idx"] for r in manager.chunk_rows(task_id) if r["status"] == "done"}
                self.assertGreaterEqual(len(survived), 3)
                self.assertEqual(manager.chunk_counts(task_id).get("running", 0), 0)
                redone = []
                manager.start(task_id, lambda c, x: (redone.append(c.index), tk.ChunkResult(f"T[{c.index}]::new"))[1])
                final = manager.wait(task_id, 60)
                self.assertEqual(final["status"], tk.COMPLETED)
                self.assertTrue(survived.isdisjoint(redone))                    # finished work is never repeated
                rows = manager.chunk_rows(task_id)
                self.assertEqual([r["idx"] for r in rows], list(range(final["total"])))
                self.assertTrue(all(r["output"].endswith(("old", "new")) for r in rows))
                self.assertGreater(final["elapsed_seconds"], 0)
            finally:
                manager.stop_all(5)


class DatabaseTests(TaskTestCase):
    def test_integrity_foreign_keys_and_cascade_delete(self):
        task_id = self.new_task()
        self.manager.start(task_id, echo)
        self.manager.wait(task_id, 30)
        report = self.manager.db.integrity_report()
        self.assertTrue(report["ok"], report)
        self.assertEqual(set(report["tables"]) >= {"tasks", "chunks", "task_events", "glossary"}, True)
        self.assertTrue(report["tables"]["chunks"]["indexes"])
        stats = self.manager.storage_stats(task_id)
        self.assertGreater(stats["outputs_bytes"], 0)
        self.manager.delete(task_id)
        with self.manager.db.read() as db:
            for table in ("chunks", "task_events", "glossary"):
                self.assertEqual(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)
        self.assertFalse((self.manager.files_dir / task_id).exists())


if __name__ == "__main__":
    unittest.main()
