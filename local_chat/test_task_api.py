"""HTTP API for long tasks, with a fake Ollama: create → progress → pause/resume → download (all formats)."""
import io
import json
import re
import time
import unittest
from unittest.mock import patch
from urllib.parse import quote

from local_chat import server as srv
from local_chat import task_service as ts
from local_chat.test_docengine import make_book
from local_chat.test_features import FeatureTestCase


class TaskApiTests(FeatureTestCase):
    def setUp(self):
        super().setUp()
        self.fake = {"calls": 0, "ctx": [], "pause_after": None, "fail_on": None}

        def ollama_post(path, payload, timeout=120):
            self.fake["calls"] += 1
            self.fake["ctx"].append(payload["options"]["num_ctx"])
            content = payload["messages"][-1]["content"]
            if self.fake["pause_after"] and self.fake["calls"] == self.fake["pause_after"]:
                srv._task_service().manager.pause(self.task_id)
            body = content.split("متن برای ترجمه:\n\n", 1)[1]
            lines = [ln if re.match(r"^\s*(#|\||```|\[\^|$)", ln) else "ترجمه: " + ln for ln in body.split("\n")]
            return {"message": {"content": "\n".join(lines) + "\n<<TERMS>>\nGPU = پردازندهٔ گرافیکی"}}

        self.patches2 = [
            patch.object(srv, "_ollama_post", ollama_post),
            patch.object(srv, "APP_DATA_DIR", srv.Path(self.tmp.name)),
            patch.object(srv, "TASK_SERVICE", None),
            patch.object(srv, "_model_context_capacity", lambda model: 16384),
            patch.object(srv, "_model_capabilities", lambda model: ["completion"]),
        ]
        for item in self.patches2:
            item.start()
        self.task_id = None

    def tearDown(self):
        service = srv.TASK_SERVICE
        if service is not None:
            service.stop_all(5)
        for item in reversed(self.patches2):
            item.stop()
        super().tearDown()

    def create(self, name="book.txt", data=None, **options):
        data = data if data is not None else make_book(chapters=6, sections=2, paragraphs=4).encode()
        opts = {"kind": "translate", "model": "text:8b", "target": "fa", **options}
        status, body = self.call("POST", "/api/tasks", raw=data, headers={
            "X-Filename": quote(name), "X-Task-Options": quote(json.dumps(opts)), "Content-Type": "application/octet-stream"})
        payload = json.loads(body)
        if status == 201:
            self.task_id = payload["task"]["id"]
        return status, payload

    def wait_status(self, wanted, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            _, body = self.call("GET", f"/api/tasks/{self.task_id}")
            task = json.loads(body)["task"]
            if task["status"] in wanted:
                return task
            time.sleep(0.05)
        self.fail(f"task never reached {wanted}; last: {task['status']}")

    def test_create_runs_and_lists_with_progress(self):
        status, payload = self.create()
        self.assertEqual(status, 201, payload)
        self.assertGreater(payload["task"]["total"], 1)
        task = self.wait_status({"completed"})
        self.assertEqual((task["done"], task["failed"], task["percent"]), (task["total"], 0, 100.0))
        self.assertEqual(task["glossary"], {"GPU": "پردازندهٔ گرافیکی"})
        self.assertEqual(len(task["chunks"]), task["total"])
        _, body = self.call("GET", "/api/tasks")
        self.assertEqual([t["id"] for t in json.loads(body)["tasks"]], [self.task_id])

    def test_chunk_size_follows_model_capacity(self):
        with patch.object(srv, "_model_context_capacity", lambda model: 4096):
            self.create(name="a.txt")
            small = srv._task_service().manager.get(self.task_id)["params"]
        self.wait_status({"completed"})
        self.assertLessEqual(small["num_ctx"], 4096)
        with patch.object(srv, "_model_context_capacity", lambda model: 65536):
            self.create(name="b.txt")
            large = srv._task_service().manager.get(self.task_id)["params"]
        self.assertLess(small["max_tokens"], large["max_tokens"])
        self.assertTrue(all(c <= 16384 for c in self.fake["ctx"]))         # never above the cap / capacity window

    def test_download_all_formats_and_incomplete_refusal(self):
        self.create()
        self.wait_status({"completed"})
        for fmt, marker in (("md", b"# "), ("txt", "ترجمه".encode()), ("docx", b"PK"), ("pdf", b"%PDF")):
            status, body = self.call("GET", f"/api/tasks/{self.task_id}/download?format={fmt}")
            self.assertEqual(status, 200, (fmt, body[:200]))
            self.assertIn(marker, body[:5000] if fmt in ("docx", "pdf") else body)
        status, body = self.call("GET", f"/api/tasks/{self.task_id}/download?format=exe")
        self.assertEqual(status, 400)

    def test_pause_resume_and_incomplete_download_is_blocked(self):
        self.fake["pause_after"] = 3
        self.task_id = "pending"
        status, payload = self.create()
        self.task_id = payload["task"]["id"]
        task = self.wait_status({"paused"})
        self.assertLess(task["done"], task["total"])
        self.assertTrue(task["resumable"])
        status, body = self.call("GET", f"/api/tasks/{self.task_id}/download?format=md")
        self.assertEqual(status, 409)
        self.assertIn("خروجی کامل نیست", json.loads(body)["error"])
        status, body = self.call("GET", f"/api/tasks/{self.task_id}/download?format=md&partial=1")
        self.assertEqual(status, 200)
        self.assertIn("ترجمه/پردازش نشده", body.decode())
        self.fake["pause_after"] = None
        status, _ = self.call("POST", f"/api/tasks/{self.task_id}/resume", {})
        self.assertEqual(status, 200)
        self.assertEqual(self.wait_status({"completed"})["done"], task["total"])

    def test_regenerate_and_cancel_endpoints(self):
        self.create()
        self.wait_status({"completed"})
        before = self.fake["calls"]
        status, body = self.call("POST", f"/api/tasks/{self.task_id}/regenerate", {"chunks": [2]})
        self.assertEqual(status, 200)
        self.wait_status({"completed"})
        self.assertEqual(self.fake["calls"] - before, 1)                   # exactly one chunk was redone
        status, _ = self.call("POST", f"/api/tasks/{self.task_id}/regenerate", {"chunks": []})
        self.assertEqual(status, 400)
        status, body = self.call("POST", f"/api/tasks/{self.task_id}/delete", {})
        self.assertEqual(json.loads(body)["task"], {"deleted": True})
        self.assertEqual(self.call("GET", f"/api/tasks/{self.task_id}")[0], 404)

    def test_validation_errors(self):
        self.assertEqual(self.create(kind="translate", model="")[0], 400)
        self.assertEqual(self.create(kind="hack")[0], 400)
        self.assertEqual(self.create(name="x.docx", data=b"PK\x03\x04bad")[0], 415)
        self.assertEqual(self.create(name="x.bin", data=bytes(range(256)) * 4)[0], 415)
        status, _ = self.call("POST", "/api/tasks", raw=b"x", headers={"X-Task-Options": "{bad"})
        self.assertEqual(status, 400)
        self.assertEqual(self.call("GET", "/api/tasks/nope")[0], 404)
        self.assertEqual(self.call("POST", "/api/tasks/nope/pause", {})[0], 404)

    def test_ui_panel_is_served(self):
        status, body = self.call("GET", "/static/tasks_panel.js")
        self.assertEqual(status, 200)
        self.assertIn(b"/api/tasks", body)
        status, body = self.call("GET", "/")
        self.assertIn(b"tasksButton", body)
        self.assertIn(b"tasks_panel.js", body)

    def test_creator_verification_is_required(self):
        with patch.object(srv, "_registered_creator", lambda: None):
            self.assertEqual(self.call("GET", "/api/tasks")[0], 403)
            status, _ = self.call("POST", "/api/tasks", raw=b"abc", headers={"X-Task-Options": "{}"})
            self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
