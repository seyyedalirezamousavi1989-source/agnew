"""End-to-end tests of programming mode: real LocalChat server + a scripted fake Ollama."""
import http.client
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as BaseServer
from pathlib import Path
from unittest.mock import patch

from local_chat import coding_agent, system_tools
from local_chat.server import ChatHandler, ThreadingHTTPServer


class FakeOllama:
    """Tiny Ollama stand-in. `script(kind, payload)` returns what to answer for each request kind."""

    def __init__(self):
        self.requests = []
        self.script = lambda kind, payload: {"content": "ok"}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_):
                pass

            def _json(self, data):
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._json({"models": [{"name": "fake-coder:latest", "size": 4 * 1024**3}]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/show":
                    self._json({"capabilities": ["completion", "tools"]})
                    return
                kind = "plan" if payload.get("format") else ("loop" if payload.get("tools") else "plain")
                outer.requests.append((kind, payload))
                answer = outer.script(kind, payload)
                if kind == "plan":
                    self._json({"message": {"role": "assistant", "content": answer["content"]}, "done": True})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                if answer.get("content"):
                    for piece in answer["content"]:
                        self.wfile.write((json.dumps({"message": {"role": "assistant", "content": piece}, "done": False}) + "\n").encode())
                        self.wfile.flush()
                if answer.get("tool_calls"):
                    self.wfile.write((json.dumps({"message": {"role": "assistant", "content": "", "tool_calls": [
                        {"function": {"name": n, "arguments": a}} for n, a in answer["tool_calls"]]}, "done": False}) + "\n").encode())
                self.wfile.write((json.dumps({"message": {"role": "assistant", "content": ""}, "done": True,
                                              "done_reason": answer.get("done_reason", "stop"), "eval_count": 7}) + "\n").encode())

        self.server = BaseServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def kinds(self):
        return [kind for kind, _ in self.requests]


class Browser:
    def __init__(self, port):
        self.port = port
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request("GET", "/api/session", headers={"Host": f"127.0.0.1:{port}"})
        response = conn.getresponse()
        self.csrf = json.loads(response.read())["csrf"]
        self.cookie = response.getheader("Set-Cookie").split(";")[0]
        conn.close()

    def headers(self, csrf=True):
        headers = {"Content-Type": "application/json", "Cookie": self.cookie, "Origin": f"http://127.0.0.1:{self.port}"}
        if csrf:
            headers["X-LocalChat-CSRF"] = self.csrf
        return headers

    def post(self, path, payload, csrf=True):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request("POST", path, json.dumps(payload), self.headers(csrf))
        response = conn.getresponse()
        data = json.loads(response.read() or b"{}")
        conn.close()
        return response.status, data

    def chat(self, text, history=(), on_event=None, csrf=True, mode="programming"):
        """Stream one chat turn; `on_event(event)` may answer approval cards. Returns (events, text)."""
        messages = list(history) + [{"role": "user", "content": text}]
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        conn.request("POST", "/api/chat", json.dumps({"model": "fake-coder:latest", "messages": messages, "mode": mode}),
                     self.headers(csrf))
        response = conn.getresponse()
        if response.status != 200:
            body = response.read()
            conn.close()
            return [{"status": response.status, "body": body.decode()}], ""
        events, content = [], ""
        for line in response:
            if not line.strip():
                continue
            event = json.loads(line)
            events.append(event)
            content += (event.get("message") or {}).get("content", "") or ""
            if on_event:
                on_event(event)
        conn.close()
        return events, content


class ProgrammingE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ollama = FakeOllama()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.port = cls.server.server_address[1]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "projects"
        root.mkdir()
        self.root = root
        self.patches = [
            patch.object(system_tools, "PROJECTS_ROOT", root),
            patch("local_chat.server.OLLAMA_URL", self.ollama.url),
            patch("local_chat.server._registered_creator", lambda: {"name": "t"}),
        ]
        for item in self.patches:
            item.start()
        from local_chat import server
        server._PROFILE_CACHE.clear()
        server._CAPS_CACHE.clear()
        self.ollama.requests.clear()
        self.browser = Browser(self.port)

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.tmp.cleanup()

    def answer_cards(self, auto_files=True, approve_exec=True, seen=None):
        def handler(event):
            if "plan_approval" in event:
                self.browser.post("/api/chat/plan-decision", {"plan_id": event["plan_approval"]["plan_id"],
                                                              "approved": True, "auto_files": auto_files})
            if "approval_required" in event:
                if seen is not None:
                    seen.append(event["approval_required"]["plan"]["action"])
                path = "/api/system-tools/approve" if approve_exec else "/api/system-tools/reject"
                self.browser.post(path, {"approval_id": event["approval_required"]["approval_id"]})
        return handler

    # ------------------------------------------------------------------ the bug the user reported
    def test_capability_question_builds_nothing_and_offers_no_tools(self):
        self.ollama.script = lambda kind, payload: {"content": ["بله، می‌توانم. ", "چه نوع بازی‌ای؟"]}
        events, text = self.browser.chat("میتونی بازی بسازی؟")
        self.assertEqual(self.ollama.kinds(), ["plain"])
        self.assertNotIn("tools", self.ollama.requests[0][1])
        self.assertIn("بله", text)
        self.assertIn("«بساز»", text)
        self.assertFalse(any("plan_approval" in e or "approval_required" in e for e in events))
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertTrue(events[-1]["done"])

    def test_vague_request_gets_requirement_interview_without_calling_the_planner(self):
        self.ollama.script = lambda kind, payload: {"content": "should not be called"}
        events, text = self.browser.chat("یه بازی شطرنج بساز")
        self.assertEqual(self.ollama.kinds(), [], "the interview is deterministic: no model call, no tools")
        self.assertIn(coding_agent.CLARIFY_HEADER, text)
        self.assertIn("ظاهر و رنگ", text)
        self.assertNotIn("قیمت", text)
        self.assertFalse(any("plan_approval" in e for e in events))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_questions_the_user_already_answered_are_not_asked_again(self):
        questions = coding_agent.interview_questions("یه بازی شطرنج دو نفره با موس گرافیکی سیاه و سفید بساز")
        joined = " ".join(questions)
        self.assertNotIn("ظاهر و رنگ", joined)
        self.assertNotIn("حالت بازی و کنترل", joined)
        self.assertIn("اندازهٔ پنجره", joined)

    def test_detailed_or_no_questions_request_skips_interview(self):
        self.assertEqual(coding_agent.interview_questions("یه بازی مار بساز بدون سوال"), [])
        self.assertEqual(coding_agent.interview_questions("تفاوت list و tuple چیست"), [])

    def test_answer_to_questions_goes_to_plan_without_asking_again(self):
        asked = {"decision": "ask", "questions": ["q؟"]}
        self.ollama.script = lambda kind, payload: {"content": json.dumps(asked)}
        history = [{"role": "user", "content": "یه بازی بساز"},
                   {"role": "assistant", "content": coding_agent.CLARIFY_HEADER + "\n\n1. q؟"}]
        events, text = self.browser.chat("مار با pygame", history=history)
        self.assertEqual(self.ollama.kinds()[:3], ["plan", "plan", "plan"])  # retried with "do not ask again"
        self.assertIn("do NOT ask again", self.ollama.requests[1][1]["messages"][0]["content"])
        self.assertNotIn("پیش از شروع، چند نکته", text)

    # ------------------------------------------------------------------ full build
    def test_full_build_flow_with_auto_files_and_exec_approval(self):
        plan = {"decision": "plan", "title": "بازی حدس عدد", "summary": "s", "project": "guess-game", "steps": [
            {"title": "ساخت پروژه", "detail": "", "tool": "create_project"},
            {"title": "منطق", "detail": "", "tool": "write_file"},
            {"title": "تست", "detail": "", "tool": "venv_test"}]}
        rounds = iter([
            {"content": ["مرحلهٔ ۱ از ۳: ساخت پروژه\n"], "tool_calls": [("create_project", {"name": "guess-game"})]},
            {"tool_calls": [
                ("write_file", {"project": "guess-game", "path": "src/logic.py", "content": "def pick():\n    return 4\n"}),
                ("write_file", {"file_path": "tests/test_logic.py", "project": "guess-game",
                                "content": "import unittest\nfrom src.logic import pick\n\nclass T(unittest.TestCase):\n    def test(self):\n        self.assertEqual(pick(), 4)\n"}),
                ("create_venv", {"project": "guess-game"}),
                ("venv_test", {"project": "guess-game"})]},
            {"content": ["همه چیز آماده است."]},
        ])
        self.ollama.script = lambda kind, payload: ({"content": json.dumps(plan)} if kind == "plan" else next(rounds))
        seen = []
        events, text = self.browser.chat("یک بازی حدس عدد بساز بدون سوال", on_event=self.answer_cards(True, True, seen))
        project = self.root / "guess-game"
        self.assertTrue((project / "src/logic.py").is_file())
        self.assertTrue((project / "tests/test_logic.py").is_file())
        self.assertEqual(seen, ["venv_test"], "auto mode must still ask for execution")
        self.assertIn("همهٔ تست‌ها موفق بودند", text)
        self.assertNotIn("<tool_call", text)
        self.assertTrue(events[-1]["done"])
        loop_requests = [p for k, p in self.ollama.requests if k == "loop"]
        self.assertTrue(all(0 < len(p["tools"]) < len(coding_agent.TOOL_DEFINITIONS) for p in loop_requests))
        self.assertTrue(all({"write_file", "check_project"} <= {t["function"]["name"] for t in p["tools"]} for p in loop_requests))
        self.assertEqual(self.ollama.requests[-1][1]["options"]["num_predict"], 4096)

    def test_without_auto_files_every_write_asks(self):
        plan = {"decision": "plan", "title": "t", "summary": "", "project": "p1", "steps": [
            {"title": "ساخت", "detail": "", "tool": "create_project"}]}
        rounds = iter([{"tool_calls": [("create_project", {"name": "p1"})]}, {"content": ["تمام"]}])
        self.ollama.script = lambda kind, payload: ({"content": json.dumps(plan)} if kind == "plan" else next(rounds))
        seen = []
        self.browser.chat("پروژه p1 را بساز", on_event=self.answer_cards(False, True, seen))
        self.assertEqual(seen, ["create_project"])
        self.assertTrue((self.root / "p1").is_dir())

    def test_rejecting_an_operation_stops_the_chain(self):
        plan = {"decision": "plan", "title": "t", "summary": "", "project": "p2", "steps": [
            {"title": "ساخت", "detail": "", "tool": "create_project"}]}
        rounds = iter([{"tool_calls": [("create_project", {"name": "p2"}), ("create_project", {"name": "p3"})]},
                       {"content": ["نباید برسد"]}])
        self.ollama.script = lambda kind, payload: ({"content": json.dumps(plan)} if kind == "plan" else next(rounds))
        events, text = self.browser.chat("بساز", on_event=self.answer_cards(False, False))
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertIn("رد شد", text)
        self.assertNotIn("نباید برسد", text)

    def test_rejecting_the_plan_runs_no_tool(self):
        plan = {"decision": "plan", "title": "t", "summary": "", "project": "p4", "steps": [
            {"title": "ساخت", "detail": "", "tool": "create_project"}]}
        self.ollama.script = lambda kind, payload: {"content": json.dumps(plan)}

        def handler(event):
            if "plan_approval" in event:
                self.browser.post("/api/chat/plan-decision", {"plan_id": event["plan_approval"]["plan_id"], "approved": False})
        events, text = self.browser.chat("بساز", on_event=handler)
        self.assertEqual(self.ollama.kinds(), ["plan"])
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertIn("تأیید نشد", text)

    def test_stop_button_cancels_while_waiting_for_plan_approval(self):
        plan = {"decision": "plan", "title": "t", "summary": "", "project": "p5", "steps": [
            {"title": "ساخت", "detail": "", "tool": "create_project"}]}
        self.ollama.script = lambda kind, payload: {"content": json.dumps(plan)}

        def handler(event):
            if "plan_approval" in event:
                status, data = self.browser.post("/api/chat/cancel", {})
                self.assertEqual((status, data), (200, {"cancelled": True}))
        started = time.monotonic()
        events, _ = self.browser.chat("بساز", on_event=handler)
        self.assertLess(time.monotonic() - started, 8)
        self.assertEqual(events[-1].get("done_reason"), "cancelled")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_text_format_tool_call_is_executed_and_not_shown(self):
        plan = {"decision": "plan", "title": "t", "summary": "", "project": "p6", "steps": [
            {"title": "ساخت", "detail": "", "tool": "create_project"}]}
        raw = "<tool_call>\n{\"name\": \"create_project\", \"arguments\": {\"name\": \"p6\"}}\n</tool_call>"
        rounds = iter([{"content": ["شروع می‌کنم\n", raw]}, {"content": ["تمام"]}])
        self.ollama.script = lambda kind, payload: ({"content": json.dumps(plan)} if kind == "plan" else next(rounds))
        events, text = self.browser.chat("بساز p6", on_event=self.answer_cards(True))
        self.assertTrue((self.root / "p6").is_dir())
        self.assertNotIn("tool_call", text)
        self.assertIn("شروع می‌کنم", text)

    def test_model_that_never_calls_tools_gets_a_clear_message(self):
        plan = {"decision": "plan", "title": "t", "summary": "", "project": "p7", "steps": [
            {"title": "ساخت", "detail": "", "tool": "create_project"}]}
        self.ollama.script = lambda kind, payload: ({"content": json.dumps(plan)} if kind == "plan" else {"content": ["کد: ..."]})
        events, text = self.browser.chat("بساز p7", on_event=self.answer_cards(True))
        self.assertIn("هیچ ابزاری را فراخوانی نکرد", text)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_plan_output_is_reported_not_silently_ignored(self):
        self.ollama.script = lambda kind, payload: {"content": "not json at all"}
        events, text = self.browser.chat("بساز یک ماشین حساب بدون سوال")
        self.assertEqual(self.ollama.kinds(), ["plan", "plan", "plan"])
        self.assertIn("نتوانستم طرح معتبری", text)

    def test_error_report_is_diagnosed_without_tools(self):
        self.ollama.script = lambda kind, payload: {"content": ["علت: ماژول نصب نیست."]}
        events, text = self.browser.chat("ModuleNotFoundError: No module named 'pygame'")
        self.assertEqual(self.ollama.kinds(), ["plain"])
        self.assertNotIn("tools", self.ollama.requests[0][1])

    # ------------------------------------------------------------------ security
    def test_programming_chat_requires_session_and_csrf(self):
        events, _ = self.browser.chat("hi", csrf=False)
        self.assertEqual(events[0]["status"], 403)
        status, _ = self.browser.post("/api/system-tools/approve", {"approval_id": "x" * 20}, csrf=False)
        self.assertEqual(status, 403)

    def test_another_session_cannot_approve(self):
        plan = {"decision": "plan", "title": "t", "summary": "", "project": "p8", "steps": [
            {"title": "ساخت", "detail": "", "tool": "create_project"}]}
        self.ollama.script = lambda kind, payload: {"content": json.dumps(plan)}
        intruder = Browser(self.port)
        # a second browser session has its own cookie even though the first one is alive
        results = []

        def handler(event):
            if "plan_approval" in event:
                results.append(intruder.post("/api/chat/plan-decision", {"plan_id": event["plan_approval"]["plan_id"], "approved": True})[0])
                self.browser.post("/api/chat/plan-decision", {"plan_id": event["plan_approval"]["plan_id"], "approved": False})
        self.browser.chat("بساز p8", on_event=handler)
        self.assertEqual(results, [404])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_general_mode_never_offers_tools(self):
        self.ollama.script = lambda kind, payload: {"content": ["سلام"]}
        events, text = self.browser.chat("بساز یک بازی", mode="general", csrf=False)
        self.assertEqual(self.ollama.kinds(), ["plain"])  # the unchanged general-chat path: no planner, no tools
        self.assertNotIn("tools", self.ollama.requests[0][1])
        self.assertIn("سلام", text)


if __name__ == "__main__":
    unittest.main()
