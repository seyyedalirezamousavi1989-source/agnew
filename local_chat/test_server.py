import http.client
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from local_chat.memory import PersistentMemory
from local_chat.server import (
    CREATOR,
    ChatHandler,
    IdentityError,
    ThreadingHTTPServer,
    _identity_session,
)


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, size=-1):
        return self.body if size < 0 else self.body[:size]

    def __iter__(self):
        return iter(self.body.splitlines(keepends=True))


class ChatServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    def setUp(self):
        self.identity_directory = tempfile.TemporaryDirectory(prefix="local-chat-identity-")
        self.identity_db = Path(self.identity_directory.name) / "identity.sqlite3"
        self.identity_path_patcher = patch(
            "local_chat.server.IDENTITY_DB_PATH", self.identity_db
        )
        self.identity_path_patcher.start()
        self.legacy_identity_db = Path(self.identity_directory.name) / "legacy.sqlite3"
        self.legacy_identity_path_patcher = patch(
            "local_chat.server.LEGACY_IDENTITY_DB_PATH", self.legacy_identity_db
        )
        self.legacy_identity_path_patcher.start()
        self.settings_path = Path(self.identity_directory.name) / "smtp_settings.json"
        self.example_path = (
            Path(self.identity_directory.name) / "smtp_settings.example.json"
        )
        self.settings_path_patcher = patch(
            "local_chat.server.SMTP_SETTINGS_PATH", self.settings_path
        )
        self.settings_path_patcher.start()
        self.example_path_patcher = patch(
            "local_chat.server.SMTP_SETTINGS_EXAMPLE_PATH", self.example_path
        )
        self.example_path_patcher.start()
        self.user_smtp_patcher = patch(
            "local_chat.server.SMTP_SETTINGS_USER_PATH",
            Path(self.identity_directory.name) / "user_smtp_settings.json",
        )
        self.user_smtp_patcher.start()
        self.appdata_patcher = patch.dict(
            "os.environ", {"LOCALAPPDATA": self.identity_directory.name}
        )
        self.appdata_patcher.start()

    def tearDown(self):
        self.appdata_patcher.stop()
        self.user_smtp_patcher.stop()
        self.example_path_patcher.stop()
        self.settings_path_patcher.stop()
        self.legacy_identity_path_patcher.stop()
        self.identity_path_patcher.stop()
        self.identity_directory.cleanup()

    def register_creator(self):
        with _identity_session() as connection:
            connection.execute(
                """
                INSERT INTO creator_identity (id, name, email, role, verified_at)
                VALUES (1, ?, ?, ?, ?)
                """,
                (CREATOR["name"], CREATOR["email"], CREATOR["role"], int(time.time())),
            )

    def identity_query(self, query, parameters=()):
        connection = sqlite3.connect(self.identity_db)
        try:
            return connection.execute(query, parameters).fetchone()
        finally:
            connection.close()

    def write_identity_settings(self, data, example=False):
        settings_path = self.example_path if example else self.settings_path
        settings_path.write_text(json.dumps(data), encoding="utf-8")

    def test_models_dir_settings_require_creator_and_validate_path(self):
        status, body = self.request("GET", "/api/settings")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["models_dir"], "")
        status, _ = self.request("POST", "/api/settings/models-dir", {"path": "C:\\x"})
        self.assertEqual(status, 403)
        status, _ = self.request("POST", "/api/ollama/restart")
        self.assertEqual(status, 403)

        self.register_creator()
        status, _ = self.request(
            "POST", "/api/settings/models-dir", {"path": "relative/dir"}
        )
        self.assertEqual(status, 400)
        status, _ = self.request(
            "POST", "/api/settings/models-dir", {"path": "/definitely/not/here"}
        )
        self.assertEqual(status, 400)

        models = Path(self.identity_directory.name) / "models"
        (models / "manifests").mkdir(parents=True)
        (models / "blobs").mkdir()
        (models / "manifests" / "qwen").write_text("{}", encoding="utf-8")
        status, body = self.request(
            "POST", "/api/settings/models-dir", {"path": str(models)}
        )
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["models_dir"], str(models))
        self.assertTrue(data["saved"]["looks_like_ollama"])
        self.assertEqual(data["saved"]["model_count"], 1)
        status, body = self.request("GET", "/api/settings")
        self.assertEqual(json.loads(body)["models_dir"], str(models))

        status, body = self.request("POST", "/api/settings/models-dir", {"path": ""})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["models_dir"], "")

    def test_prompt_fitting_keeps_latest_message_and_system_prompt(self):
        from local_chat.server import PROMPT_CHAR_BUDGET, _fit_messages

        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 3000}
            for i in range(10)
        ]
        history.insert(8, {"role": "system", "content": "web " * 5000})
        history.append({"role": "user", "content": "سوال آخر"})
        fitted = _fit_messages("SYS", history)
        # what no longer fits is condensed into a digest appended to the system prompt (no silent amnesia)
        self.assertEqual(fitted[0]["role"], "system")
        self.assertTrue(fitted[0]["content"].startswith("SYS"))
        self.assertIn("خلاصهٔ خودکار", fitted[0]["content"])
        self.assertEqual(fitted[-1]["content"], "سوال آخر")
        self.assertLessEqual(sum(len(m["content"]) for m in fitted), PROMPT_CHAR_BUDGET)
        self.assertTrue(any(m["content"].startswith("web") for m in fitted))

    def test_open_url_requires_creator_and_http_scheme(self):
        status, _ = self.request("POST", "/api/open-url", {"url": "https://example.com"})
        self.assertEqual(status, 403)
        self.register_creator()
        with patch("webbrowser.open") as opened:
            for bad in ("javascript:alert(1)", "file:///C:/x", "ftp://a.b", "", 5):
                status, _ = self.request("POST", "/api/open-url", {"url": bad})
                self.assertEqual(status, 400)
            opened.assert_not_called()
            status, _ = self.request("POST", "/api/open-url", {"url": "https://example.com/a"})
            self.assertEqual(status, 200)
            opened.assert_called_once_with("https://example.com/a")

    def test_shutdown_requires_creator_and_stops_ollama_then_calls_hook(self):
        import local_chat.server as srv

        status, _ = self.request("POST", "/api/shutdown")
        self.assertEqual(status, 403)
        self.register_creator()
        called = threading.Event()
        with patch.object(srv, "_stop_ollama") as stop, patch.object(
            srv, "SHUTDOWN_HOOK", called.set
        ):
            status, _ = self.request("POST", "/api/shutdown")
            self.assertEqual(status, 200)
            self.assertTrue(called.wait(3))
            stop.assert_called_once()

    @patch("local_chat.server.urlopen")
    def test_large_models_get_lighter_profile(self, mock_urlopen):
        from local_chat.server import _PROFILE_CACHE, _model_profile

        _PROFILE_CACHE.clear()
        mock_urlopen.return_value = FakeResponse(
            b'{"models":[{"name":"big:30b","size":19000000000},{"name":"small:7b","size":4500000000}]}'
        )
        self.assertEqual(_model_profile("big:30b")["num_ctx"], 4096)
        self.assertEqual(_model_profile("big:30b")["num_predict"], 2048)
        self.assertEqual(_model_profile("small:7b")["num_ctx"], 8192)
        _PROFILE_CACHE.clear()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, payload=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if body else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        status = response.status
        connection.close()
        return status, data

    def test_serves_self_contained_interface_and_font(self):
        page_status, page_body = self.request("GET", "/")
        font_status, font_body = self.request("GET", "/static/fonts/Vazirmatn-var.woff2")

        self.assertEqual(page_status, 200)
        self.assertIn("گفت‌وگوی محلی".encode(), page_body)
        self.assertEqual(font_status, 200)
        self.assertTrue(font_body.startswith(b"wOF2"))



    def test_site_builder_ui_and_routes_are_removed(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("حافظهٔ بلندمدت من".encode(), body)
        self.assertNotIn("ساخت وب‌سایت".encode(), body)
        agent_status, _ = self.request("GET", "/agent")
        self.assertEqual(agent_status, 404)

    def test_explicit_memory_command_saves_without_calling_ollama(self):
        self.register_creator()
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentMemory(Path(directory) / "memory.sqlite3")
            with patch("local_chat.server.MEMORY_STORE", memory):
                status, body = self.request(
                    "POST",
                    "/api/chat",
                    {
                        "model": "qwen3:8b",
                        "messages": [
                            {"role": "user", "content": "یادت باشه من فارسی صحبت می‌کنم"}
                        ],
                    },
                )
                saved_items = memory.list()
        self.assertEqual(status, 200)
        self.assertIn("ذخیره کردم".encode(), body)
        self.assertEqual(saved_items[0]["content"], "من فارسی صحبت می‌کنم")

    def test_memory_api_lists_and_forgets_saved_items(self):
        self.register_creator()
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentMemory(Path(directory) / "memory.sqlite3")
            memory.add("ترجیح می‌دهم پاسخ‌ها کوتاه باشند")
            with patch("local_chat.server.MEMORY_STORE", memory):
                status, body = self.request("GET", "/api/memory")
                self.assertEqual(status, 200)
                self.assertEqual(len(json.loads(body)["memories"]), 1)
                status, body = self.request(
                    "POST", "/api/memory/forget", {"query": "پاسخ‌ها کوتاه"}
                )
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["deleted"], 1)

    def test_server_reports_version_for_launcher_health_check(self):
        status, body = self.request("GET", "/api/status")

        self.assertEqual(status, 200)
        from local_chat.version import APP_VERSION
        self.assertEqual(json.loads(body)["version"], APP_VERSION)

    def test_existing_creator_database_is_migrated_to_stable_app_data_path(self):
        creator = (
            CREATOR["name"],
            CREATOR["email"],
            CREATOR["role"],
            int(time.time()),
        )
        with sqlite3.connect(self.legacy_identity_db) as connection:
            connection.execute(
                """
                CREATE TABLE creator_identity (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    role TEXT NOT NULL,
                    verified_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO creator_identity VALUES (1, ?, ?, ?, ?)", creator
            )

        from local_chat.server import _registered_creator

        migrated = _registered_creator()

        self.assertEqual(migrated["name"], CREATOR["name"])
        self.assertTrue(self.identity_db.is_file())

    def test_capability_registry_lists_registered_capabilities_without_details(self):
        status, body = self.request("GET", "/api/capabilities")

        self.assertEqual(status, 200)
        capabilities = json.loads(body)["capabilities"]
        ids = {capability["id"] for capability in capabilities}
        self.assertEqual(
            ids,
            {"local_chat", "web_search", "creator_identity", "file_upload", "vision", "audio", "model_manager", "approved_system_tools", "forex_chat", "forex_chart_capture"},
        )
        self.assertTrue(all(capability["status"] == "available" for capability in capabilities))
        self.assertTrue(all("purpose" not in capability for capability in capabilities))



    def test_capability_discovery_selects_only_relevant_summaries(self):
        status, body = self.request(
            "GET", "/api/capabilities/discover?query=latest%20news"
        )

        self.assertEqual(status, 200)
        capabilities = json.loads(body)["capabilities"]
        ids = {capability["id"] for capability in capabilities}
        self.assertEqual(ids, {"local_chat", "web_search"})
        self.assertTrue(all("dependencies" not in capability for capability in capabilities))

        status, body = self.request(
            "GET", "/api/capabilities/discover?query=who%20am%20i"
        )
        self.assertEqual(status, 200)
        ids = {capability["id"] for capability in json.loads(body)["capabilities"]}
        self.assertIn("creator_identity", ids)
        self.assertNotIn("web_search", ids)

    def test_capability_documentation_is_loaded_only_by_detail_endpoint(self):
        status, body = self.request("GET", "/api/capabilities/web_search")

        self.assertEqual(status, 200)
        detail = json.loads(body)
        self.assertIn("dependencies", detail["capability"])
        self.assertIn("DuckDuckGo", detail["documentation"])

        status, body = self.request("GET", "/api/capabilities/not-a-capability")
        self.assertEqual(status, 404)

    def test_unavailable_capability_is_not_returned_as_discoverable(self):
        from local_chat.capabilities import (
            _REGISTRY,
            discover_capabilities,
            list_capabilities,
            register_capability,
        )

        @register_capability(
            id="disabled_test",
            name="Disabled test",
            description="A test-only unavailable feature.",
            purpose="Validate availability filtering.",
            activation_conditions=("disabled phrase",),
            inputs=(),
            outputs=(),
            dependencies=(),
            limitations=(),
            documentation="architecture.md",
            availability=lambda: False,
        )
        def disabled_test_implementation():
            return None

        try:
            summaries = {item["id"]: item for item in list_capabilities()}
            self.assertEqual(summaries["disabled_test"]["status"], "unavailable")
            self.assertNotIn(
                "disabled_test",
                {item["id"] for item in discover_capabilities("disabled phrase")},
            )
        finally:
            _REGISTRY.pop("disabled_test")


    def test_system_prompt_contains_live_capability_map_and_behavior_rules(self):
        from local_chat.server import _system_prompt
        # the detailed map is sent when the user asks about abilities (ability_cue) ...
        prompt = _system_prompt(forex=True, ability_cue=True)
        self.assertIn("نقشهٔ واقعی قابلیت‌های همین ایجنت", prompt)
        self.assertIn("گفت‌وگوی تحلیل فارکس", prompt)
        self.assertIn("قواعد آگاهی از قابلیت‌ها", prompt)
        # ... otherwise a one-line list of names keeps the model aware of what exists, at a fraction of the size
        lean = _system_prompt(forex=True)
        self.assertIn("گفت‌وگوی تحلیل فارکس", lean)
        self.assertNotIn("نقشهٔ واقعی قابلیت‌های همین ایجنت", lean)
        self.assertLess(len(lean), len(prompt))

    def test_capability_question_is_answered_without_model_context(self):
        self.register_creator()
        for question in (
            "چه قابلیت‌هایی داری؟",
            "تو چه کارایی بلدی؟",
            "امکاناتت چیه؟",
            "قابلیت‌هاتو بگو",
            "قابلیت سرچ هم داری؟",
            "سرچ هم داری؟",
            "میتونی سرچ کنی؟",
            "تو چی بلدی؟",
            "What can you do?",
        ):
            with self.subTest(question=question):
                with patch("local_chat.server.urlopen") as mock_model:
                    status, body = self.request(
                        "POST",
                        "/api/chat",
                        {
                            "model": "qwen3:14b",
                            "messages": [{"role": "user", "content": question}],
                        },
                    )

                self.assertEqual(status, 200)
                answer = json.loads(body)["message"]["content"]
                self.assertIn("جست‌وجوی وب", answer)
                if question in {"چه قابلیت‌هایی داری؟", "تو چه کارایی بلدی؟", "امکاناتت چیه؟", "قابلیت‌هاتو بگو", "تو چی بلدی؟", "What can you do?"}:
                    self.assertIn("گفت‌وگو", answer)
                mock_model.assert_not_called()

    def test_incidental_capability_words_do_not_trigger_capability_inventory(self):
        from local_chat.server import _capability_question_answer

        ordinary_messages = (
            "این قابلیت توی برنامه چطور کار می‌کنه؟",
            "قابلیت ذخیره‌سازی رو به این پروژه اضافه کن.",
            "درباره قابلیت‌های Django توضیح بده.",
            "این امکان رو بررسی کن و مزایا و معایبش رو بگو.",
            "مدل چه قابلیت‌هایی برای تحلیل داده دارد؟",
            "در نسخه بعدی امکان جست‌وجو را بهتر کن.",
        )
        for message in ordinary_messages:
            with self.subTest(message=message):
                self.assertIsNone(_capability_question_answer(message))

    def test_specific_capability_question_returns_only_relevant_feature(self):
        from local_chat.server import _capability_question_answer

        answer = _capability_question_answer("آیا عکس و تصویر رو می‌تونی تحلیل کنی؟")
        self.assertIsNotNone(answer)
        self.assertIn("درک تصویر", answer)
        self.assertNotIn("جست‌وجوی وب", answer)

        forex_answer = _capability_question_answer("تحلیل فارکس و جفت ارزها رو بلدی؟")
        self.assertIsNotNone(forex_answer)
        self.assertIn("تحلیل فارکس", forex_answer)

    @patch(
        "local_chat.server._smtp_settings",
        return_value={
            "username": CREATOR["email"],
            "password": "test-app-password",
            "host": "smtp.gmail.com",
            "port": 587,
            "from_email": CREATOR["email"],
        },
    )
    @patch("local_chat.server._send_creator_verification_email")
    def test_first_run_sends_and_stores_only_a_hashed_one_time_code(
        self, send_email, _smtp_settings
    ):
        status, body = self.request("GET", "/api/identity")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["verified"])

        status, body = self.request("POST", "/api/identity/start")
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(payload["sent"])
        self.assertNotIn("code", payload)
        send_email.assert_called_once()
        code = send_email.call_args.args[0]
        self.assertRegex(code, r"^\d{6}$")

        row = self.identity_query(
            "SELECT code_hash, salt FROM creator_verification WHERE id = 1"
        )
        self.assertIsNotNone(row)
        self.assertNotEqual(row[0], code)
        self.assertNotIn(code, row[0])

        status, body = self.request("POST", "/api/identity/start")
        self.assertEqual(status, 429)
        send_email.assert_called_once()

    @patch(
        "local_chat.server._smtp_settings",
        return_value={
            "username": CREATOR["email"],
            "password": "test-app-password",
            "host": "smtp.gmail.com",
            "port": 587,
            "from_email": CREATOR["email"],
        },
    )
    @patch("local_chat.server._send_creator_verification_email")
    def test_verified_creator_persists_and_is_recognized_without_calling_model(
        self, send_email, _smtp_settings
    ):
        status, body = self.request("POST", "/api/identity/start")
        self.assertEqual(status, 200)
        code = send_email.call_args.args[0]

        status, body = self.request(
            "POST", "/api/identity/verify", {"code": "000000" if code != "000000" else "000001"}
        )
        self.assertEqual(status, 400)

        status, body = self.request(
            "POST", "/api/identity/verify", {"code": code}
        )
        self.assertEqual(status, 200)
        verified = json.loads(body)
        self.assertEqual(verified["profile"]["name"], CREATOR["name"])
        self.assertEqual(verified["profile"]["email"], CREATOR["email"])
        self.assertEqual(verified["profile"]["role"], CREATOR["role"])

        status, body = self.request("GET", "/api/identity")
        persisted = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(persisted["verified"])
        self.assertEqual(persisted["profile"]["name"], CREATOR["name"])

        restarted_server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
        restarted_thread = threading.Thread(
            target=restarted_server.serve_forever, daemon=True
        )
        restarted_thread.start()
        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1", restarted_server.server_address[1], timeout=5
            )
            connection.request("GET", "/api/identity")
            response = connection.getresponse()
            restarted_identity = json.loads(response.read())
            connection.close()
            self.assertTrue(restarted_identity["verified"])
            self.assertEqual(
                restarted_identity["profile"]["name"], CREATOR["name"]
            )

            for question in (
                "من کی هستم؟",
                "سازنده‌ات کیست؟",
                "خودتو معرفی کن و بگو سازندت کیه",
                "معرفی کن سازندش منم",
                "اسم من چیه؟",
                "من کیم؟",
                "چی درباره‌م میدونی؟",
                "سید علیرضا موسوی کیه؟",
                "Who created you?",
            ):
                chat_payload = json.dumps(
                    {
                        "model": "qwen3:14b",
                        "messages": [{"role": "user", "content": question}],
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                connection = http.client.HTTPConnection(
                    "127.0.0.1", restarted_server.server_address[1], timeout=5
                )
                with patch("local_chat.server.urlopen") as mock_model:
                    connection.request(
                        "POST",
                        "/api/chat",
                        chat_payload,
                        {"Content-Type": "application/json"},
                    )
                    response = connection.getresponse()
                    body = response.read()
                connection.close()
                self.assertEqual(response.status, 200)
                answer = json.loads(body)["message"]["content"]
                self.assertIn(CREATOR["name"], answer)
                self.assertIn(CREATOR["email"], answer)
                mock_model.assert_not_called()
        finally:
            restarted_server.shutdown()
            restarted_server.server_close()
            restarted_thread.join()

    def test_chat_requires_verified_creator(self):
        status, body = self.request(
            "POST",
            "/api/chat",
            {
                "model": "qwen3:14b",
                "messages": [{"role": "user", "content": "سلام"}],
            },
        )

        self.assertEqual(status, 403)
        self.assertIn("تأیید", json.loads(body)["error"])

    def test_missing_smtp_password_is_reported_without_creating_a_code(self):
        with patch(
            "local_chat.server._smtp_settings",
            side_effect=IdentityError("برای ارسال کد، رمز برنامه را تنظیم کنید.", 503),
        ):
            status, body = self.request("POST", "/api/identity/start")

        self.assertEqual(status, 503)
        self.assertIn("رمز برنامه", json.loads(body)["error"])
        challenge_count = self.identity_query(
            "SELECT COUNT(*) FROM creator_verification"
        )[0]
        self.assertEqual(challenge_count, 0)

    def test_local_code_from_settings_is_accepted_without_email(self):
        self.write_identity_settings({"verification_code": "123456"})

        with patch("local_chat.server._send_creator_verification_email") as send_email:
            status, body = self.request("POST", "/api/identity/start")
            self.assertEqual(status, 200)
            self.assertTrue(json.loads(body)["local_code"])
            self.assertNotIn("123456", body.decode("utf-8"))

            status, body = self.request(
                "POST", "/api/identity/verify", {"code": "123456"}
            )

        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["verified"])
        send_email.assert_not_called()

    def test_six_digit_password_from_edited_example_file_is_accepted(self):
        self.write_identity_settings({"password": "111111"}, example=True)

        with patch("local_chat.server._send_creator_verification_email") as send_email:
            status, body = self.request("POST", "/api/identity/start")

        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["local_code"])
        send_email.assert_not_called()

    def test_smtp_email_is_sent_with_starttls_to_fixed_owner_address(self):
        from local_chat.server import _send_creator_verification_email

        with patch("local_chat.server.smtplib.SMTP") as smtp_factory:
            _send_creator_verification_email(
                "123456",
                {
                    "username": CREATOR["email"],
                    "password": "test-app-password",
                    "host": "smtp.gmail.com",
                    "port": 587,
                    "from_email": CREATOR["email"],
                },
            )

        client = smtp_factory.return_value.__enter__.return_value
        client.starttls.assert_called_once()
        client.login.assert_called_once_with(CREATOR["email"], "test-app-password")
        message = client.send_message.call_args.args[0]
        self.assertEqual(message["To"], CREATOR["email"])
        self.assertIn("123456", message.get_content())

    @patch(
        "local_chat.server._smtp_settings",
        return_value={
            "username": CREATOR["email"],
            "password": "test-app-password",
            "host": "smtp.gmail.com",
            "port": 587,
            "from_email": CREATOR["email"],
        },
    )
    @patch("local_chat.server._send_creator_verification_email")
    def test_failed_codes_are_counted_and_lock_challenge(
        self, send_email, _smtp_settings
    ):
        self.request("POST", "/api/identity/start")
        code = send_email.call_args.args[0]
        wrong_code = "000000" if code != "000000" else "000001"
        for attempt in range(5):
            status, _body = self.request(
                "POST", "/api/identity/verify", {"code": wrong_code}
            )
        self.assertEqual(status, 429)
        remaining = self.identity_query(
            "SELECT COUNT(*) FROM creator_verification"
        )[0]
        attempts = self.identity_query(
            "SELECT attempts FROM creator_verification WHERE id = 1"
        )[0]
        self.assertEqual(remaining, 1)
        self.assertEqual(attempts, 5)
        send_email.assert_called_once()

    @patch(
        "local_chat.server._smtp_settings",
        return_value={
            "username": CREATOR["email"],
            "password": "test-app-password",
            "host": "smtp.gmail.com",
            "port": 587,
            "from_email": CREATOR["email"],
        },
    )
    @patch("local_chat.server._send_creator_verification_email")
    def test_expired_code_is_rejected_and_removed(self, send_email, _smtp_settings):
        with patch("local_chat.server.time.time", return_value=1_000):
            status, _body = self.request("POST", "/api/identity/start")
        code = send_email.call_args.args[0]

        with patch(
            "local_chat.server.time.time",
            return_value=1_000 + 10 * 60 + 1,
        ):
            status, body = self.request(
                "POST", "/api/identity/verify", {"code": code}
            )

        self.assertEqual(status, 410)
        self.assertIn("مهلت", json.loads(body)["error"])
        remaining = self.identity_query(
            "SELECT COUNT(*) FROM creator_verification"
        )[0]
        self.assertEqual(remaining, 0)

    @patch("local_chat.server._ddg_results", return_value=[])
    def test_search_endpoint_validates_and_returns_search_results(self, mock_search):
        import local_chat.settings as app_settings

        app_settings.set_search_settings({"engine": "duckduckgo"})
        status, body = self.request("POST", "/api/search", {"query": "آب و هوا"})

        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["results"], [])
        self.assertEqual(data["engine"], "none")
        mock_search.assert_called_once_with("آب و هوا")

    def test_search_endpoint_rejects_empty_query(self):
        status, body = self.request("POST", "/api/search", {"query": " "})

        self.assertEqual(status, 400)
        self.assertIn("جست‌وجو", json.loads(body)["error"])

    @patch("local_chat.server._ddg_results")
    def test_search_endpoint_reports_search_provider_errors(self, mock_search):
        import local_chat.settings as app_settings

        app_settings.set_search_settings({"engine": "duckduckgo"})
        mock_search.side_effect = RuntimeError("موتور جست‌وجو موقتاً در دسترس نیست")
        status, body = self.request("POST", "/api/search", {"query": "موضوع"})

        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["results"], [])
        self.assertTrue(data["errors"])

    @patch("local_chat.server._fetch_public_page", return_value="متن صفحه")
    @patch("local_chat.server.urlopen")
    def test_duckduckgo_search_parses_results_and_loads_pages(
        self, mock_urlopen, mock_fetch_page
    ):
        mock_urlopen.return_value = FakeResponse(
            (
                '<div class="result"><a class="result__a" '
                'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle">'
                '&lt;عنوان&gt;</a><div class="result__snippet">خلاصهٔ صفحه</div></div>'
            ).encode("utf-8")
        )
        mock_urlopen.return_value.headers = {"Content-Type": "text/html; charset=UTF-8"}
        from local_chat.server import _search_duckduckgo

        results = _search_duckduckgo("موضوع")

        self.assertEqual(results[0]["title"], "<عنوان>")
        self.assertEqual(results[0]["url"], "https://example.com/article")
        self.assertEqual(results[0]["snippet"], "خلاصهٔ صفحه")
        self.assertEqual(results[0]["content"], "متن صفحه")
        request = mock_urlopen.call_args.args[0]
        self.assertIn("html.duckduckgo.com/html/", request.full_url)
        self.assertIsNone(request.get_header("X-goog-api-key"))
        mock_fetch_page.assert_called_once_with("https://example.com/article")

    def test_public_page_fetcher_rejects_private_addresses(self):
        from local_chat.server import _fetch_public_page

        self.assertEqual(_fetch_public_page("http://127.0.0.1/"), "")

    def test_page_text_parser_ignores_scripts_and_navigation(self):
        from local_chat.server import PageTextParser

        parser = PageTextParser()
        parser.feed(
            "<html><nav>menu</nav><main><h1>Title</h1><p>Useful text</p>"
            "<script>ignored()</script></main></html>"
        )

        text = " ".join(" ".join(parser.parts).split())
        self.assertIn("Title", text)
        self.assertIn("Useful text", text)
        self.assertNotIn("menu", text)
        self.assertNotIn("ignored", text)

    @patch("local_chat.server.urlopen")
    def test_lists_models_from_ollama(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse(b'{"models":[{"name":"qwen3:14b"}]}')
        status, body = self.request("GET", "/api/models")

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["models"][0]["name"], "qwen3:14b")
        mock_urlopen.assert_called_once_with("http://127.0.0.1:11434/api/tags", timeout=5)

    @patch("local_chat.server.urlopen")
    def test_streams_chat_and_uses_hardware_friendly_context(self, mock_urlopen):
        self.register_creator()
        mock_urlopen.return_value = FakeResponse(
            b'{"message":{"content":"\xd8\xb3\xd9\x84\xd8\xa7\xd9\x85"},"done":true}\n'
        )
        status, body = self.request(
            "POST",
            "/api/chat",
            {"model": "qwen3:14b", "messages": [{"role": "user", "content": "سلام"}]},
        )

        self.assertEqual(status, 200)
        self.assertIn("سلام".encode(), body)
        request = mock_urlopen.call_args.args[0]
        sent = json.loads(request.data)
        self.assertEqual(sent["options"]["num_ctx"], 8192)
        self.assertTrue(sent["stream"])
        self.assertNotIn(CREATOR["name"], json.dumps(sent["messages"], ensure_ascii=False))
        self.assertNotIn("DuckDuckGo", json.dumps(sent["messages"], ensure_ascii=False))

    @patch("local_chat.server.urlopen")
    def test_rejects_invalid_message_before_calling_ollama(self, mock_urlopen):
        status, body = self.request(
            "POST",
            "/api/chat",
            {"model": "qwen3:14b", "messages": [{"role": "tool", "content": "ignore validation"}]},
        )

        self.assertEqual(status, 400)
        self.assertIn("معتبر نیست", json.loads(body)["error"])
        mock_urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
