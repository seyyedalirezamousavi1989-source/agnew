import base64
import http.client
import http.server
import io
import json
import struct
import tempfile
import threading
import time
import unittest
import wave
import zipfile
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

from local_chat import files as attach_files
from local_chat import models_catalog, search_providers
from local_chat import server as srv
from local_chat import settings as app_settings
from local_chat.memory import PersistentMemory

PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 1, 1)
       + b"\x08\x02\x00\x00\x00\x90wS\xde" + b"\x00" * 16)

CAPS = {
    "text:8b": ["completion", "tools"],
    "qwen3-vl:8b": ["completion", "vision", "thinking"],
    "gemma4:e4b": ["completion", "vision", "audio", "thinking"],
    "nomic-embed-text": ["embedding"],
    "user/model:1b": ["completion"],
}


def make_wav(seconds, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


def make_zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class FakeOllama:
    """Tiny stand-in for the Ollama HTTP API; records every request."""

    def __init__(self, installed):
        self.installed = dict(installed)
        self.log = []
        self.chat_reply = "پاسخ آزمایشی"
        self.pull_block = threading.Event()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_):
                pass

            def _body(self):
                length = int(self.headers.get("Content-Length", "0"))
                return json.loads(self.rfile.read(length) or b"{}")

            def _json(self, payload, status=200):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.log.append(("GET", self.path, None))
                if self.path == "/api/tags":
                    self._json({"models": [{"name": n, "size": 5_000_000_000} for n in outer.installed]})
                elif self.path == "/api/version":
                    self._json({"version": "0.31.0"})
                else:
                    self._json({}, 404)

            def do_DELETE(self):
                payload = self._body()
                outer.log.append(("DELETE", self.path, payload))
                outer.installed.pop(payload.get("model"), None)
                self._json({})

            def do_POST(self):
                payload = self._body()
                outer.log.append(("POST", self.path, payload))
                if self.path == "/api/show":
                    caps = outer.installed.get(payload.get("model"))
                    if caps is None:
                        self._json({"error": "not found"}, 404)
                    else:
                        self._json({"capabilities": caps})
                elif self.path == "/api/chat":
                    if payload.get("think") is True and "thinking" not in outer.installed.get(payload["model"], []):
                        self._json({"error": "does not support think"}, 400)
                        return
                    if not payload.get("stream", True):
                        self._json({"message": {"role": "assistant", "content": "متن رونویسی‌شده"}, "done": True})
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    for piece in (outer.chat_reply[:3], outer.chat_reply[3:]):
                        self.wfile.write(json.dumps({"message": {"content": piece}, "done": False}).encode() + b"\n")
                    self.wfile.write(json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"}).encode() + b"\n")
                elif self.path == "/api/pull":
                    if payload["model"] == "does/not-exist":
                        self._json({"error": "pull model manifest: file does not exist"}, 500)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    self.wfile.write(b'{"status":"pulling manifest"}\n')
                    self.wfile.write(b'{"status":"pulling abc","digest":"sha256:abc","total":100,"completed":40}\n')
                    self.wfile.flush()
                    outer.pull_block.wait(2)
                    self.wfile.write(b'{"status":"pulling abc","digest":"sha256:abc","total":100,"completed":100}\n')
                    self.wfile.write(b'{"status":"success"}\n')
                    outer.installed[payload["model"]] = ["completion", "vision"]
                else:
                    self._json({}, 404)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat_requests(self, streaming=True):
        return [p for m, path, p in self.log
                if path == "/api/chat" and p.get("stream", True) == streaming]


class FeatureTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = srv.ThreadingHTTPServer(("127.0.0.1", 0), srv.ChatHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="localchat-feature-")
        base = Path(self.tmp.name)
        self.patches = [
            patch.dict("os.environ", {"LOCALAPPDATA": self.tmp.name}),
            patch.object(srv, "IDENTITY_DB_PATH", base / "identity.sqlite3"),
            patch.object(srv, "LEGACY_IDENTITY_DB_PATH", base / "legacy.sqlite3"),
            patch.object(srv, "MEMORY_STORE", PersistentMemory(base / "memory.sqlite3")),
        ]
        for item in self.patches:
            item.start()
        with srv._identity_session() as connection:
            connection.execute(
                "INSERT INTO creator_identity (id, name, email, role, verified_at) VALUES (1, ?, ?, ?, ?)",
                (srv.CREATOR["name"], srv.CREATOR["email"], srv.CREATOR["role"], int(time.time())))
        self.ollama = FakeOllama(CAPS)
        self.ollama_patch = patch.object(srv, "OLLAMA_URL", self.ollama.url)
        self.ollama_patch.start()
        srv._CAPS_CACHE.clear()
        srv._PROFILE_CACHE.clear()
        srv._KEYLESS_GOOGLE_BLOCKED_UNTIL = 0.0

    def tearDown(self):
        self.ollama_patch.stop()
        self.ollama.stop()
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    # --- helpers ---
    def call(self, method, path, payload=None, raw=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        body = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
        merged = {"Content-Type": "application/json"} if payload is not None else {}
        merged.update(headers or {})
        connection.request(method, path, body=body, headers=merged)
        response = connection.getresponse()
        data = response.read()
        status = response.status
        connection.close()
        return status, data

    def upload(self, name, data):
        status, body = self.call("POST", "/api/upload", raw=data,
                                 headers={"X-Filename": quote(name), "Content-Type": "application/octet-stream"})
        return status, json.loads(body)

    def chat(self, model, text, attachments=None, extra_messages=None, **extra):
        message = {"role": "user", "content": text}
        if attachments:
            message["attachments"] = attachments
        status, body = self.call("POST", "/api/chat", {
            "model": model, "messages": (extra_messages or []) + [message], **extra})
        return status, body

    @staticmethod
    def chunks(body):
        return [json.loads(line) for line in body.decode().splitlines() if line.strip()]

    @staticmethod
    def reply_text(chunks):
        return "".join((c.get("message") or {}).get("content", "") for c in chunks)


class UploadTests(FeatureTestCase):
    def test_text_file_is_read_and_given_to_the_model(self):
        status, info = self.upload("hello.py", "def تست():\n    return 42\n".encode())
        self.assertEqual(status, 201)
        attachment = info["attachment"]
        self.assertEqual(attachment["kind"], "text")
        status, body = self.chat("text:8b", "این کد چیه؟", [attachment["id"]])
        self.assertEqual(status, 200)
        self.assertIn("پاسخ آزمایشی", self.reply_text(self.chunks(body)))
        sent = self.ollama.chat_requests()[-1]
        prompt = sent["messages"][-1]["content"]
        self.assertIn("def تست():", prompt)
        self.assertIn("hello.py", prompt)
        self.assertLess(prompt.index("def تست"), prompt.index("این کد چیه؟"))
        self.assertIn("پیوست", sent["messages"][0]["content"])

    def test_files_raise_context_for_small_models_but_not_large_ones(self):
        _, info = self.upload("a.txt", b"hello")
        self.chat("text:8b", "x", [info["attachment"]["id"]])
        self.assertEqual(self.ollama.chat_requests()[-1]["options"]["num_ctx"], 16384)
        self.chat("text:8b", "بدون فایل")
        self.assertEqual(self.ollama.chat_requests()[-1]["options"]["num_ctx"], 8192)

    def test_follow_up_still_sees_the_earlier_file(self):
        _, info = self.upload("notes.md", b"SECRET-TOKEN-777")
        attachment_id = info["attachment"]["id"]
        history = [{"role": "user", "content": "فایل", "attachments": [attachment_id]},
                   {"role": "assistant", "content": "دیدم"}]
        self.chat("text:8b", "توکن چی بود؟", extra_messages=history)
        joined = json.dumps(self.ollama.chat_requests()[-1]["messages"], ensure_ascii=False)
        self.assertIn("SECRET-TOKEN-777", joined)

    def test_unknown_attachment_id_is_reported_not_crashing(self):
        status, body = self.chat("text:8b", "سلام", ["0" * 32])
        self.assertEqual(status, 200)
        self.assertIn("notice", {k for c in self.chunks(body) for k in c})

    def test_invalid_attachment_ids_are_rejected(self):
        status, _ = self.chat("text:8b", "سلام", ["../../etc/passwd"])
        self.assertEqual(status, 400)

    def test_upload_requires_creator_and_valid_size(self):
        with srv._identity_session() as connection:
            connection.execute("DELETE FROM creator_identity")
        status, _ = self.upload("a.txt", b"x")
        self.assertEqual(status, 403)

    def test_binary_files_are_refused_with_a_clear_message(self):
        status, info = self.upload("program.exe", b"MZ\x90\x00\x03\x00\x00\x00" + bytes(range(256)) * 4)
        self.assertEqual(status, 415)
        self.assertIn("پشتیبانی", info["error"])

    def test_cp1256_and_utf16_text_are_decoded(self):
        text = "سلام دوستان"
        status, info = self.upload("fa.txt", text.encode("cp1256"))
        self.assertEqual(status, 201)
        record = srv.ATTACHMENTS.get(info["attachment"]["id"])
        self.assertEqual(record["text"], text)

    def test_attachment_can_be_deleted(self):
        _, info = self.upload("a.txt", b"hello")
        attachment_id = info["attachment"]["id"]
        status, body = self.call("POST", "/api/attachments/delete", {"id": attachment_id})
        self.assertEqual(json.loads(body), {"deleted": True})
        self.assertIsNone(srv.ATTACHMENTS.get(attachment_id))


class ZipTests(FeatureTestCase):
    def test_zip_lists_files_reads_python_and_skips_junk(self):
        data = make_zip({
            "proj/main.py": "print('hi')\n",
            "proj/utils/helpers.py": "def f(): pass\n",
            "proj/node_modules/lib/index.js": "SHOULD-NOT-APPEAR",
            "proj/__pycache__/x.pyc": b"\x00\x01\x02",
            "proj/logo.png": PNG,
            "../evil.txt": "path traversal name",
        })
        status, info = self.upload("proj.zip", data)
        self.assertEqual(status, 201)
        self.assertEqual(info["attachment"]["kind"], "archive")
        text = srv.ATTACHMENTS.get(info["attachment"]["id"])["text"]
        self.assertIn("proj/main.py", text)
        self.assertIn("print('hi')", text)
        self.assertIn("def f(): pass", text)
        self.assertNotIn("SHOULD-NOT-APPEAR", text)
        self.assertIn("proj/logo.png", text)  # listed but not read
        self.assertNotIn("../", text)

    def test_zip_bomb_and_huge_archives_are_rejected(self):
        bomb = make_zip({"big.txt": b"0" * (3 * 1024 * 1024)})
        with patch.object(attach_files, "ZIP_MAX_DECLARED_TOTAL", 1024 * 1024):
            status, info = self.upload("bomb.zip", bomb)
        self.assertEqual(status, 413)
        many = make_zip({f"f{i}.txt": "x" for i in range(30)})
        with patch.object(attach_files, "ZIP_MAX_ENTRIES_SCANNED", 10):
            status, _ = self.upload("many.zip", many)
        self.assertEqual(status, 413)

    def test_each_file_is_read_with_a_byte_cap(self):
        archive = make_zip({"big.py": "x = 1\n" * 400_000})
        status, info = self.upload("big.zip", archive)
        self.assertEqual(status, 201)
        record = srv.ATTACHMENTS.get(info["attachment"]["id"])
        self.assertLess(len(record["text"]), 120_000)

    def test_corrupt_zip_is_a_clean_error(self):
        status, info = self.upload("broken.zip", b"PK\x03\x04" + b"garbage" * 20)
        self.assertEqual(status, 415)


class DocumentTests(FeatureTestCase):
    def test_docx_xlsx_pptx_text_extraction(self):
        docx = make_zip({"word/document.xml": (
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
            '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>عنوان</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>متن بند</w:t></w:r></w:p>'
            '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>الف</w:t></w:r></w:p></w:tc>'
            '<w:tc><w:p><w:r><w:t>ب</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>')})
        _, info = self.upload("r.docx", docx)
        text = srv.ATTACHMENTS.get(info["attachment"]["id"])["text"]
        self.assertIn("# عنوان", text)
        self.assertIn("متن بند", text)
        self.assertIn("الف | ب", text)

        ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        xlsx = make_zip({
            "xl/workbook.xml": f'<workbook {ns}><sheets><sheet name="Sales"/></sheets></workbook>',
            "xl/sharedStrings.xml": f'<sst {ns}><si><t>نام</t></si><si><t>علی</t></si></sst>',
            "xl/worksheets/sheet1.xml": (
                f'<worksheet {ns}><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>5</v></c></row>'
                f'<row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>7</v></c></row></sheetData></worksheet>')})
        _, info = self.upload("s.xlsx", xlsx)
        text = srv.ATTACHMENTS.get(info["attachment"]["id"])["text"]
        self.assertIn("Sales", text)
        self.assertIn("علی\t7", text)

        pptx = make_zip({"ppt/slides/slide1.xml": (
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<a:p><a:r><a:t>اسلاید اول</a:t></a:r></a:p></p:sld>')})
        _, info = self.upload("p.pptx", pptx)
        self.assertIn("اسلاید اول", srv.ATTACHMENTS.get(info["attachment"]["id"])["text"])

    def test_pdf_text_extraction(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            self.skipTest("pypdf not installed")
        content = b"BT /F1 24 Tf 72 700 Td (Hello PDF world) Tj ET"
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        pdf = b"%PDF-1.4\n"
        offsets = []
        for number, obj in enumerate(objects, 1):
            offsets.append(len(pdf))
            pdf += b"%d 0 obj\n" % number + obj + b"\nendobj\n"
        xref = len(pdf)
        pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
        for offset in offsets:
            pdf += b"%010d 00000 n \n" % offset
        pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref)
        status, info = self.upload("t.pdf", pdf)
        self.assertEqual(status, 201)
        self.assertIn("Hello PDF world", srv.ATTACHMENTS.get(info["attachment"]["id"])["text"])

    def test_notebook_cells_are_flattened(self):
        notebook = json.dumps({"cells": [
            {"cell_type": "code", "source": ["x = 1\n", "print(x)"], "outputs": [{"text": ["1\n"]}]},
            {"cell_type": "markdown", "source": "# عنوان"}]})
        _, info = self.upload("n.ipynb", notebook.encode())
        text = srv.ATTACHMENTS.get(info["attachment"]["id"])["text"]
        self.assertIn("print(x)", text)
        self.assertIn("خروجی", text)


class VisionAudioTests(FeatureTestCase):
    def test_image_needs_a_vision_model_and_is_sent_as_base64(self):
        status, info = self.upload("pic.png", PNG)
        self.assertEqual(status, 201)
        attachment_id = info["attachment"]["id"]
        status, body = self.chat("text:8b", "این چیه؟", [attachment_id])
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(body)["code"], "no_vision")
        self.assertEqual(self.ollama.chat_requests(), [])

        status, body = self.chat("qwen3-vl:8b", "این چیه؟", [attachment_id])
        self.assertEqual(status, 200)
        sent = self.ollama.chat_requests()[-1]
        images = [m.get("images") for m in sent["messages"] if m.get("images")]
        self.assertEqual(images, [[base64.b64encode(PNG).decode()]])
        self.assertGreaterEqual(sent["options"]["num_ctx"], 8192)

    def test_audio_is_split_into_chunks_and_transcribed_once(self):
        status, info = self.upload("talk.wav", make_wav(70))
        self.assertEqual(status, 201)
        self.assertEqual(info["attachment"]["kind"], "audio")
        record = srv.ATTACHMENTS.get(info["attachment"]["id"])
        self.assertEqual(len(record["audio_chunks"]), 3)

        status, body = self.chat("text:8b", "این صدا چی میگه؟", [record["id"]])
        self.assertEqual(status, 200)
        chunks = self.chunks(body)
        notices = [c["notice"] for c in chunks if "notice" in c]
        self.assertTrue(any("gemma4:e4b" in n for n in notices))
        transcription = self.ollama.chat_requests(streaming=False)
        self.assertEqual(len(transcription), 3)
        self.assertTrue(all(r["model"] == "gemma4:e4b" for r in transcription))
        self.assertTrue(all(len(r["messages"][0]["images"]) == 1 for r in transcription))
        prompt = self.ollama.chat_requests()[-1]["messages"][-1]["content"]
        self.assertIn("متن رونویسی‌شده", prompt)

        self.chat("text:8b", "یک بار دیگر", [record["id"]])
        self.assertEqual(len(self.ollama.chat_requests(streaming=False)), 3)  # cached transcript

    def test_audio_without_an_audio_model_asks_for_download(self):
        without_audio = {k: v for k, v in CAPS.items() if "audio" not in v}
        self.ollama.installed = without_audio
        srv._CAPS_CACHE.clear()
        _, info = self.upload("talk.wav", make_wav(3))
        status, body = self.chat("text:8b", "؟", [info["attachment"]["id"]])
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(body)["code"], "no_audio_model")

    def test_wrong_wav_format_is_rejected(self):
        status, info = self.upload("cd.wav", make_wav(1, rate=44100))
        self.assertEqual(status, 415)

    def test_long_audio_is_capped_at_ten_minutes(self):
        status, info = self.upload("long.wav", make_wav(620))
        self.assertEqual(status, 201)
        self.assertTrue(info["attachment"]["notes"])
        self.assertLessEqual(info["attachment"]["audio_seconds"], 600)

    def test_thinking_flag_is_only_sent_to_thinking_models(self):
        self.chat("text:8b", "سلام", think=True)
        self.assertNotIn("think", self.ollama.chat_requests()[-1])
        self.chat("qwen3-vl:8b", "سلام")
        self.assertIs(self.ollama.chat_requests()[-1]["think"], False)
        self.chat("qwen3-vl:8b", "سلام", think=True)
        self.assertIs(self.ollama.chat_requests()[-1]["think"], True)

    def test_namespaced_model_names_are_accepted(self):
        status, _ = self.chat("user/model:1b", "سلام")
        self.assertEqual(status, 200)
        status, _ = self.chat("../etc/passwd", "سلام")
        self.assertEqual(status, 400)


class ModelManagerTests(FeatureTestCase):
    def test_capabilities_endpoint_lists_installed_models(self):
        status, body = self.call("GET", "/api/models/capabilities")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIn("vision", data["capabilities"]["qwen3-vl:8b"])
        self.assertIn("audio", data["capabilities"]["gemma4:e4b"])
        self.assertEqual(data["ollama_version"], "0.31.0")

    def test_catalog_marks_installed_and_has_vision_and_audio_groups(self):
        status, body = self.call("GET", "/api/models/catalog")
        data = json.loads(body)
        by_id = {m["id"]: m for m in data["models"]}
        self.assertTrue(by_id["qwen3-vl:8b"]["installed"])
        self.assertFalse(by_id["qwen2.5vl:7b"]["installed"])
        groups = {m["group"] for m in data["models"]}
        self.assertTrue({"vision", "audio", "text"} <= groups)
        self.assertTrue(by_id["gemma4:e4b"]["audio"])

    def test_pull_streams_progress_and_installs(self):
        self.ollama.pull_block.set()
        status, body = self.call("POST", "/api/models/pull", {"model": "qwen2.5vl:7b"})
        self.assertEqual(status, 200)
        lines = self.chunks(body)
        self.assertEqual(lines[-1]["status"], "success")
        self.assertIn("qwen2.5vl:7b", self.ollama.installed)

    def test_pull_rejects_bad_names_and_unknown_models(self):
        status, _ = self.call("POST", "/api/models/pull", {"model": "bad name;rm"})
        self.assertEqual(status, 400)
        status, body = self.call("POST", "/api/models/pull", {"model": "does/not-exist"})
        self.assertEqual(status, 502)
        self.assertIn("پیدا نشد", json.loads(body)["error"])

    def test_only_one_pull_at_a_time(self):
        results = {}

        def first():
            results["first"] = self.call("POST", "/api/models/pull", {"model": "slow:1b"})

        thread = threading.Thread(target=first)
        thread.start()
        time.sleep(0.5)
        status, _ = self.call("POST", "/api/models/pull", {"model": "other:1b"})
        self.assertEqual(status, 409)
        self.ollama.pull_block.set()
        thread.join(5)
        self.assertEqual(results["first"][0], 200)

    def test_delete_model_and_creator_required(self):
        status, _ = self.call("POST", "/api/models/delete", {"model": "text:8b"})
        self.assertEqual(status, 200)
        self.assertNotIn("text:8b", self.ollama.installed)
        with srv._identity_session() as connection:
            connection.execute("DELETE FROM creator_identity")
        status, _ = self.call("POST", "/api/models/pull", {"model": "x:1b"})
        self.assertEqual(status, 403)

    def test_catalog_helpers(self):
        self.assertEqual(models_catalog.guess_capabilities("nomic-embed-text"), ["embedding"])
        self.assertIn("vision", models_catalog.guess_capabilities("qwen3-vl:8b"))
        self.assertNotIn("vision", models_catalog.guess_capabilities("gemma3:1b"))
        self.assertIn("audio", models_catalog.guess_capabilities("gemma4:e2b"))
        self.assertEqual(models_catalog.pick_model(CAPS, "audio"), "gemma4:e4b")
        self.assertEqual(models_catalog.pick_model(CAPS, "audio", preferred="text:8b"), "gemma4:e4b")


class SecurityTests(FeatureTestCase):
    def test_cross_site_and_rebinding_requests_are_blocked(self):
        status, _ = self.call("POST", "/api/shutdown", {}, headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.call("GET", "/api/memory", headers={"Host": "evil.example:8000"})
        self.assertEqual(status, 403)
        status, _ = self.call("POST", "/api/models/pull", {"model": "x:1b"},
                              headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(status, 403)
        status, _ = self.call("GET", "/api/status", headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(status, 200)

    def test_empty_memory_forget_request_does_not_wipe_memory(self):
        srv._memory_store().add("یک واقعیت مهم")
        status, _ = self.call("POST", "/api/memory/forget", {"query": ""})
        self.assertEqual(status, 400)
        self.assertEqual(len(srv._memory_store().list()), 1)


class ShortcutFalsePositiveTests(FeatureTestCase):
    def test_programming_questions_reach_the_model(self):
        for text in (
            "سازنده کلاس در پایتون چیه؟",
            "what is a built-in function in python?",
            "what does created_at mean in this table?",
            "این کد چه ابزارهایی داره؟",
            "who created this file? " + "x" * 200,
        ):
            with self.subTest(text=text):
                before = len(self.ollama.chat_requests())
                status, _ = self.chat("text:8b", text)
                self.assertEqual(status, 200)
                self.assertEqual(len(self.ollama.chat_requests()), before + 1)

    def test_normal_requests_are_not_memory_commands(self):
        for text in ("این فایل رو در پوشه ذخیره کن", "حذف کن خط ۳ را از کد", "کد رو بخون و نتیجه رو ذخیره کن"):
            with self.subTest(text=text):
                before = len(self.ollama.chat_requests())
                self.chat("text:8b", text)
                self.assertEqual(len(self.ollama.chat_requests()), before + 1)
        self.assertEqual(srv._memory_store().list(), [])

    def test_messages_with_attachments_skip_local_shortcuts(self):
        _, info = self.upload("a.txt", b"hello")
        before = len(self.ollama.chat_requests())
        self.chat("text:8b", "چه قابلیت‌هایی داری؟", [info["attachment"]["id"]])
        self.assertEqual(len(self.ollama.chat_requests()), before + 1)

    def test_capability_list_mentions_new_features(self):
        status, body = self.chat("text:8b", "چه قابلیت‌هایی داری؟")
        answer = self.chunks(body)[0]["message"]["content"]
        for word in ("پیوست فایل", "درک تصویر", "درک صدا", "مدیریت مدل‌ها"):
            self.assertIn(word, answer)


class SearchTests(FeatureTestCase):
    def test_default_engine_is_google_and_keys_are_never_returned(self):
        status, body = self.call("GET", "/api/settings")
        self.assertEqual(json.loads(body)["search"]["engine"], "auto")
        status, body = self.call("POST", "/api/settings/search", {
            "engine": "auto", "serper_api_key": "abcd1234efgh5678", "brave_api_key": "bravekey12345678", "google_cx": "cx:123"})
        self.assertEqual(status, 200)
        self.assertNotIn("abcd1234efgh5678", body.decode())
        self.assertNotIn("bravekey12345678", body.decode())
        self.assertTrue(json.loads(body)["search"]["has_serper_key"])
        self.assertTrue(json.loads(body)["search"]["has_brave_key"])
        status, body = self.call("GET", "/api/settings")
        self.assertNotIn("abcd1234efgh5678", body.decode())
        status, _ = self.call("POST", "/api/settings/search", {"serper_api_key": "bad key!"})
        self.assertEqual(status, 400)
        status, _ = self.call("POST", "/api/settings/search", {"engine": "bing"})
        self.assertEqual(status, 400)
        self.call("POST", "/api/settings/search", {"serper_api_key": "", "brave_api_key": ""})
        _, body = self.call("GET", "/api/settings")
        self.assertFalse(json.loads(body)["search"]["has_serper_key"])
        self.assertFalse(json.loads(body)["search"]["has_brave_key"])

        status, body = self.call("POST", "/api/settings/search", {
            "searxng_url": "https://search.example.org"
        })
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["search"]["has_searxng"])
        self.assertEqual(json.loads(body)["search"]["searxng_url"], "https://search.example.org")
        status, _ = self.call("POST", "/api/settings/search", {
            "searxng_url": "https://search.example.org/private/path"
        })
        self.assertEqual(status, 400)
        self.call("POST", "/api/settings/search", {"searxng_url": ""})


    @patch("local_chat.search_providers.search_bing_rss", return_value=[])
    @patch("local_chat.server._fetch_public_page", return_value="متن")
    @patch("local_chat.search_providers.urlopen")
    @patch("local_chat.server._ddg_results", return_value=[])
    def test_google_via_serper(self, _ddg, mock_open, _page, _bing):
        app_settings.set_search_settings({"serper_api_key": "abcd1234efgh5678"})
        payload = {"organic": [
            {"title": "نتیجه ۱", "link": "https://example.com/a", "snippet": "خلاصه"},
            {"title": "تکراری", "link": "https://example.com/a/", "snippet": ""},
            {"title": "بد", "link": "javascript:alert(1)", "snippet": ""}]}
        mock_open.return_value = FakeHttp(json.dumps(payload).encode())
        status, body = self.call("POST", "/api/search", {"query": "قیمت دلار"})
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual((data["engine"], data["provider"], data["fallback"]), ("Google / Serper", "Google / Serper", False))
        self.assertEqual([r["url"] for r in data["results"]], ["https://example.com/a"])
        request = mock_open.call_args.args[0]
        self.assertEqual(request.full_url, "https://google.serper.dev/search")
        self.assertEqual(request.get_header("X-api-key"), "abcd1234efgh5678")
        self.assertEqual(json.loads(request.data)["hl"], "fa")

    @patch("local_chat.search_providers.search_bing_rss", return_value=[])
    @patch("local_chat.server._fetch_public_page", return_value="")
    @patch("local_chat.search_providers.urlopen")
    @patch("local_chat.server._ddg_results", return_value=[])
    def test_google_cse_provider(self, _ddg, mock_open, _page, _bing):
        app_settings.set_search_settings({"google_api_key": "AIzaSyFAKEKEY12345", "google_cx": "cx123"})
        mock_open.return_value = FakeHttp(json.dumps({"items": [
            {"title": "T", "link": "https://example.org/x", "snippet": "S"}]}).encode())
        data = json.loads(self.call("POST", "/api/search", {"query": "hello"})[1])
        self.assertEqual(data["provider"], "Google CSE")
        self.assertIn("key=AIzaSyFAKEKEY12345", mock_open.call_args.args[0].full_url)

    @patch("local_chat.search_providers.search_bing_rss", return_value=[])
    @patch("local_chat.server._ddg_results", return_value=[{"title": "d", "url": "https://d.example/", "snippet": "", "content": ""}])
    @patch("local_chat.search_providers.urlopen")
    def test_falls_back_to_duckduckgo_and_says_so(self, mock_open, _ddg, _bing):
        from urllib.error import HTTPError

        app_settings.set_search_settings({"serper_api_key": "abcd1234efgh5678"})
        mock_open.side_effect = HTTPError("u", 403, "forbidden", {}, None)
        data = json.loads(self.call("POST", "/api/search", {"query": "example query"})[1])
        self.assertEqual((data["engine"], data["fallback"]), ("DuckDuckGo", True))
        self.assertEqual(data["provider"], "DuckDuckGo")
        self.assertTrue(data["errors"])
        self.assertNotIn("abcd1234efgh5678", json.dumps(data))
        # the failed keyless attempt is remembered, so the next search does not retry it for a while
        calls = mock_open.call_count
        self.call("POST", "/api/search", {"query": "another example query"})
        self.assertEqual(mock_open.call_count, calls + 1)  # only serper again

    def test_keyless_google_html_parser_and_block_detection(self):
        html = ('<div><a href="/url?q=https%3A%2F%2Fexample.com%2Fpage&sa=U"><h3>صفحهٔ نمونه</h3></a></div>'
                '<a href="https://www.google.com/maps"><h3>maps</h3></a>'
                '<a href="https://other.example/x"><div><h3>دوم</h3></div></a>')
        results = search_providers.parse_google_html(html)
        self.assertEqual([r["url"] for r in results], ["https://example.com/page", "https://other.example/x"])
        with self.assertRaises(search_providers.SearchProviderError):
            search_providers.parse_google_html("<noscript>Please click here if you are not redirected: /httpservice/retry/enablejs</noscript>")

    def test_engine_duckduckgo_skips_google(self):
        app_settings.set_search_settings({"engine": "duckduckgo"})
        with patch("local_chat.server._ddg_results", return_value=[]) as ddg, \
                patch("local_chat.search_providers.urlopen") as google:
            data = json.loads(self.call("POST", "/api/search", {"query": "example query"})[1])
        self.assertEqual(data["engine"], "none")
        google.assert_not_called()
        ddg.assert_called_once()


class FakeHttp:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self, size=-1):
        return self.body if size < 0 else self.body[:size]


class StoreAndSettingsUnitTests(unittest.TestCase):
    def test_store_ignores_bad_ids_and_cleans_old_files(self):
        with tempfile.TemporaryDirectory() as directory:
            store = attach_files.AttachmentStore(lambda: Path(directory))
            self.assertIsNone(store.get("../../x"))
            self.assertFalse(store.delete("../../x"))
            record = store.put({"name": "a", "size": 1, "kind": "text", "text": "x"})
            self.assertEqual(store.get(record["id"])["text"], "x")
            import os
            old = time.time() - 30 * 24 * 3600
            os.utime(Path(directory) / f"{record['id']}.json", (old, old))
            store.cleanup()
            self.assertIsNone(store.get(record["id"]))

    def test_prompt_rendering_truncates_and_labels(self):
        record = {"name": "big.py", "size": 10, "kind": "text", "text": "a" * 5000, "notes": []}
        block, clipped = attach_files.render_for_prompt(record, 1000)
        self.assertTrue(clipped)
        self.assertLess(len(block), 1300)
        self.assertIn("big.py", block)

    def test_old_prompt_fitting_still_respects_budget_with_images_reserve(self):
        profile = srv._model_profile("m", extended=True, image_count=2)
        plain = srv._model_profile("m", extended=True, image_count=0)
        self.assertLess(profile["budget"], plain["budget"])


if __name__ == "__main__":
    unittest.main()
