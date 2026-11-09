"""Regression tests for the v2 memory: multiline Persian commands, categories, dedupe, migration, retrieval."""
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from local_chat.memory import (MemoryStoreError, PersistentMemory, describe_save_report, format_listing,
                               parse_memory_command, split_entries, validate_entry)

# NOTE: replace with the real 12-model inventory of the user's machine; the test only needs 12 distinct names
OLLAMA_INVENTORY = (
    "gemma3:12b", "gemma3:27b", "qwen2.5:14b", "qwen2.5-coder:14b", "qwen3:8b", "qwen3:30b-a3b",
    "llama3.1:8b", "llama3.2:3b", "mistral:7b", "deepseek-r1:14b", "phi4:14b", "nomic-embed-text:latest",
)


class MultilineCommandTests(unittest.TestCase):
    def test_trigger_anywhere_single_and_multiline(self):
        block = "مدل‌های Ollama من:\n- gemma3:12b\n- qwen3:8b"
        cases = (
            "یادت باشه من فارسی صحبت می‌کنم",
            "من فارسی صحبت می‌کنم، اینو یادت باشه",
            "این اطلاعات رو ذخیره کن\n" + block,
            block + "\nاین اطلاعات رو ذخیره کن",
            block + "\nبه خاطر بسپار",
            "به خاطر بسپار:\n" + block,
            "یادت باشه\n" + block,
        )
        for text in cases:
            with self.subTest(text=text[:30]):
                action, content = parse_memory_command(text)
                self.assertEqual(action, "save")
                self.assertTrue(content)
                self.assertNotIn("یادت باشه", content)
                self.assertNotIn("ذخیره کن", content)

    def test_trigger_removed_but_everything_else_kept(self):
        _, content = parse_memory_command("این اطلاعات رو ذخیره کن\nمدل‌هام:\n- a1:7b\n- b2:8b")
        self.assertEqual([line.strip() for line in content.splitlines()], ["مدل‌هام:", "- a1:7b", "- b2:8b"])

    def test_ordinary_requests_are_not_commands(self):
        for text in ("این فایل رو در پوشه ذخیره کن", "فقط یادت باشه x رو بنویس", "کد رو بخون و نتیجه رو ذخیره کن",
                     "```python\nprint(1)\n```\nیادت باشه", "سلام چطوری؟"):
            with self.subTest(text=text):
                self.assertIsNone(parse_memory_command(text))

    def test_long_list_is_not_rejected_by_line_count(self):
        text = "این لیست رو ذخیره کن:\n" + "\n".join(f"- مورد شمارهٔ {i} از لیست طولانی" for i in range(60))
        action, content = parse_memory_command(text)
        self.assertEqual(action, "save")
        self.assertEqual(len(split_entries(content)), 60)


class SplittingAndValidationTests(unittest.TestCase):
    def test_header_prefixes_items_but_labelled_items_stay_alone(self):
        self.assertEqual(split_entries("مدل‌های من اینا هستن\n- a1:7b\n- b2:8b"),
                         ["مدل‌های من: a1:7b", "مدل‌های من: b2:8b"])
        self.assertEqual(split_entries("سخت‌افزارم اینه:\nکارت گرافیک: RTX 3060\nرم: 32GB"),
                         ["کارت گرافیک: RTX 3060", "رم: 32GB"])

    def test_invalid_entries_are_rejected_with_reason(self):
        for text in ("سلام", "امروز ساعت سه جلسه دارم", "password: hunter2hunter2", "چرا؟", "x", "..."):
            with self.subTest(text=text):
                ok, reason = validate_entry(text)
                self.assertFalse(ok)
                self.assertTrue(reason)
        self.assertTrue(validate_entry("همیشه امروز را با تاریخ شمسی بنویس")[0])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "memory.sqlite3"

    def tearDown(self):
        self.dir.cleanup()

    def test_all_twelve_models_survive_restart_and_are_retrievable(self):
        text = "مدل‌های Ollama من اینا هستن:\n" + "\n".join(f"- {name}" for name in OLLAMA_INVENTORY)
        store = PersistentMemory(self.path)
        action, content = parse_memory_command("این اطلاعات رو ذخیره کن\n" + text)
        report = store.save_text(content)
        self.assertEqual(len(report["saved"]), 12)
        self.assertEqual(report["rejected"], [])
        del store
        reopened = PersistentMemory(self.path)              # «application restart»
        stored = " ".join(item["content"] for item in reopened.list())
        for name in OLLAMA_INVENTORY:
            self.assertIn(name, stored)
        self.assertTrue(all(item["category"] == "model" for item in reopened.list()))
        context = reopened.prompt_context("چه مدل‌هایی دارم؟")
        for name in OLLAMA_INVENTORY:
            self.assertIn(name, context)

    def test_duplicates_and_value_updates(self):
        store = PersistentMemory(self.path)
        self.assertEqual(len(store.save_text("کارت گرافیک: RTX 3060 12GB")["saved"]), 1)
        again = store.save_text("کارت گرافیک: RTX 3060 12GB")
        self.assertEqual(len(again["duplicates"]), 1)
        changed = store.save_text("کارت گرافیک: RTX 4070")
        self.assertEqual(len(changed["updated"]), 1)
        self.assertEqual([item["content"] for item in store.list()], ["کارت گرافیک: RTX 4070"])
        self.assertEqual(describe_save_report(changed), describe_save_report(changed))
        # a model list must never be clobbered by the single-value rule
        store.save_text("مدل‌های من:\n- a1:7b\n- b2:8b")
        self.assertEqual(len([i for i in store.list() if i["category"] == "model"]), 2)

    def test_rejected_entries_are_reported_and_good_ones_still_saved(self):
        store = PersistentMemory(self.path)
        report = store.save_text("من فارسی صحبت می‌کنم\nامروز بارون میاد\nسلام")
        self.assertEqual(len(report["saved"]), 1)
        self.assertEqual(len(report["rejected"]), 2)
        self.assertIn("ذخیره نکردم", describe_save_report(report))
        self.assertEqual(len(store.list()), 1)

    def test_failed_write_is_never_reported_as_saved(self):
        store = PersistentMemory(self.path)
        with patch.object(PersistentMemory, "_upsert", side_effect=sqlite3.OperationalError("disk full")):
            with self.assertRaises(MemoryStoreError):
                store.save_text("من ویندوز استفاده می‌کنم")
        self.assertEqual(store.list(), [])                   # rolled back, nothing half-written

    def test_transaction_is_atomic_for_a_batch(self):
        store = PersistentMemory(self.path)
        calls = {"n": 0}
        original = PersistentMemory._upsert

        def flaky(self, db, content, now):
            calls["n"] += 1
            if calls["n"] == 3:
                raise sqlite3.OperationalError("boom")
            return original(self, db, content, now)

        with patch.object(PersistentMemory, "_upsert", flaky):
            with self.assertRaises(MemoryStoreError):
                store.save_text("مورد شمارهٔ یک\nمورد شمارهٔ دو\nمورد شمارهٔ سه")
        self.assertEqual(store.list(), [])

    def test_relevant_memories_beat_newest_ones(self):
        store = PersistentMemory(self.path)
        store.save_text("همیشه جواب‌ها را کوتاه بنویس")
        store.save_text("کارت گرافیک: RTX 3060 12GB")
        for index in range(40):
            store.save_text(f"فکت بی‌ربط شمارهٔ {index} دربارهٔ باغچه و گل‌های حیاط")
        context = store.prompt_context("کارت گرافیکم چی بود؟", max_items=6)
        self.assertIn("RTX 3060", context)                    # old but relevant
        self.assertIn("همیشه جواب‌ها را کوتاه", context)       # persistent instruction always present
        self.assertLessEqual(context.count("باغچه"), 3)
        self.assertEqual(store.prompt_context("") .count("- "), 30)

    def test_format_listing_groups_by_category(self):
        store = PersistentMemory(self.path)
        store.save_text("کارت گرافیک: RTX 3060\nهمیشه فارسی جواب بده")
        text = format_listing(store.list())
        self.assertIn("سخت‌افزار", text)
        self.assertIn("دستورهای دائمی", text)


class MigrationTests(unittest.TestCase):
    def test_v1_database_is_backed_up_and_all_memories_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            db = sqlite3.connect(path)
            db.execute("""CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT NOT NULL,
                normalized TEXT NOT NULL UNIQUE, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)""")
            now = int(time.time())
            old = [("1", "من فارسی صحبت می‌کنم"), ("2", "qwen3:8b روی سیستم من نصب است"),
                   ("3", "پروژهٔ LocalChat با پایتون است")]
            db.executemany("INSERT INTO memories VALUES (?,?,?,?,?)", [(i, c, c.casefold(), now, now) for i, c in old])
            db.commit()
            db.close()
            store = PersistentMemory(path)
            self.assertIsNotNone(store.last_backup)
            self.assertTrue(Path(store.last_backup).exists())
            items = {item["id"]: item for item in store.list()}
            self.assertEqual({i: items[i]["content"] for i in items}, dict(old))
            self.assertEqual(items["2"]["category"], "model")
            self.assertEqual(items["3"]["category"], "project")
            # backup still holds the untouched v1 data
            backup = sqlite3.connect(store.last_backup)
            self.assertEqual(backup.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 3)
            backup.close()
            before = len(list(Path(directory).glob("*.bak-*")))
            PersistentMemory(path)                           # second open: no new migration, no new backup
            self.assertEqual(len(list(Path(directory).glob("*.bak-*"))), before)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------- through the real HTTP chat endpoint
from local_chat import server as _srv                      # noqa: E402
from local_chat.test_features import FeatureTestCase       # noqa: E402


class ChatIntegrationTests(FeatureTestCase):
    def reply(self, status_body):
        status, body = status_body
        self.assertEqual(status, 200)
        return self.reply_text(self.chunks(body))

    def test_multiline_persian_command_saves_all_twelve_models_and_survives_restart(self):
        text = "این اطلاعات رو ذخیره کن\nمدل‌های Ollama من:\n" + "\n".join(f"- {name}" for name in OLLAMA_INVENTORY)
        before = len(self.ollama.chat_requests())
        answer = self.reply(self.chat("text:8b", text))
        self.assertIn("12 مورد جدید", answer)
        self.assertEqual(len(self.ollama.chat_requests()), before)         # handled locally, the model never saw it
        status, body = self.call("GET", "/api/memory")
        stored = " ".join(item["content"] for item in json.loads(body)["memories"])
        for name in OLLAMA_INVENTORY:
            self.assertIn(name, stored)
        reopened = PersistentMemory(_srv.MEMORY_STORE.path)                 # «restart»
        self.assertEqual(len(reopened.list()), 12)

    def test_stored_memories_reach_the_model_when_relevant(self):
        self.reply(self.chat("text:8b", "یادت باشه کارت گرافیک من RTX 3060 با ۱۲ گیگابایت است"))
        for index in range(35):
            _srv.MEMORY_STORE.save_text(f"نکتهٔ بی‌ربط شمارهٔ {index} دربارهٔ آشپزی و دستور غذا")
        self.reply(self.chat("text:8b", "کارت گرافیکم چیست؟"))
        system = self.ollama.chat_requests()[-1]["messages"][0]["content"]
        self.assertIn("RTX 3060", system)

    def test_forget_and_list_commands(self):
        self.reply(self.chat("text:8b", "به خاطر بسپار: من با ویندوز کار می‌کنم"))
        self.assertIn("ویندوز", self.reply(self.chat("text:8b", "چه چیزهایی یادت هست؟")))
        self.assertIn("حذف شد", self.reply(self.chat("text:8b", "فراموش کن: ویندوز")))
        self.assertEqual(_srv.MEMORY_STORE.list(), [])

    def test_ordinary_save_request_is_not_hijacked(self):
        self.reply(self.chat("text:8b", "این فایل رو توی پوشه ذخیره کن"))
        self.assertEqual(_srv.MEMORY_STORE.list(), [])

    def test_invalid_entries_are_reported_not_silently_saved(self):
        answer = self.reply(self.chat("text:8b", "یادت باشه امروز ساعت سه جلسه دارم"))
        self.assertIn("ذخیره نکردم", answer)
        self.assertEqual(_srv.MEMORY_STORE.list(), [])
