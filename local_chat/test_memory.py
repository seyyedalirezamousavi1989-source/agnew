import tempfile
import unittest
from pathlib import Path

from local_chat.memory import PersistentMemory, parse_memory_command


class MemoryCommandTests(unittest.TestCase):
    def test_save_phrases_in_colloquial_persian(self):
        cases = {
            "یادت باشه من فارسی‌زبانم": "من فارسی‌زبانم",
            "این رو ذخیره کن: من قهوه دوست دارم": "من قهوه دوست دارم",
            "من از پاسخ کوتاه خوشم میاد، اینو یادت باشه": "من از پاسخ کوتاه خوشم میاد",
            "به خاطر بسپار که من ویندوز استفاده می‌کنم": "من ویندوز استفاده می‌کنم",
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(parse_memory_command(phrase), ("save", expected))

    def test_forget_and_list_phrases(self):
        self.assertEqual(parse_memory_command("این رو فراموش کن: من قهوه دوست دارم"),
                         ("forget", "من قهوه دوست دارم"))
        self.assertEqual(parse_memory_command("همه حافظه رو پاک کن"), ("forget_all", ""))
        self.assertEqual(parse_memory_command("چه چیزهایی یادت هست؟"), ("list", ""))


class PersistentMemoryTests(unittest.TestCase):
    def test_memory_persists_deduplicates_and_can_be_forgotten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            memory = PersistentMemory(path)
            first = memory.add("کاربر فارسی صحبت می‌کند")
            second = memory.add("کاربر فارسی صحبت می‌کند")
            self.assertFalse(first["duplicate"])
            self.assertTrue(second["duplicate"])
            reloaded = PersistentMemory(path)
            self.assertEqual(len(reloaded.list()), 1)
            self.assertEqual(reloaded.forget("فارسی"), 1)
            self.assertEqual(reloaded.list(), [])

    def test_prompt_context_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentMemory(Path(directory) / "memory.sqlite3")
            memory.add("ا" * 1000)
            self.assertLessEqual(len(memory.prompt_context(max_chars=100)), 100)


if __name__ == "__main__":
    unittest.main()
