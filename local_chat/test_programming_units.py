"""Unit tests: request classifier, system_tools safety rules, docker_runtime and procutil basics."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_chat import coding_agent, procutil, system_tools
from local_chat.coding_agent import classify_request


S1 = "a" * 40
S2 = "b" * 40


class ClassifierTests(unittest.TestCase):
    def test_capability_questions_are_discussion_not_action(self):
        for text in ("میتونی بازی بسازی؟", "می‌تونی یه بازی مار بسازی", "can you build a snake game?"):
            self.assertEqual(classify_request(text), "discuss", text)

    def test_clear_orders_are_actions(self):
        for text in ("یک بازی حدس عدد با پایتون بساز", "build a todo app in python"):
            self.assertEqual(classify_request(text), "action", text)

    def test_empty_is_chat(self):
        self.assertEqual(classify_request("   "), "chat")

    def test_plain_explanation_is_chat(self):
        self.assertEqual(classify_request("تفاوت list و tuple را توضیح بده"), "chat")


class SystemToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patch = patch.object(system_tools, "PROJECTS_ROOT", root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def prepare(self, action, args, session=S1):
        return system_tools.prepare_action(action, args, session)

    def test_path_traversal_and_disallowed_locations_rejected(self):
        approved = self.prepare("create_project", {"name": "demo"})
        system_tools.approve_action(approved["approval_id"], S1)
        for bad in ("../evil.py", "/etc/passwd", "src/../../x.py", "secrets/key.py", "src/a.exe"):
            with self.assertRaises(system_tools.SystemToolError, msg=bad):
                p = self.prepare("write_file", {"project": "demo", "path": bad, "content": "x"})
                system_tools.approve_action(p["approval_id"], S1)

    def test_approval_is_one_use_and_bound_to_session(self):
        p = self.prepare("create_project", {"name": "once"})
        with self.assertRaises(system_tools.SystemToolError):
            system_tools.approve_action(p["approval_id"], S2)
        system_tools.approve_action(p["approval_id"], S1)
        with self.assertRaises(system_tools.SystemToolError):
            system_tools.approve_action(p["approval_id"], S1)

    def test_edit_file_requires_unique_old_text_and_keeps_backup(self):
        system_tools.approve_action(self.prepare("create_project", {"name": "ed"})["approval_id"], S1)
        system_tools.approve_action(self.prepare("write_file", {"project": "ed", "path": "src/a.py", "content": "x = 1\nx = 1\n"})["approval_id"], S1)
        with self.assertRaises(system_tools.SystemToolError):
            system_tools.approve_action(self.prepare("edit_file", {"project": "ed", "path": "src/a.py", "old_text": "x = 1", "new_text": "x = 2"})["approval_id"], S1)
        system_tools.approve_action(self.prepare("write_file", {"project": "ed", "path": "src/b.py", "content": "y = 1\n"})["approval_id"], S1)
        system_tools.approve_action(self.prepare("edit_file", {"project": "ed", "path": "src/b.py", "old_text": "y = 1", "new_text": "y = 2"})["approval_id"], S1)
        base = Path(self.tmp.name) / "ed"
        self.assertEqual((base / "src/b.py").read_text(), "y = 2\n")
        self.assertTrue(any((base / ".history").rglob("*b.py*")))


class ProcutilTests(unittest.TestCase):
    def test_run_capture_captures_output_and_times_out(self):
        ok = procutil.run_capture([sys.executable, "-c", "print('hi')"], timeout=20)
        self.assertEqual(ok.returncode, 0)
        self.assertIn("hi", ok.stdout)
        slow = procutil.run_capture([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
        self.assertTrue(slow.timed_out)


if __name__ == "__main__":
    unittest.main()
