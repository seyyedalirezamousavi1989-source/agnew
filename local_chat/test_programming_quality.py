"""Quality tests for programming mode: feedback/modify flow, always-on project awareness, loop guard,
history compaction, static project analysis, tool discovery and (optionally) the real screenshot harness."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_chat import coding_agent, project_analysis, system_tools, toolfinder
from local_chat.coding_agent import classify_request
from local_chat.test_programming_e2e import ProgrammingE2ETests

S = "c" * 40


def make_project(root, name="chess", files=None):
    project = root / name
    for folder in ("src", "tests", "data", "output", "logs"):
        (project / folder).mkdir(parents=True)
    (project / "requirements.txt").write_text("pygame==2.6.1\n", encoding="utf-8")
    (project / "README.md").write_text("# chess\n", encoding="utf-8")
    for relative, content in (files or {}).items():
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return project


WEAK_MAIN = "import pygame\n\ndef main():\n    pygame.init()\n    screen = pygame.display.set_mode((300, 300))\n"
BOARD = "class ChessBoard:\n    def __init__(self):\n        self.board = [[None] * 8 for _ in range(8)]\n"
TOOL_LOG = ("پروژه: `chess`\n\n> ✅ **ساخت فایل جدید** — فایل src/main.py ساخته شد (1.4 KB).\n\n"
            "```text\nlong test output\n```\nمرحلهٔ 2 از 12: چیزی\nبازی ساخته شد.")


class ClassifierQualityTests(unittest.TestCase):
    HISTORY = [{"role": "user", "content": "بازی شطرنج بساز"}, {"role": "assistant", "content": TOOL_LOG}]

    def test_complaints_after_a_build_are_change_requests(self):
        for text in ("بازی گرافیکی نیست و صفحش اندازه کوچیکی داره و مهره هم تو بازی نیست",
                     "اندازه صفحه باید بزرگتر بشه و سیاه و سفید باشه", "بازهم هیچ تغییری نکرد"):
            self.assertEqual(classify_request(text, self.HISTORY, True), "action", text)

    def test_capability_question_is_still_only_a_discussion(self):
        self.assertEqual(classify_request("میتونی بازی بسازی؟", [], True), "discuss")

    def test_modification_detection(self):
        self.assertTrue(coding_agent.is_modification("بازهم هیچ تغییری نکرد", self.HISTORY, "chess"))
        self.assertFalse(coding_agent.is_modification("یک بازی جدید مار بساز", self.HISTORY, "chess"))
        self.assertFalse(coding_agent.is_modification("بازهم هیچ تغییری نکرد", self.HISTORY, ""))


class HelperTests(unittest.TestCase):
    def test_history_is_compacted_so_the_model_cannot_imitate_tool_logs(self):
        out = coding_agent.compact_history([{"role": "user", "content": "x"}, {"role": "assistant", "content": TOOL_LOG}])
        text = out[1]["content"]
        self.assertNotIn("long test output", text)
        self.assertNotIn("مرحلهٔ 2 از 12", text)
        self.assertIn("src/main.py", text)
        self.assertIn("بازی ساخته شد", text)

    def test_loop_detection_cuts_repeated_paragraphs_but_not_normal_text(self):
        paragraph = "بازی شطرنج ۲ بعدی ساده با مهره‌های ۳ بعدی و صفحه سیاه و سفید ایجاد شد. این بازی شامل صفحه شطرنج و مهره‌ها است.\n"
        looped = paragraph * 8
        cut = coding_agent.detect_loop(looped)
        self.assertIsNotNone(cut)
        self.assertLess(cut, len(looped) // 2)
        normal = "".join(f"جمله شمارهٔ {i} با محتوای متفاوت دربارهٔ موضوعی تازه و مستقل از قبلی.\n" for i in range(40))
        self.assertIsNone(coding_agent.detect_loop(normal))

    def test_active_project_comes_from_history_when_it_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_project(root, "chess")
            with patch.object(system_tools, "PROJECTS_ROOT", root):
                self.assertEqual(coding_agent.active_project_from_history([{"role": "assistant", "content": TOOL_LOG}]), "chess")
                self.assertEqual(coding_agent.active_project_from_history([{"role": "assistant", "content": "پروژه: `gone`"}]), "")


class ProjectAwarenessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patch = patch.object(system_tools, "PROJECTS_ROOT", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def do(self, action, **args):
        prepared = system_tools.prepare_action(action, args, S)
        return system_tools.approve_action(prepared["approval_id"], S)

    def test_context_lists_root_files_and_what_was_done(self):
        self.do("create_project", name="demo")
        self.do("write_file", project="demo", path="src/main.py", content="print('x')\n")
        context = system_tools.project_context(None)
        self.assertIn(str(self.root), context)
        self.assertIn("[demo]", context)
        self.assertIn("src/main.py", context)
        self.assertIn("write_file", context)  # the manifest log
        self.assertIn("آخرین عملیات", context)

    def test_empty_state_is_explicit(self):
        self.assertIn("هنوز هیچ پروژه‌ای", system_tools.project_context(None))

    def test_most_recent_project_is_active(self):
        make_project(self.root, "old")
        os.utime(self.root / "old", (1, 1))
        make_project(self.root, "fresh")
        self.assertEqual(system_tools.active_project_name(), "fresh")

    def test_check_project_finds_unwired_modules_and_missing_requirements(self):
        project = make_project(self.root, "chess", {
            "src/main.py": "import pygame\nfrom chessboard import ChessBoard\n\ndef main():\n    pass\n",
            "src/chessboard.py": BOARD, "src/rules.py": "import numpy\n\ndef legal(move):\n    pass\n",
            "src/broken.py": "def x(:\n"})
        report = system_tools.read_only_action("check_project", {"project": "chess"})
        text = " ".join(report["problems"])
        self.assertFalse(report["healthy"])
        self.assertIn("خطای نحوی در src/broken.py", text)
        self.assertIn("rules", text)           # never imported
        self.assertIn("numpy", text)           # not in requirements.txt
        self.assertIn("هیچ‌چیز را اجرا نمی‌کند", text)  # main.py has no __main__ guard
        self.assertTrue(any("legal" in item for item in report["stubs"]))

    def test_check_project_accepts_a_wired_pygame_project(self):
        main = ("import pygame\nfrom board import Board\n\ndef main():\n    pygame.init()\n    s = pygame.display.set_mode((900, 700))\n"
                "    b = Board()\n    while True:\n        for e in pygame.event.get():\n            pass\n        pygame.display.flip()\n\n"
                "if __name__ == '__main__':\n    main()\n")
        make_project(self.root, "ok", {"src/main.py": main, "src/board.py": "class Board:\n    pass\n",
                                        "tests/test_board.py": "import unittest\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n"})
        report = system_tools.read_only_action("check_project", {"project": "ok"})
        self.assertTrue(report["healthy"], report["problems"])
        self.assertEqual(report["gui"]["window_size_literal"], [900, 700])

    def test_snapshot_contains_real_code_entry_first(self):
        project = make_project(self.root, "chess", {"src/main.py": WEAK_MAIN, "src/chessboard.py": BOARD})
        text = project_analysis.snapshot(project, 20000)
        self.assertLess(text.index("src/main.py"), text.index("src/chessboard.py"))
        self.assertIn("set_mode((300, 300))", text)

    def test_screenshot_arguments_are_validated(self):
        make_project(self.root, "chess")
        (self.root / "chess" / ".venv" / "Scripts").mkdir(parents=True)
        (self.root / "chess" / ".venv" / "bin").mkdir(parents=True)
        (self.root / "chess" / ".venv" / "Scripts" / "python.exe").write_text("")
        (self.root / "chess" / ".venv" / "bin" / "python").write_text("")
        (self.root / "chess" / "src" / "main.py").write_text("x = 1\n")
        for bad in ({"clicks": [[1, 2, 3]]}, {"clicks": [[-5, 2]]}, {"clicks": [[1, 1]] * 13}, {"keys": ["ESCAPE"]}, {"keys": ["rm -rf"]}):
            with self.assertRaises(system_tools.SystemToolError, msg=str(bad)):
                system_tools.prepare_action("screenshot_check", {"project": "chess", **bad}, S)
        good = system_tools.prepare_action("screenshot_check", {"project": "chess", "clicks": [[10, 20]], "keys": ["RETURN"]}, S)
        self.assertEqual(good["plan"]["spec"]["clicks"], [[10, 20]])


class ToolFinderTests(unittest.TestCase):
    def test_user_pinned_path_wins_and_unknown_tools_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / ("docker.exe" if os.name == "nt" else "mydocker")
            exe.write_text("")
            data = Path(tmp) / "data"
            data.mkdir()
            (data / "tool_paths.json").write_text(json.dumps({"docker": str(exe)}), encoding="utf-8")
            with patch.object(toolfinder.app_settings, "app_data_dir", lambda: data):
                self.assertEqual(toolfinder.find("docker"), str(exe))
                self.assertIsNone(toolfinder.find("not-a-tool"))

    def test_well_known_folders_are_searched_outside_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            flutter = home / "flutter" / "bin" / ("flutter.bat" if os.name == "nt" else "flutter")
            flutter.parent.mkdir(parents=True)
            flutter.write_text("")
            with patch.dict(os.environ, {"PATH": "", "USERPROFILE": str(home), "HOME": str(home)}), \
                    patch.object(toolfinder, "_home", lambda: home), patch.object(toolfinder, "_configured", lambda: {}):
                self.assertEqual(toolfinder.find("flutter"), str(flutter))

    def test_docker_found_outside_path_is_used_by_docker_runtime(self):
        from local_chat import docker_runtime
        with patch.object(toolfinder, "find", lambda name: "/opt/x/docker" if name == "docker" else None):
            self.assertEqual(docker_runtime._docker_exe(), "/opt/x/docker")


class QualityE2ETests(ProgrammingE2ETests):
    """Re-uses the server/fake-Ollama fixture of the e2e tests (their own tests are skipped via the empty loader below)."""

    def test_feedback_after_a_build_edits_the_real_code_and_sees_it(self):
        make_project(self.root, "chess", {"src/main.py": WEAK_MAIN, "src/chessboard.py": BOARD,
                                           "src/helpers.py": "def f():\n    return 1\n"})
        plan = {"decision": "plan", "title": "اصلاح", "summary": "s", "project": "chess",
                "spec": ["پنجره ۱۰۰۰×۸۰۰", "سیاه و سفید"],
                "steps": [{"title": "خواندن", "detail": "", "tool": "read_file"}, {"title": "ویرایش", "detail": "", "tool": "edit_file"}]}
        rounds = iter([
            {"tool_calls": [("read_file", {"project": "chess", "path": "src/main.py"}),
                            ("edit_file", {"project": "chess", "path": "src/main.py", "old_text": "(300, 300)", "new_text": "(1000, 800)"})]},
            {"content": ["انجام شد."]}])
        self.ollama.script = lambda kind, payload: ({"content": json.dumps(plan)} if kind == "plan" else next(rounds))
        history = [{"role": "user", "content": "بازی شطرنج بساز"}, {"role": "assistant", "content": TOOL_LOG}]
        events, text = self.browser.chat("اندازه صفحه باید بزرگتر بشه و سیاه و سفید باشه", history=history,
                                         on_event=self.answer_cards(True, True))
        self.assertNotIn(coding_agent.CLARIFY_HEADER, text, "feedback on an existing project must not start a new interview")
        self.assertIn("(1000, 800)", (self.root / "chess" / "src" / "main.py").read_text())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["chess"], "must not create a second project")
        planner_system = self.ollama.requests[0][1]["messages"][0]["content"]
        self.assertIn("chess", planner_system)
        self.assertIn("set_mode((300, 300))", planner_system, "the planner sees the real current code")
        loop_systems = [p["messages"][0]["content"] for k, p in self.ollama.requests if k == "loop"]
        self.assertTrue(all("مشخصات الزامی" in item and "پنجره ۱۰۰۰×۸۰۰" in item for item in loop_systems))
        self.assertIn("set_mode((300, 300))", loop_systems[0])
        self.assertIn("set_mode((1000, 800))", loop_systems[-1], "the code snapshot is refreshed after the edit")
        compacted = self.ollama.requests[0][1]["messages"][1:]
        self.assertFalse(any("long test output" in m["content"] for m in compacted))

    def test_every_phase_knows_the_projects_on_disk(self):
        make_project(self.root, "chess", {"src/main.py": WEAK_MAIN})
        self.ollama.script = lambda kind, payload: {"content": ["پاسخ"]}
        self.browser.chat("تفاوت list و tuple را توضیح بده")
        system = self.ollama.requests[0][1]["messages"][0]["content"]
        self.assertIn(str(self.root), system)
        self.assertIn("[chess]", system)
        self.assertIn("src/main.py", system)

    def test_repetition_loop_is_cut_and_reported(self):
        paragraph = "این یک بند کاملاً تکراری است که مدل بارها و بارها می‌نویسد و باید قطع شود تا کاربر منتظر نماند.\n"
        self.ollama.script = lambda kind, payload: {"content": [paragraph] * 60}
        events, text = self.browser.chat("تفاوت list و tuple را توضیح بده")
        self.assertIn("حلقهٔ تکرار", text)
        self.assertLess(text.count("کاملاً تکراری"), 15)
        self.assertTrue(events[-1]["done"])

    def test_no_planner_nonsense_questions_rule_is_in_the_prompt(self):
        self.assertIn("never about money", coding_agent.PLANNER_SYSTEM_PROMPT)


def load_tests(loader, tests, pattern):
    """Do not re-run the inherited e2e tests here; only this module's own tests."""
    suite = unittest.TestSuite()
    for case in (ClassifierQualityTests, HelperTests, ProjectAwarenessTests, ToolFinderTests, RealToolTests):
        suite.addTests(loader.loadTestsFromTestCase(case))
    for name in ("test_feedback_after_a_build_edits_the_real_code_and_sees_it", "test_every_phase_knows_the_projects_on_disk",
                 "test_repetition_loop_is_cut_and_reported", "test_no_planner_nonsense_questions_rule_is_in_the_prompt"):
        suite.addTest(QualityE2ETests(name))
    return suite


@unittest.skipUnless(os.environ.get("LOCALCHAT_REAL_TOOLS") == "1", "set LOCALCHAT_REAL_TOOLS=1 (creates a venv, installs pygame from PyPI)")
class RealToolTests(unittest.TestCase):
    def test_real_screenshot_check_on_a_pygame_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(system_tools, "PROJECTS_ROOT", root):
                def do(action, **args):
                    return system_tools.approve_action(system_tools.prepare_action(action, args, S)["approval_id"], S)
                self.assertTrue(do("create_project", name="g")["ok"])
                self.assertTrue(do("create_venv", project="g")["ok"])
                self.assertTrue(do("venv_install", project="g", packages=["pygame"])["ok"])
                main = ("import pygame, sys\nfrom board import Board\n\ndef main():\n    pygame.init()\n    screen = pygame.display.set_mode((800, 600))\n"
                        "    board = Board()\n    while True:\n        for e in pygame.event.get():\n            if e.type == pygame.MOUSEBUTTONDOWN:\n"
                        "                board.click(e.pos)\n        screen.fill((20, 20, 20))\n        board.draw(screen)\n        pygame.display.flip()\n\n"
                        "if __name__ == '__main__':\n    main()\n")
                board = ("import pygame\n\nclass Board:\n    def __init__(self):\n        self.sel = None\n    def click(self, pos):\n        self.sel = pos\n"
                         "    def draw(self, s):\n        pygame.draw.rect(s, (230, 230, 230), (0, 0, 400, 300))\n        if self.sel:\n"
                         "            pygame.draw.circle(s, (200, 0, 0), self.sel, 30)\n")
                self.assertTrue(do("write_file", project="g", path="src/main.py", content=main)["ok"])
                self.assertTrue(do("write_file", project="g", path="src/board.py", content=board)["ok"])
                result = do("screenshot_check", project="g", clicks=[[500, 400]])
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["window"], [800, 600])
                self.assertTrue(any("after_click_1" in note and "تغییر کرد" in note for note in result["observations"]), result["observations"])
                self.assertTrue((root / "g" / "output" / "screenshots" / "start.png").is_file())
                # a stub main that never draws is reported honestly
                (root / "g" / "src" / "main.py").write_text("def main():\n    pass\n\nif __name__ == '__main__':\n    main()\n")
                result = do("screenshot_check", project="g")
                self.assertFalse(result["ok"])
                self.assertIn("هیچ فریمی", result["summary"])


if __name__ == "__main__":
    unittest.main()
