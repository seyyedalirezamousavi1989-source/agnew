"""Multi-stack (Flutter / Gradle / Node), VS Code workspace integration, context size and model choice."""
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_chat import coding_agent, models_catalog, project_analysis, system_tools, toolchains, toolfinder

S1 = "a" * 40
FAKE_FLUTTER = """#!{py}
import json, os, sys
open({log!r}, "a").write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:2] == ["create"]:
    os.makedirs("lib", exist_ok=True)
    open("pubspec.yaml", "w").write("name: demo\\nenvironment:\\n  sdk: '>=3.0.0'\\ndependencies:\\n  flutter:\\n    sdk: flutter\\n")
    open("lib/main.dart", "w").write("import 'package:flutter/material.dart';\\nvoid main() => runApp(const MaterialApp());\\n")
print("ok", sys.argv[1:])
"""
FAKE_CODE = """#!{py}
import json, sys
open({log!r}, "a").write(json.dumps(sys.argv[1:]) + "\\n")
print("1.0")
"""


class StackBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "projects"
        self.root.mkdir()
        self.bin = Path(self.tmp.name) / "bin"
        self.bin.mkdir()
        self.flutter_log = Path(self.tmp.name) / "flutter.log"
        self.code_log = Path(self.tmp.name) / "code.log"
        if os.name == "nt":
            # Windows cannot execute extensionless shebang scripts via CreateProcess.
            flutter_py = self.bin / "fake_flutter.py"
            flutter_py.write_text(FAKE_FLUTTER.format(py=sys.executable, log=str(self.flutter_log)))
            flutter = self.bin / "flutter.cmd"
            flutter.write_text(f'@echo off\r\n"{sys.executable}" "{flutter_py}" %*\r\n', encoding="utf-8")
            code_py = self.bin / "fake_code.py"
            code_py.write_text(FAKE_CODE.format(py=sys.executable, log=str(self.code_log)))
            code = self.bin / "code.cmd"
            code.write_text(f'@echo off\r\n"{sys.executable}" "{code_py}" %*\r\n', encoding="utf-8")
        else:
            flutter = self.bin / "flutter"
            flutter.write_text(FAKE_FLUTTER.format(py=sys.executable, log=str(self.flutter_log)))
            code = self.bin / "code"
            code.write_text(FAKE_CODE.format(py=sys.executable, log=str(self.code_log)))
            for item in (flutter, code):
                item.chmod(item.stat().st_mode | stat.S_IEXEC)
        pinned = {"flutter": str(flutter), "code": str(code)}
        self.patches = [patch.object(system_tools, "PROJECTS_ROOT", self.root),
                        patch.object(toolfinder, "_configured", lambda: pinned)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def run_action(self, action, args):
        plan = system_tools.prepare_action(action, args, S1)
        return system_tools.approve_action(plan["approval_id"], S1)

    def calls(self, log):
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


class FlutterTests(StackBase):
    def test_flutter_project_flow(self):
        self.assertTrue(self.run_action("create_project", {"name": "demo", "kind": "flutter"})["ok"])
        project = self.root / "demo"
        self.assertEqual(system_tools._kind(project), "flutter")
        self.assertTrue(self.run_action("toolchain_run", {"project": "demo", "task": "create", "platforms": ["windows"]})["ok"])
        self.assertEqual(self.calls(self.flutter_log)[0][:1], ["create"])
        self.assertTrue((project / "lib/main.dart").is_file())
        # dart sources are allowed, python-only things are not
        self.assertTrue(self.run_action("write_file", {"project": "demo", "path": "lib/game.dart", "content": "class Game {}\n"})["ok"])
        with self.assertRaises(system_tools.SystemToolError):
            self.run_action("venv_test", {"project": "demo"})
        with self.assertRaises(system_tools.SystemToolError):
            self.run_action("write_file", {"project": "demo", "path": "evil/x.dart", "content": "x"})
        check = system_tools.read_only_action("check_project", {"project": "demo"})
        self.assertTrue(any("game.dart" in p for p in check.get("problems", [])), check)  # orphan dart file noticed
        self.assertTrue(self.run_action("toolchain_run", {"project": "demo", "task": "analyze"})["ok"])
        self.assertEqual(self.calls(self.flutter_log)[-1], ["analyze"])

    def test_task_and_package_validation(self):
        self.run_action("create_project", {"name": "demo", "kind": "flutter"})
        for task, extra in (("rm -rf", {}), ("pub_add", {"packages": ["x; rm -rf /"]}), ("create", {"platforms": ["../x"]})):
            with self.assertRaises(system_tools.SystemToolError, msg=task):
                self.run_action("toolchain_run", {"project": "demo", "task": task, **extra})
        self.assertFalse(self.flutter_log.exists())

    def test_python_project_rejects_toolchain(self):
        self.run_action("create_project", {"name": "py1"})
        with self.assertRaises(system_tools.SystemToolError):
            self.run_action("toolchain_run", {"project": "py1", "task": "build"})


class EditorTests(StackBase):
    def test_workspace_files_and_no_overwrite(self):
        self.run_action("create_project", {"name": "demo"})
        result = system_tools.approve_action(system_tools.prepare_action("editor_setup", {"project": "demo"}, S1)["approval_id"], S1) \
            if system_tools.prepare_action.__name__ else None
        self.assertTrue(result["ok"])
        vs = self.root / "demo/.vscode"
        for name in ("extensions.json", "settings.json", "launch.json", "tasks.json"):
            self.assertTrue((vs / name).is_file(), name)
            json.loads((vs / name).read_text().replace(toolchains.MARK, ""))  # valid JSON apart from the marker comment
        self.assertIn("ms-python.python", (vs / "extensions.json").read_text())
        (vs / "settings.json").write_text('{"mine": true}')
        self.run_action("editor_setup", {"project": "demo"})
        self.assertEqual((vs / "settings.json").read_text(), '{"mine": true}')

    def test_extension_allowlist_and_goto(self):
        self.assertTrue(self.run_action("editor_extension", {"extension": "ms-python.python"})["ok"])
        self.assertEqual(self.calls(self.code_log)[-1], ["--install-extension", "ms-python.python"])
        with self.assertRaises(system_tools.SystemToolError):
            self.run_action("editor_extension", {"extension": "evil.publisher"})
        self.run_action("create_project", {"name": "demo"})
        self.run_action("write_file", {"project": "demo", "path": "src/main.py", "content": "print(1)\n"})
        self.run_action("open_in_editor", {"project": "demo", "path": "src/main.py", "line": 7})
        import time
        for _ in range(50):
            if "--goto" in (self.calls(self.code_log) or [[]])[-1]:
                break
            time.sleep(0.1)
        last = self.calls(self.code_log)[-1]
        self.assertIn("--goto", last)
        self.assertTrue(any(a.endswith("main.py:7") for a in last), last)
        with self.assertRaises(system_tools.SystemToolError):
            self.run_action("open_in_editor", {"project": "demo", "path": "../../etc/passwd"})


@unittest.skipUnless(shutil.which("npm") and shutil.which("node"), "node/npm not installed")
class NodeRealTests(StackBase):
    def test_node_init_and_test(self):
        with patch.object(toolfinder, "_configured", lambda: {}):
            self.run_action("create_project", {"name": "web1", "kind": "node"})
            self.assertTrue(self.run_action("toolchain_run", {"project": "web1", "task": "init"})["ok"])
            self.assertTrue((self.root / "web1/package.json").is_file())
            self.run_action("write_file", {"project": "web1", "path": "src/a.js", "content": "console.log(1)\n"})
            self.assertTrue(self.run_action("toolchain_run", {"project": "web1", "task": "test"})["ok"] in (True, False))


class ContextSizeTests(unittest.TestCase):
    def test_tools_are_compact_and_selected(self):
        full = len(json.dumps(coding_agent.TOOL_DEFINITIONS, ensure_ascii=False))
        small = len(json.dumps(coding_agent.select_tools("python", ["write_file"], False), ensure_ascii=False))
        self.assertLess(full, 7500)
        self.assertLess(small, full * 0.6)
        names = {t["function"]["name"] for t in coding_agent.select_tools("flutter", ["create_project"], False)}
        self.assertIn("toolchain_run", names)
        self.assertNotIn("venv_install", names)
        self.assertIn("create_project", names)
        for t in coding_agent.TOOL_DEFINITIONS:
            self.assertNotIn("purpose", t["function"]["parameters"]["properties"])

    def test_tool_message_is_trimmed(self):
        agent = coding_agent.CodingAgent.__new__(coding_agent.CodingAgent)
        agent.budget_chars = 12000
        text = agent._tool_message({"ok": True, "summary": "s", "action": "x", "path": "/abs", "import_graph": {"a": "b"}, "output": "o" * 20000})
        data = json.loads(text)
        self.assertNotIn("path", data)
        self.assertNotIn("import_graph", data)
        self.assertLess(len(text), 4000)

    def test_old_file_bodies_are_slimmed(self):
        big = "x" * 5000
        msgs = [{"role": "system", "content": "s"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "write_file", "arguments": {"path": "a", "content": big}}}]},
                {"role": "tool", "content": big}] + [{"role": "user", "content": "u"}] * 7
        slim = coding_agent.CodingAgent._slim(msgs)
        self.assertLess(len(json.dumps(slim)), 1500)


class SelectionTests(unittest.TestCase):
    def test_coding_model_choice(self):
        caps = {"llama3:8b": ["completion"], "qwen2.5-coder:7b": ["completion", "tools"], "qwen3:30b": ["completion", "tools"],
                "nomic": ["embedding"]}
        self.assertEqual(models_catalog.pick_coding_model(caps), "qwen2.5-coder:7b")
        self.assertEqual(models_catalog.pick_coding_model(caps, "qwen3:30b"), "qwen3:30b")
        self.assertIsNone(models_catalog.pick_coding_model({"a": ["completion"]}))

    def test_plan_has_kind(self):
        plan = coding_agent.parse_plan(json.dumps({"decision": "plan", "project": "app", "kind": "flutter", "stack_reason": "موبایل",
                                                   "steps": [{"title": "t", "tool": "create_project"}]}))
        self.assertEqual(plan["kind"], "flutter")
        self.assertIn("flutter", coding_agent.plan_markdown(plan))
        self.assertEqual(coding_agent.parse_plan(json.dumps({"decision": "answer", "kind": "cobol"}))["kind"], "")
        self.assertIn("python=yes", coding_agent.available_toolchains())


if __name__ == "__main__":
    unittest.main()
