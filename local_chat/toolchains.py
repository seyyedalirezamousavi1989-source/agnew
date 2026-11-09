"""Fixed, allow-listed command templates for non-Python stacks (Flutter, Gradle, Node) and for VS Code.

Nothing here runs a command; it only *builds* argument lists (never a shell string) from validated inputs.
The model chooses a ``task`` from a short list and a few validated parameters; it never writes a command line.
"""
import json
import os
import re
from pathlib import Path

if __package__:
    from . import toolfinder
else:
    import toolfinder

WINDOWS = os.name == "nt"


class ToolchainError(Exception):
    pass


# ----------------------------------------------------------------------------- project kinds
COMMON_EXT = {".txt", ".md", ".json", ".csv", ".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".toml", ".yaml", ".yml",
              ".ini", ".cfg", ".xml", ".sql", ".svg"}
KINDS = {
    "python": {"label": "Python", "dirs": ("src", "tests", "data", "assets", "docs"),
               "root": {"requirements.txt", "README.md", ".gitignore"}, "ext": COMMON_EXT | {".py"}},
    "flutter": {"label": "Flutter/Dart", "dirs": ("lib", "test", "assets", "docs", "web"),
                "root": {"pubspec.yaml", "analysis_options.yaml", "README.md", ".gitignore"}, "ext": COMMON_EXT | {".dart"}},
    "gradle": {"label": "Gradle (Java/Kotlin)", "dirs": ("src", "app", "docs"),
               "root": {"build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", "gradle.properties",
                        "README.md", ".gitignore"},
               "ext": COMMON_EXT | {".java", ".kt", ".kts", ".gradle", ".properties", ".groovy"}},
    "node": {"label": "Node.js", "dirs": ("src", "test", "public", "docs"),
             "root": {"package.json", "README.md", ".gitignore", "index.js", "index.html", "tsconfig.json", "vite.config.js"},
             "ext": COMMON_EXT | {".mjs", ".cjs", ".vue", ".scss"}},
}
KIND_TOOL = {"flutter": "flutter", "gradle": "gradle", "node": "npm"}
# gradle sources live in src/** (root project) or app/** (gradle init layout): both are allowed for the "gradle" kind.


def policy(kind):
    return KINDS.get(kind) or KINDS["python"]


# ----------------------------------------------------------------------------- validation
PUB_PACKAGE = re.compile(r"^[a-z][a-z0-9_]{0,60}(:\^?[0-9][0-9A-Za-z.+-]{0,20})?$")
NPM_PACKAGE = re.compile(r"^(@[a-z0-9][a-z0-9._-]{0,60}/)?[a-z0-9][a-z0-9._-]{0,60}(@[0-9A-Za-z.^~<>=*-]{1,20})?$")
FLUTTER_PLATFORMS = {"windows", "android", "web", "linux", "macos", "ios"}
GRADLE_TYPES = {"java-application", "kotlin-application", "java-library", "kotlin-library", "basic"}
GRADLE_DSL = {"kotlin", "groovy"}
TASKS = {
    "flutter": {"create", "pub_get", "pub_add", "analyze", "test", "build_windows", "build_apk", "build_web", "doctor"},
    "gradle": {"init", "build", "test", "run", "clean"},
    "node": {"init", "install", "add", "test", "build", "start"},
}
TASK_HELP = {
    "flutter": "create (platforms), pub_get, pub_add (packages), analyze, test, build_windows, build_apk, build_web, doctor",
    "gradle": "init (type, dsl), build, test, run, clean",
    "node": "init, install, add (packages), test, build, start",
}
EXTENSIONS = {
    "python": ["ms-python.python", "ms-python.vscode-pylance"],
    "flutter": ["Dart-Code.dart-code", "Dart-Code.flutter"],
    "gradle": ["vscjava.vscode-java-pack", "vscjava.vscode-gradle"],
    "node": ["dbaeumer.vscode-eslint", "esbenp.prettier-vscode"],
}
EXTENSION_ALLOWLIST = {ext for items in EXTENSIONS.values() for ext in items} | {"ms-azuretools.vscode-docker", "fwcd.kotlin"}


def _tool_env(*executables):
    path_parts = [str(Path(exe).parent) for exe in executables if exe]
    git = toolfinder.find("git")
    if git:
        path_parts.append(str(Path(git).parent))
    java = toolfinder.find("java")
    # Preserve the host environment (SystemRoot, TEMP, USERPROFILE, APPDATA,
    # COMSPEC, etc.). A minimal environment can make .bat/.cmd toolchains fail
    # on Windows even when the executable is correctly discovered.
    env = dict(os.environ)
    env.update({"PATH": os.pathsep.join(path_parts + [os.environ.get("PATH", "")]), "CI": "true",
                "FLUTTER_SUPPRESS_ANALYTICS": "true", "DART_SUPPRESS_ANALYTICS": "true",
                "NPM_CONFIG_UPDATE_NOTIFIER": "false", "GRADLE_OPTS": "-Dorg.gradle.console=plain"})
    if java and not os.environ.get("JAVA_HOME"):
        home = Path(java).resolve().parent.parent
        if home.name and (home / "bin").is_dir():
            env["JAVA_HOME"] = str(home)
    return env


def _packages(value, pattern, label):
    if not isinstance(value, list) or not 1 <= len(value) <= 6 or any(not isinstance(item, str) or not pattern.fullmatch(item) for item in value):
        raise ToolchainError(f"packages باید فهرست ۱ تا ۶ نام معتبر {label} باشد.")
    return list(value)


def _flutter_name(slug):
    name = re.sub(r"[^a-z0-9_]", "_", slug.lower()).strip("_") or "app"
    return name if name[0].isalpha() else "app_" + name


def build(project, kind, task, args):
    """Return {argv, cwd, timeout, env, text, tool, ok_on_timeout} for an allow-listed task or raise ToolchainError."""
    if kind not in KIND_TOOL:
        raise ToolchainError("این پروژه از نوع Python است؛ برای آن از venv_* استفاده کن. toolchain_run فقط برای پروژهٔ flutter/gradle/node است.")
    if task not in TASKS[{"flutter": "flutter", "gradle": "gradle", "node": "node"}[kind]]:
        raise ToolchainError(f"task مجاز برای {kind}: {TASK_HELP[kind]}")
    slug = project.name
    spec = {"cwd": str(project), "ok_on_timeout": False, "tool": KIND_TOOL[kind]}
    if kind == "flutter":
        exe = toolfinder.find("flutter")
        if not exe:
            raise ToolchainError("Flutter پیدا نشد (نه در PATH و نه در پوشه‌های معمول). مسیر flutter.bat را در tool_paths.json بنویس.")
        argv, timeout = [exe], 600
        if task == "create":
            if (project / "pubspec.yaml").exists():
                raise ToolchainError("این پروژه از قبل pubspec.yaml دارد؛ create دوباره لازم نیست.")
            platforms = args.get("platforms") or ["windows"]
            if not isinstance(platforms, list) or not platforms or any(p not in FLUTTER_PLATFORMS for p in platforms):
                raise ToolchainError("platforms باید فهرستی از این‌ها باشد: " + ", ".join(sorted(FLUTTER_PLATFORMS)))
            argv += ["create", "--project-name", _flutter_name(slug), "--org", "com.localchat", "--platforms", ",".join(platforms), "."]
            text, timeout = f"ساخت اسکلت Flutter ({','.join(platforms)})", 600
        elif task == "pub_get":
            argv += ["pub", "get"]; text = "دریافت وابستگی‌ها (flutter pub get)"
        elif task == "pub_add":
            pkgs = _packages(args.get("packages"), PUB_PACKAGE, "pub")
            argv += ["pub", "add", *pkgs]; text = "افزودن بسته‌ها: " + ", ".join(pkgs)
        elif task == "analyze":
            argv += ["analyze"]; text = "تحلیل ایستای کد (flutter analyze)"
        elif task == "test":
            argv += ["test"]; text = "اجرای تست‌ها (flutter test)"; timeout = 900
        elif task == "build_windows":
            argv += ["build", "windows"]; text = "ساخت نسخهٔ ویندوز (flutter build windows)"; timeout = 1500
        elif task == "build_apk":
            argv += ["build", "apk", "--debug"]; text = "ساخت APK آزمایشی"; timeout = 1500
        elif task == "build_web":
            argv += ["build", "web"]; text = "ساخت نسخهٔ وب"; timeout = 900
        else:
            argv += ["doctor", "-v"]; text = "بررسی سلامت Flutter"; timeout = 300
        if task != "create" and task != "doctor" and not (project / "pubspec.yaml").exists():
            raise ToolchainError("pubspec.yaml وجود ندارد؛ ابتدا toolchain_run با task=create را اجرا کن.")
        return {**spec, "argv": argv, "timeout": timeout, "env": _tool_env(exe), "text": text}
    if kind == "gradle":
        wrapper = project / ("gradlew.bat" if WINDOWS else "gradlew")
        exe = str(wrapper) if wrapper.is_file() and task != "init" else toolfinder.find("gradle")
        if not exe:
            raise ToolchainError("Gradle پیدا نشد (نه در PATH و نه در پوشه‌های معمول). مسیر gradle.bat را در tool_paths.json بنویس.")
        flags = ["--console=plain", "--no-daemon"]
        if task == "init":
            if any((project / name).exists() for name in ("build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts")):
                raise ToolchainError("این پروژه از قبل فایل Gradle دارد؛ init دوباره لازم نیست.")
            kind_type = args.get("type") or "java-application"
            dsl = args.get("dsl") or "kotlin"
            if kind_type not in GRADLE_TYPES or dsl not in GRADLE_DSL:
                raise ToolchainError("type یکی از " + ", ".join(sorted(GRADLE_TYPES)) + " و dsl یکی از kotlin/groovy باشد.")
            argv = [exe, "init", "--type", kind_type, "--dsl", dsl, "--package", "app", "--project-name", re.sub(r"[^A-Za-z0-9_-]", "-", slug),
                    "--use-defaults", "--no-incubating", *flags]
            text, timeout = f"ساخت اسکلت Gradle ({kind_type}, {dsl})", 900
        else:
            if not any((project / name).exists() for name in ("build.gradle", "build.gradle.kts")):
                raise ToolchainError("build.gradle پیدا نشد؛ ابتدا toolchain_run با task=init را اجرا کن.")
            gradle_task = {"build": "build", "test": "test", "run": "run", "clean": "clean"}[task]
            argv = [exe, gradle_task, *flags]
            text = {"build": "ساخت (gradle build)", "test": "اجرای تست‌ها (gradle test)", "run": "اجرای برنامه (gradle run، حداکثر ۶۰ ثانیه)",
                    "clean": "پاک‌سازی خروجی ساخت"}[task]
            timeout = 60 if task == "run" else 1200
            spec["ok_on_timeout"] = task == "run"
        return {**spec, "argv": argv, "timeout": timeout, "env": _tool_env(exe), "text": text}
    # Node.js / npm. On Windows prefer invoking npm's JavaScript CLI through
    # node.exe directly. This avoids cmd.exe/PATHEXT quoting differences between
    # Windows versions and also works when npm.cmd is discovered outside PATH.
    exe = toolfinder.find("npm")
    if not exe:
        raise ToolchainError("npm پیدا نشد (نه در PATH و نه در پوشه‌های معمول). Node.js را نصب کن یا مسیر را در tool_paths.json بنویس.")
    npm_prefix = [exe]
    if WINDOWS and Path(exe).suffix.lower() in {".cmd", ".bat"}:
        node_exe = toolfinder.find("node")
        npm_cli_candidates = [
            Path(exe).parent / "node_modules" / "npm" / "bin" / "npm-cli.js",
            Path(exe).parent / "node_modules" / "npm" / "bin" / "npm.js",
        ]
        npm_cli = next((candidate for candidate in npm_cli_candidates if candidate.is_file()), None)
        if node_exe and npm_cli:
            npm_prefix = [node_exe, str(npm_cli)]
    base_install = ["install", "--ignore-scripts", "--no-audit", "--no-fund"]
    timeout = 600
    if task == "init":
        if (project / "package.json").exists():
            raise ToolchainError("package.json از قبل وجود دارد.")
        argv, text = [*npm_prefix, "init", "-y"], "ساخت package.json"
    elif task == "install":
        argv, text = [*npm_prefix, *base_install], "نصب وابستگی‌ها (npm install، بدون اسکریپت‌های نصب)"
    elif task == "add":
        pkgs = _packages(args.get("packages"), NPM_PACKAGE, "npm")
        argv, text = [*npm_prefix, *base_install, *pkgs], "افزودن بسته‌ها: " + ", ".join(pkgs)
    elif task == "test":
        argv, text, timeout = [*npm_prefix, "test"], "اجرای تست‌ها (npm test؛ اسکریپت test پروژه اجرا می‌شود)", 600
    elif task == "build":
        argv, text, timeout = [*npm_prefix, "run", "build"], "ساخت (npm run build)", 900
    else:
        argv, text, timeout = [*npm_prefix, "start"], "اجرای برنامه (npm start، حداکثر ۳۰ ثانیه)", 30
        spec["ok_on_timeout"] = True
    if task != "init" and not (project / "package.json").exists():
        raise ToolchainError("package.json وجود ندارد؛ ابتدا toolchain_run با task=init را اجرا کن.")
    return {**spec, "argv": argv, "timeout": timeout, "env": _tool_env(*npm_prefix), "text": text}


# ----------------------------------------------------------------------------- VS Code workspace files
MARK = "// LocalChat generated"


def _json(data):
    return MARK + "\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def editor_files(project, kind):
    """Fixed .vscode files for the kind (so F5 / Ctrl+Shift+B work out of the box)."""
    files = {}
    recommend = {"recommendations": EXTENSIONS.get(kind, [])}
    files[".vscode/extensions.json"] = _json(recommend)
    if kind == "python":
        interpreter = "${workspaceFolder}/.venv/Scripts/python.exe" if WINDOWS else "${workspaceFolder}/.venv/bin/python"
        files[".vscode/settings.json"] = _json({
            "python.defaultInterpreterPath": interpreter, "python.analysis.extraPaths": ["${workspaceFolder}/src"],
            "python.testing.unittestEnabled": True, "python.testing.pytestEnabled": False,
            "python.testing.unittestArgs": ["-v", "-s", "./tests", "-p", "test_*.py"],
            "files.exclude": {".history": True, ".localchat": True, "**/__pycache__": True},
            "terminal.integrated.env.windows": {"PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src"},
            "terminal.integrated.env.linux": {"PYTHONPATH": "${workspaceFolder}:${workspaceFolder}/src"},
            "terminal.integrated.env.osx": {"PYTHONPATH": "${workspaceFolder}:${workspaceFolder}/src"}})
        env = {"PYTHONPATH": "${workspaceFolder}" + os.pathsep + "${workspaceFolder}/src"}
        files[".vscode/launch.json"] = _json({"version": "0.2.0", "configurations": [
            {"name": "Run src/main.py", "type": "debugpy", "request": "launch", "program": "${workspaceFolder}/src/main.py",
             "cwd": "${workspaceFolder}", "env": env, "console": "integratedTerminal"},
            {"name": "Run current file", "type": "debugpy", "request": "launch", "program": "${file}", "cwd": "${workspaceFolder}",
             "env": env, "console": "integratedTerminal"}]})
        files[".vscode/tasks.json"] = _json({"version": "2.0.0", "tasks": [
            {"label": "Run tests", "type": "shell", "command": "${command:python.interpreterPath}",
             "args": ["-m", "unittest", "discover", "-s", "tests", "-v"], "group": "test", "problemMatcher": []}]})
    elif kind == "flutter":
        files[".vscode/launch.json"] = _json({"version": "0.2.0", "configurations": [
            {"name": "Flutter", "type": "dart", "request": "launch", "program": "lib/main.dart"}]})
        files[".vscode/settings.json"] = _json({"dart.lineLength": 100, "editor.formatOnSave": True,
                                                "files.exclude": {".history": True, ".localchat": True}})
    elif kind == "gradle":
        files[".vscode/settings.json"] = _json({"java.configuration.updateBuildConfiguration": "automatic",
                                                "files.exclude": {".history": True, ".localchat": True, ".gradle": True, "build": True}})
        wrapper = ".\\gradlew.bat" if WINDOWS else "./gradlew"
        command = wrapper if (project / ("gradlew.bat" if WINDOWS else "gradlew")).exists() else "gradle"
        files[".vscode/tasks.json"] = _json({"version": "2.0.0", "tasks": [
            {"label": "gradle build", "type": "shell", "command": command, "args": ["build", "--console=plain"], "group": "build", "problemMatcher": []},
            {"label": "gradle test", "type": "shell", "command": command, "args": ["test", "--console=plain"], "group": "test", "problemMatcher": []},
            {"label": "gradle run", "type": "shell", "command": command, "args": ["run", "--console=plain"], "problemMatcher": []}]})
    elif kind == "node":
        main = "index.js"
        try:
            main = json.loads((project / "package.json").read_text(encoding="utf-8")).get("main") or main
        except (OSError, ValueError):
            pass
        files[".vscode/launch.json"] = _json({"version": "0.2.0", "configurations": [
            {"name": "Run " + str(main)[:60], "type": "node", "request": "launch", "program": "${workspaceFolder}/" + str(main)[:100]}]})
        files[".vscode/tasks.json"] = _json({"version": "2.0.0", "tasks": [
            {"type": "npm", "script": "test", "group": "test", "label": "npm test", "problemMatcher": []},
            {"type": "npm", "script": "start", "label": "npm start", "problemMatcher": []}]})
        files[".vscode/settings.json"] = _json({"files.exclude": {".history": True, ".localchat": True}})
    return files
