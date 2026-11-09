"""Find developer tools that are installed but not on PATH (Docker, VS Code, Flutter, Gradle, Java, Node, Git ...).

Only *finding* and *version probing* happen here; nothing is installed or modified.
A user can pin a tool by writing ``tool_paths.json`` into the LocalChat data folder::

    {"docker": "D:\\\\Tools\\\\Docker\\\\resources\\\\bin\\\\docker.exe", "flutter": "E:\\\\dev\\\\flutter\\\\bin\\\\flutter.bat"}
"""
import glob
import json
import os
import re
import shutil
import threading
from pathlib import Path

if __package__:
    from . import settings as app_settings
    from .procutil import run_capture
else:
    import settings as app_settings
    from procutil import run_capture

WINDOWS = os.name == "nt"
EXTENSIONS = (".exe", ".cmd", ".bat", ".com") if WINDOWS else ("",)

# command names on PATH, per logical tool
NAMES = {
    "docker": ("docker",),
    "code": ("code", "code-insiders", "codium"),
    "flutter": ("flutter",),
    "dart": ("dart",),
    "gradle": ("gradle",),
    "java": ("java",),
    "node": ("node",),
    "npm": ("npm",),
    "git": ("git",),
    "adb": ("adb",),
    "dotnet": ("dotnet",),
    "py": ("py",),
}
VERSION_ARGS = {
    "docker": ["--version"], "code": ["--version"], "dart": ["--version"], "gradle": ["--version"],
    "java": ["-version"], "node": ["--version"], "npm": ["--version"], "git": ["--version"],
    "adb": ["version"], "dotnet": ["--version"], "py": ["--list"],
}
_CACHE = {"time": 0.0, "data": {}}
_LOCK = threading.RLock()


def _env(name):
    return os.environ.get(name) or ""


def _home():
    return Path(os.path.expanduser("~"))


def _patterns(tool):
    """Candidate file patterns (may contain *) in addition to PATH."""
    home = _home()
    pf = [p for p in (_env("ProgramFiles"), _env("ProgramFiles(x86)"), _env("ProgramW6432")) if p]
    local = _env("LOCALAPPDATA")
    drives = [f"{letter}:\\" for letter in "CDEFG"] if WINDOWS else ["/opt", "/usr/local", str(home)]
    out = []

    def add(*parts):
        out.append(os.path.join(*parts))

    if tool == "docker":
        for base in pf:
            add(base, "Docker", "Docker", "resources", "bin", "docker.exe")
        add(str(home), ".docker", "bin", "docker.exe")
        if not WINDOWS:
            out += ["/usr/local/bin/docker", "/usr/bin/docker", "/opt/homebrew/bin/docker"]
    elif tool == "code":
        if local:
            add(local, "Programs", "Microsoft VS Code", "bin", "code.cmd")
            add(local, "Programs", "VSCodium", "bin", "codium.cmd")
        for base in pf:
            add(base, "Microsoft VS Code", "bin", "code.cmd")
        for drive in drives:
            add(drive, "Microsoft VS Code", "bin", "code.cmd")
        if not WINDOWS:
            out += ["/usr/local/bin/code", "/usr/bin/code", "/snap/bin/code"]
    elif tool in {"flutter", "dart"}:
        roots = [_env("FLUTTER_ROOT"), _env("FLUTTER_HOME"), str(home / "flutter"), str(home / "development" / "flutter"),
                 str(home / "dev" / "flutter"), str(home / "fvm" / "default"), str(home / "sdk" / "flutter"),
                 os.path.join(local, "flutter") if local else "", os.path.join(local, "Programs", "flutter") if local else ""]
        for drive in drives:
            roots += [os.path.join(drive, "flutter"), os.path.join(drive, "src", "flutter"), os.path.join(drive, "dev", "flutter"),
                      os.path.join(drive, "tools", "flutter"), os.path.join(drive, "sdk", "flutter"), os.path.join(drive, "Flutter")]
        for root in roots:
            if not root:
                continue
            if tool == "flutter":
                add(root, "bin", "flutter.bat" if WINDOWS else "flutter")
            else:
                add(root, "bin", "cache", "dart-sdk", "bin", "dart.exe" if WINDOWS else "dart")
        if tool == "dart":
            for drive in drives:
                add(drive, "dart-sdk", "bin", "dart.exe")
    elif tool == "gradle":
        for root in (_env("GRADLE_HOME"), _env("GRADLE_USER_HOME")):
            if root:
                add(root, "bin", "gradle.bat" if WINDOWS else "gradle")
        for drive in drives:
            add(drive, "Gradle", "gradle-*", "bin", "gradle.bat" if WINDOWS else "gradle")
            add(drive, "gradle", "gradle-*", "bin", "gradle.bat" if WINDOWS else "gradle")
            add(drive, "tools", "gradle*", "bin", "gradle.bat" if WINDOWS else "gradle")
        for base in pf:
            add(base, "Gradle", "gradle-*", "bin", "gradle.bat")
        add(str(home), ".gradle", "wrapper", "dists", "gradle-*", "*", "gradle-*", "bin", "gradle.bat" if WINDOWS else "gradle")
        add(str(home), ".sdkman", "candidates", "gradle", "current", "bin", "gradle")
        add(str(home), "scoop", "apps", "gradle", "current", "bin", "gradle.bat")
    elif tool == "java":
        for root in (_env("JAVA_HOME"), _env("JDK_HOME")):
            if root:
                add(root, "bin", "java.exe" if WINDOWS else "java")
        for base in pf:
            add(base, "Java", "*", "bin", "java.exe")
            add(base, "Eclipse Adoptium", "*", "bin", "java.exe")
            add(base, "Microsoft", "jdk-*", "bin", "java.exe")
            add(base, "Android", "Android Studio", "jbr", "bin", "java.exe")
            add(base, "JetBrains", "*", "jbr", "bin", "java.exe")
        add(str(home), ".jdks", "*", "bin", "java.exe" if WINDOWS else "java")
    elif tool in {"node", "npm"}:
        exe = ("node.exe" if tool == "node" else "npm.cmd") if WINDOWS else tool
        for base in pf:
            add(base, "nodejs", exe)
        if local:
            add(local, "Programs", "nodejs", exe)
        add(str(home), ".nvm", "current", exe)
        add(str(home), "scoop", "apps", "nodejs", "current", exe)
    elif tool == "git":
        for base in pf:
            add(base, "Git", "cmd", "git.exe")
        if local:
            add(local, "Programs", "Git", "cmd", "git.exe")
    elif tool == "adb":
        for root in (_env("ANDROID_HOME"), _env("ANDROID_SDK_ROOT"), os.path.join(local, "Android", "Sdk") if local else "",
                     str(home / "Android" / "Sdk")):
            if root:
                add(root, "platform-tools", "adb.exe" if WINDOWS else "adb")
    elif tool == "dotnet":
        for base in pf:
            add(base, "dotnet", "dotnet.exe")
    return out


def _configured():
    path = app_settings.app_data_dir() / "tool_paths.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def find(tool):
    """Absolute path of ``tool`` (see NAMES) or ``None``. Order: user setting, PATH, well-known folders."""
    if tool not in NAMES:
        return None
    pinned = _configured().get(tool)
    if isinstance(pinned, str) and pinned.strip():
        candidate = Path(pinned.strip().strip('"'))
        if candidate.is_file():
            return str(candidate)
    for name in NAMES[tool]:
        found = shutil.which(name)
        if found:
            return found
    for pattern in _patterns(tool):
        if "*" in pattern:
            matches = sorted(glob.glob(pattern), reverse=True)  # newest version folder first
            for match in matches:
                if os.path.isfile(match):
                    return match
        elif os.path.isfile(pattern):
            return pattern
    return None


def env_with_dir(exe, env=None):
    """Environment whose PATH also contains the tool's folder (Docker needs its credential helpers next to it)."""
    merged = dict(os.environ if env is None else env)
    folder = str(Path(exe).parent)
    merged["PATH"] = folder + os.pathsep + merged.get("PATH", "")
    return merged


def _first_line(text):
    lines = [line.strip() for line in (text or "").splitlines()
             if line.strip() and not line.strip().startswith("Picked up") and re.search(r"[A-Za-z0-9]", line)]
    for line in lines:
        if re.search(r"\d+\.\d+", line):
            return line[:160]
    return lines[0][:160] if lines else ""


def _flutter_version(exe):
    """Read the version from files (running `flutter --version` can take half a minute)."""
    root = Path(exe).resolve().parent.parent
    for relative in ("bin/cache/flutter.version.json", "version"):
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if relative.endswith(".json"):
            try:
                return json.loads(text).get("frameworkVersion", "")
            except ValueError:
                continue
        return text.splitlines()[0] if text else ""
    return ""


def _probe(tool, exe):
    info = {"found": True, "path": exe, "on_path": bool(shutil.which(NAMES[tool][0])) and
            os.path.normcase(shutil.which(NAMES[tool][0]) or "") == os.path.normcase(exe)}
    if tool == "flutter":
        info["version"] = _flutter_version(exe) or "(نسخه از فایل خوانده نشد)"
        return info
    args = VERSION_ARGS.get(tool)
    if not args:
        return info
    try:
        result = run_capture([exe, *args], timeout=20, env=env_with_dir(exe))
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        info["version"] = _first_line(result.stdout if tool != "java" else text)
        if result.returncode != 0 and not info["version"]:
            info["version"] = "(اجرا شد ولی نسخه خوانده نشد)"
    except OSError as exc:
        info["version"] = f"(اجرا نشد: {str(exc)[:80]})"
    return info


def scan(force=False, tools=None, max_age=120):
    """Dict tool → {found, path, version, on_path}; probes run in parallel and are cached for ``max_age`` seconds."""
    import time as _time
    now = _time.monotonic()
    with _LOCK:
        if not force and _CACHE["data"] and now - _CACHE["time"] < max_age and tools is None:
            return dict(_CACHE["data"])
    names = list(tools or NAMES)
    result = {}

    def work(tool):
        exe = find(tool)
        result[tool] = _probe(tool, exe) if exe else {"found": False}

    threads = [threading.Thread(target=work, args=(tool,), daemon=True) for tool in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=45)
    for tool in names:
        result.setdefault(tool, {"found": False, "version": "(بررسی زمان‌بر شد)"})
    if tools is None:
        with _LOCK:
            _CACHE.update(time=now, data=dict(result))
    return result


def describe(found):
    """Short Persian summary for prompts / the environment report."""
    parts = []
    for tool, info in found.items():
        if info.get("found"):
            parts.append(f"{tool}: {info.get('version') or 'نصب است'}" + ("" if info.get("on_path") else " (خارج از PATH)"))
    missing = [tool for tool, info in found.items() if not info.get("found")]
    text = "؛ ".join(parts) or "هیچ ابزاری پیدا نشد"
    if missing:
        text += " | پیدا نشد: " + ", ".join(missing)
    return text
