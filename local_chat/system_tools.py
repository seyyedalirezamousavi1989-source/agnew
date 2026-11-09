"""Approval-gated, allowlisted local project tools for LocalChat's programming mode.

* No shell is ever used; every command is a fixed argument list built here.
* Files live only in managed projects under PROJECTS_ROOT, in fixed folders, with fixed
  text extensions; symlinks and ``..`` are rejected.
* Every mutating / executing operation is *prepared* first (preview for the user) and only
  runs after ``approve_action`` is called with the one-use token of the same browser session.
"""
import difflib
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
from pathlib import Path

if __package__:
    from . import docker_runtime
    from . import project_analysis
    from . import settings as app_settings
    from . import toolchains
    from . import toolfinder
    from .capabilities import register_capability
    from .procutil import run_capture
else:
    import docker_runtime
    import project_analysis
    import settings as app_settings
    import toolchains
    import toolfinder
    from capabilities import register_capability
    from procutil import run_capture

PROJECTS_ROOT = app_settings.app_data_dir() / "projects"
APPROVAL_TTL = 600
MAX_SCRIPT_BYTES = 256 * 1024
MAX_OUTPUT_CHARS = 12000
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_PACKAGE = docker_runtime.PACKAGE
_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,59}$")
_RESERVED = {"con", "prn", "aux", "nul", *{f"com{i}" for i in range(1, 10)}, *{f"lpt{i}" for i in range(1, 10)}}
ALLOWED_DIRS = ("src", "tests", "data", "assets", "docs")
ROOT_FILES = {"requirements.txt", "README.md", ".gitignore"}
LISTABLE_ROOT_FILES = ROOT_FILES | {"run.bat"}
TEXT_EXTENSIONS = {".py", ".txt", ".md", ".json", ".csv", ".html", ".css", ".js", ".ts", ".jsx", ".tsx",
                   ".toml", ".yaml", ".yml", ".ini", ".cfg", ".xml", ".sql", ".svg"}
LAUNCHER_MARK = "REM LocalChat launcher"
MAX_READ_CHARS = 40000
MAX_LIST_ENTRIES = 300
MAX_DIFF_LINES = 160
SCREENSHOT_KEYS = {"RETURN", "SPACE", "LEFT", "RIGHT", "UP", "DOWN", "TAB", "BACKSPACE", "a", "d", "w", "s", "r", "n", "p", "1", "2", "3"}

READ_ONLY_ACTIONS = {"read_file", "list_files", "check_environment", "check_project", "list_projects"}
SAFE_WRITE_ACTIONS = {"create_project", "create_venv", "write_file", "update_file", "edit_file", "create_launcher", "editor_setup"}
EXEC_ACTIONS = {"venv_install", "venv_run", "venv_test", "docker_test", "docker_run", "screenshot_check", "open_in_editor", "toolchain_run", "editor_extension"}
ACTION_LABELS = {
    "create_project": "ساخت پوشهٔ پروژه",
    "create_venv": "ساخت محیط مجازی (venv)",
    "write_file": "ساخت فایل جدید",
    "update_file": "بازنویسی فایل (با پشتیبان)",
    "edit_file": "ویرایش بخشی از فایل (با پشتیبان)",
    "create_launcher": "ساخت فایل اجرای run.bat",
    "venv_install": "نصب کتابخانه در venv پروژه",
    "venv_run": "اجرای برنامه با venv پروژه",
    "venv_test": "اجرای تست‌ها با venv پروژه",
    "docker_test": "اجرای تست در Docker ایزوله",
    "docker_run": "اجرای اسکریپت در Docker ایزوله",
    "read_file": "خواندن فایل",
    "list_files": "فهرست فایل‌ها",
    "check_environment": "بررسی محیط (پایتون، Docker، VS Code، Flutter، Gradle، venv)",
    "check_project": "بررسی ساختار و اتصال فایل‌های پروژه",
    "list_projects": "فهرست پروژه‌های ساخته‌شده",
    "screenshot_check": "اجرای کوتاه بدون پنجره و اسکرین‌شات از برنامه",
    "open_in_editor": "باز کردن پروژه (یا یک فایل/خط) در VS Code",
    "toolchain_run": "اجرای ابزار ساخت پروژه (Flutter / Gradle / Node)",
    "editor_setup": "تنظیم VS Code برای پروژه (F5، تست، مفسر)",
    "editor_extension": "نصب افزونهٔ VS Code",
}
_pending = {}
_lock = threading.RLock()
_PYTHON_CACHE = {"value": None, "expires": 0.0}


class SystemToolError(ValueError):
    pass


# ----------------------------------------------------------------------------- helpers
def _tail(text, limit=MAX_OUTPUT_CHARS):
    text = text or ""
    return text if len(text) <= limit else "…(ابتدای خروجی حذف شد)…\n" + text[-limit:]


def _kb(size):
    return f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"


def _project_path(name):
    if not isinstance(name, str) or not _SLUG.fullmatch(name) or name in _RESERVED:
        raise SystemToolError("نام پروژه باید فقط شامل حروف کوچک انگلیسی، عدد، خط تیره یا زیرخط باشد (مثل snake-game).")
    root = PROJECTS_ROOT.resolve()
    target = (root / name).resolve()
    if target.parent != root:
        raise SystemToolError("مسیر پروژه معتبر نیست.")
    return target


def _existing_project(name):
    project = _project_path(name)
    if project.is_symlink() or not project.is_dir():
        raise SystemToolError(f"پروژهٔ «{name}» پیدا نشد؛ ابتدا create_project را اجرا کنید یا نام را از list_files/فهرست پروژه‌ها بردارید.")
    return project


def _venv_python(project):
    venv = project / ".venv"
    candidate = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv.is_symlink() or not candidate.is_file():
        raise SystemToolError("محیط مجازی این پروژه وجود ندارد؛ ابتدا create_venv را اجرا کنید.")
    return candidate


def _clean_env(extra=None):
    env = {key: value for key, value in os.environ.items()
           if key.upper() not in {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__", "PYTHONSTARTUP"}}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    if extra:
        env.update(extra)
    return env


def system_python():
    """Command (argument list) of a real Python interpreter that can create venvs.

    In the packaged Windows app ``sys.executable`` is LocalChat itself, so the interpreter
    is searched on PATH instead and verified by actually running it.
    """
    now = time.monotonic()
    if _PYTHON_CACHE["value"] and _PYTHON_CACHE["expires"] > now:
        return list(_PYTHON_CACHE["value"])
    candidates = []
    if not getattr(sys, "frozen", False):
        candidates.append([sys.executable])
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            candidates.append([found])
    launcher = shutil.which("py")
    if launcher:
        candidates.append([launcher, "-3"])
    for command in candidates:
        try:
            probe = run_capture([*command, "-c", "import sys, venv; print(sys.version.split()[0])"], timeout=20,
                                env=_clean_env())
        except OSError:
            continue
        if probe.returncode == 0 and re.match(r"^\d+\.\d+", probe.stdout.strip()):
            _PYTHON_CACHE.update(value=list(command), expires=now + 300)
            return list(command)
    raise SystemToolError(
        "Python روی این رایانه پیدا نشد (یا نسخهٔ Microsoft Store بدون نصب واقعی است). "
        "Python 3.10+ را از python.org نصب کنید و گزینهٔ «Add python.exe to PATH» را بزنید؛ سپس دوباره تلاش کنید.")


def _kind(project):
    """Project kind: python (default) | flutter | gradle | node — from the manifest, else inferred from marker files."""
    try:
        kind = json.loads((Path(project) / MANIFEST_DIR / "manifest.json").read_text(encoding="utf-8")).get("kind")
    except (OSError, ValueError, AttributeError):
        kind = None
    if kind in toolchains.KINDS:
        return kind
    project = Path(project)
    if (project / "pubspec.yaml").is_file():
        return "flutter"
    if any((project / name).is_file() for name in ("build.gradle", "build.gradle.kts")):
        return "gradle"
    if (project / "package.json").is_file():
        return "node"
    return "python"


def _set_kind(project, kind):
    folder = Path(project) / MANIFEST_DIR
    folder.mkdir(exist_ok=True)
    manifest = _read_manifest(Path(project))
    manifest["kind"] = kind
    _write_atomic(folder / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))


def _root_files(project):
    return toolchains.policy(_kind(project))["root"] | {"run.bat"}


def _dirs(project):
    return toolchains.policy(_kind(project))["dirs"]


def _safe_file(project, relative, must_exist=None):
    """Validate a project-relative path: fixed folders / root files only, no symlinks, no '..'."""
    if not isinstance(relative, str) or not relative or len(relative) > 200 or "\\" in relative:
        raise SystemToolError("مسیر فایل معتبر نیست؛ از / استفاده کنید، مثل src/main.py.")
    parts = relative.split("/")
    if len(parts) == 1:
        if parts[0] not in _root_files(project):
            raise SystemToolError("در ریشهٔ پروژه فقط این فایل‌ها مجازند: " + ", ".join(sorted(toolchains.policy(_kind(project))["root"])) +
                                  "؛ بقیه را در " + ", ".join(_dirs(project)) + " بگذارید.")
    else:
        policy = toolchains.policy(_kind(project))
        if len(parts) > (5 if _kind(project) == "python" else 9) or any(not _SEGMENT.fullmatch(part) or part.endswith(".") for part in parts):
            raise SystemToolError("مسیر فایل فقط می‌تواند نام‌های ساده انگلیسی (بدون فاصله و حروف فارسی) و حداکثر ۵ سطح باشد.")
        if parts[0] not in policy["dirs"]:
            raise SystemToolError("پوشهٔ مقصد باید یکی از این‌ها باشد: " + ", ".join(policy["dirs"]) + ".")
        if any(Path(part).stem.lower() in _RESERVED for part in parts):
            raise SystemToolError("نام رزروشدهٔ ویندوز (مثل con یا nul) برای فایل مجاز نیست.")
        if Path(parts[-1]).suffix.lower() not in policy["ext"]:
            raise SystemToolError("پسوند فایل مجاز نیست؛ فقط فایل‌های متنی (" + ", ".join(sorted(e[1:] for e in policy["ext"])) + "). "
                                  "فایل باینری (عکس/صدا) را نمی‌توان ساخت؛ از رسم با کد استفاده کنید.")
    root = project.resolve()
    current = project
    for part in parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise SystemToolError("پیوند نمادین در مسیر فایل مجاز نیست.")
    target = project.joinpath(*parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SystemToolError("مسیر فایل خارج از پروژه است.") from exc
    if target.is_symlink():
        raise SystemToolError("پیوند نمادین مجاز نیست.")
    if must_exist is True and not target.is_file():
        raise SystemToolError(f"فایل «{relative}» وجود ندارد؛ با list_files فهرست واقعی فایل‌ها را ببین.")
    if must_exist is False and target.exists():
        raise SystemToolError("این فایل از قبل وجود دارد؛ برای تغییر آن از edit_file (تغییر کوچک) یا update_file (بازنویسی کامل) استفاده کنید.")
    return target


def _is_default_stub(project, relative):
    """True for the placeholder requirements.txt / README.md that create_project writes."""
    if relative not in {"requirements.txt", "README.md"}:
        return False
    path = project / relative
    if path.is_symlink() or not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.strip() == "# Project dependencies" or "Python project managed by LocalChat." in text


def _check_content(content):
    if not isinstance(content, str) or not content.strip():
        raise SystemToolError("محتوای فایل خالی است؛ فایل را کامل بنویس.")
    if "\x00" in content or len(content.encode("utf-8")) > MAX_SCRIPT_BYTES:
        raise SystemToolError("محتوای فایل بیش از ۲۵۶ کیلوبایت است یا نویسهٔ نامعتبر دارد؛ آن را به چند فایل کوچک‌تر بشکن.")


def _read_text(path):
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


def _diff(old, new, relative):
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), "قبلی/" + relative, "جدید/" + relative,
                                     lineterm="", n=2))
    shown = diff[:MAX_DIFF_LINES]
    if len(diff) > MAX_DIFF_LINES:
        shown.append(f"… ({len(diff) - MAX_DIFF_LINES} خط تغییر دیگر نمایش داده نشد)")
    return diff, "\n".join(shown)


def _write_atomic(target, text, newline="\n"):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline=newline) as output:
        output.write(text)
    temporary.replace(target)


def _backup(project, relative, old):
    history = project / ".history"
    history.mkdir(exist_ok=True)
    backup = history / (relative.replace("/", "__") + "." + time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}.bak")
    backup.write_text(old, encoding="utf-8", newline="\n")
    return backup.relative_to(project).as_posix()


def _package_name(spec):
    return re.sub(r"[-_.]+", "-", re.split(r"[\[<>=!~;\s]", spec, maxsplit=1)[0]).lower()


def _merge_requirements(path, specs):
    """Add/replace package lines by normalised name instead of piling up duplicates."""
    existing = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    by_name = {_package_name(spec): spec for spec in specs}
    merged, seen = [], set()
    for line in existing:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if stripped == "# Project dependencies":
                continue
            merged.append(line)
            continue
        name = _package_name(stripped)
        if name in by_name:
            if name not in seen:
                merged.append(by_name[name])
                seen.add(name)
        else:
            merged.append(line)
    for name, spec in by_name.items():
        if name not in seen:
            merged.append(spec)
    path.write_text("\n".join(merged).strip() + "\n", encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------- listing & inspection
def list_projects(limit=40):
    root = PROJECTS_ROOT
    if not root.is_dir():
        return []
    items = []
    for item in sorted(root.iterdir()):
        if not item.is_dir() or item.is_symlink() or not _SLUG.fullmatch(item.name):
            continue
        count = sum(1 for path in item.rglob("*")
                    if path.is_file() and ".venv" not in path.parts and ".history" not in path.parts
                    and ".localchat" not in path.parts and "__pycache__" not in path.parts)
        items.append({"name": item.name, "files": count, "venv": (item / ".venv").is_dir()})
        if len(items) >= limit:
            break
    return items


def recent_projects_overview(max_projects=4, max_files=25):
    """Short text for prompts: most recently used projects and their files."""
    root = PROJECTS_ROOT
    if not root.is_dir():
        return ""
    found = [item for item in root.iterdir() if item.is_dir() and not item.is_symlink() and _SLUG.fullmatch(item.name)]
    found.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    lines = []
    for item in found[:max_projects]:
        try:
            listing = read_only_action("list_files", {"project": item.name})
        except SystemToolError:
            continue
        files = [entry["path"] for entry in listing["files"]][:max_files]
        lines.append(f"- {item.name}: " + (", ".join(files) or "(خالی)") + (" [venv دارد]" if listing["has_venv"] else ""))
    if len(found) > max_projects:
        lines.append(f"(و {len(found) - max_projects} پروژهٔ دیگر)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- manifest & always-on context
MANIFEST_DIR = ".localchat"
MAX_LOG = 80


def _manifest_path(project):
    return project / MANIFEST_DIR / "manifest.json"


def _read_manifest(project):
    try:
        data = json.loads(_manifest_path(project).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"log": []}
    return data if isinstance(data, dict) and isinstance(data.get("log"), list) else {"log": []}


def _log_action(project, plan, result):
    """Append one line to the project's own activity log, so later turns know what was really done."""
    try:
        if not project.is_dir() or project.is_symlink():
            return
        manifest = _read_manifest(project)
        entry = {"time": time.strftime("%Y-%m-%d %H:%M"), "action": plan.get("action"),
                 "path": plan.get("path") if isinstance(plan.get("path"), str) and not os.path.isabs(plan.get("path", "")) else "",
                 "ok": bool(result.get("ok")), "summary": str(result.get("summary") or "")[:200]}
        if plan.get("action") == "venv_install":
            entry["path"] = ", ".join(plan.get("packages", []))[:120]
        manifest["log"] = (manifest["log"] + [entry])[-MAX_LOG:]
        folder = project / MANIFEST_DIR
        folder.mkdir(exist_ok=True)
        _write_atomic(_manifest_path(project), json.dumps(manifest, ensure_ascii=False, indent=1))
    except OSError:
        pass


def _project_tree(project, limit=60):
    entries = []
    for directory in ("",) + tuple(_dirs(project)) + ("output",):
        base = project / directory if directory else project
        if not base.is_dir() or base.is_symlink():
            continue
        iterator = [p for p in base.iterdir() if p.is_file()] if not directory else sorted(base.rglob("*"))
        for item in sorted(iterator):
            if item.is_symlink() or not item.is_file() or "__pycache__" in item.parts:
                continue
            relative = item.relative_to(project).as_posix()
            if not directory and relative not in _root_files(project):
                continue
            entries.append(f"{relative} ({_kb(item.stat().st_size)})")
            if len(entries) >= limit:
                return entries
    return entries


def project_context(active=None, max_projects=6, log_lines=14):
    """Always-on text for the model: where projects live and exactly what exists in them (real disk state)."""
    root = PROJECTS_ROOT
    lines = [f"پوشهٔ اصلی پروژه‌ها روی این رایانه: {root}"]
    found = []
    if root.is_dir():
        found = [item for item in root.iterdir() if item.is_dir() and not item.is_symlink() and _SLUG.fullmatch(item.name)]
        found.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    if not found:
        lines.append("هنوز هیچ پروژه‌ای ساخته نشده است.")
        return "\n".join(lines)
    if active and all(item.name != active for item in found):
        active = None
    active = active or found[0].name  # the most recently touched project is the working one
    ordered = sorted(found, key=lambda item: item.name != active)
    lines.append(f"پروژهٔ فعال (آخرین پروژهٔ کاری): {active}")
    for item in ordered[:max_projects]:
        tree = _project_tree(item, limit=60 if item.name == active else 14)
        venv = "دارد" if (item / ".venv").is_dir() else "ندارد"
        reqs = []
        try:
            reqs = [ln.strip() for ln in _read_text(item / "requirements.txt").splitlines()
                    if ln.strip() and not ln.strip().startswith("#")][:12]
        except OSError:
            pass
        lines.append(f"\n[{item.name}] مسیر: {item}\n  فایل‌های واقعی: " + (", ".join(tree) or "(خالی)") +
                     f"\n  محیط مجازی: {venv}" + (f" | کتابخانه‌ها: {', '.join(reqs)}" if reqs else ""))
        if item.name == active:
            log = _read_manifest(item)["log"][-log_lines:]
            if log:
                lines.append("  آخرین عملیات انجام‌شده (به ترتیب زمان):")
                for entry in log:
                    mark = "✅" if entry.get("ok") else "❌"
                    lines.append(f"   {mark} {entry.get('time', '')} {entry.get('action', '')} {entry.get('path', '')} — {entry.get('summary', '')}"[:230])
    if len(found) > max_projects:
        lines.append(f"\n(و {len(found) - max_projects} پروژهٔ قدیمی‌تر دیگر؛ با list_projects ببین)")
    return "\n".join(lines)


def active_project_name():
    root = PROJECTS_ROOT
    if not root.is_dir():
        return ""
    found = [item for item in root.iterdir() if item.is_dir() and not item.is_symlink() and _SLUG.fullmatch(item.name)]
    found.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return found[0].name if found else ""


def open_projects_folder():
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(PROJECTS_ROOT))  # noqa: S606 - fixed, app-owned folder
        return True
    return False


def environment_report(project_name=None):
    report = {"ok": True, "action": "check_environment", "projects_root": str(PROJECTS_ROOT), "os": os.name}
    try:
        command = system_python()
        version = run_capture([*command, "-c", "import sys; print(sys.version.split()[0])"], timeout=20, env=_clean_env())
        report["python"] = {"available": True, "version": version.stdout.strip(), "command": " ".join(command)}
    except SystemToolError as exc:
        report["python"] = {"available": False, "message": str(exc)}
    report["docker"] = docker_runtime.status()
    report["toolchains"] = toolfinder.scan()
    report["toolchains_text"] = toolfinder.describe(report["toolchains"])
    editor = report["toolchains"].get("code", {})
    if editor.get("found"):
        try:
            listed = run_capture([editor["path"], "--list-extensions"], timeout=25, env=toolfinder.env_with_dir(editor["path"]))
            installed = {line.strip().lower() for line in listed.stdout.splitlines() if line.strip()}
            report["vscode_extensions"] = sorted(installed)[:60]
            report["vscode_missing"] = {kind: [e for e in items if e.lower() not in installed] for kind, items in toolchains.EXTENSIONS.items()}
        except OSError:
            pass
    report["projects"] = list_projects()
    if project_name:
        project = _existing_project(project_name)
        info = {"name": project_name, "has_venv": (project / ".venv").is_dir()}
        if info["has_venv"]:
            try:
                python = _venv_python(project)
                packages = run_capture([str(python), "-m", "pip", "list", "--format=json"], cwd=project, timeout=60,
                                       env=_clean_env())
                if packages.returncode == 0:
                    listed = json.loads(packages.stdout)
                    info["packages"] = [f"{item['name']}=={item['version']}" for item in listed][:80]
            except (SystemToolError, OSError, ValueError, KeyError, TypeError):
                info["packages"] = []
        req = project / "requirements.txt"
        if req.is_file():
            info["requirements"] = [line.strip() for line in _read_text(req).splitlines()
                                    if line.strip() and not line.strip().startswith("#")][:60]
        report["project"] = info
    report["summary"] = "بررسی محیط انجام شد."
    return report


def read_only_action(action, args):
    """Non-mutating inspection of managed projects; runs without an approval prompt."""
    if action not in READ_ONLY_ACTIONS or not isinstance(args, dict):
        raise SystemToolError("عملیات خواندنی معتبر نیست.")
    if action == "check_environment":
        name = args.get("project")
        return environment_report(name if isinstance(name, str) and name else None)
    if action == "list_projects":
        items = list_projects()
        return {"ok": True, "action": action, "root": str(PROJECTS_ROOT), "projects": items,
                "context": project_context(None), "summary": f"{len(items)} پروژه در {PROJECTS_ROOT} فهرست شد."}
    project = _existing_project(args.get("project"))
    if action == "check_project":
        kind = _kind(project)
        report = project_analysis.analyze(project) if kind == "python" else project_analysis.analyze_generic(project, kind)
        problems = report["problems"]
        return {"ok": True, "action": action, "project": args.get("project"), **report,
                "summary": ("ساختار پروژه سالم است." if report["healthy"] else
                            f"{len(problems)} مشکل در ساختار پروژه پیدا شد: " + " | ".join(problems[:3]))[:400]}
    if action == "read_file":
        target = _safe_file(project, args.get("path"), must_exist=True)
        text = _read_text(target)
        return {"ok": True, "action": action, "path": args["path"], "chars": len(text),
                "truncated": len(text) > MAX_READ_CHARS, "content": text[:MAX_READ_CHARS],
                "summary": f"{args['path']} خوانده شد ({len(text)} نویسه)."}
    entries = []
    for directory in ("",) + tuple(_dirs(project)):
        base = project / directory if directory else project
        if not base.is_dir() or base.is_symlink():
            continue
        iterator = base.iterdir() if not directory else base.rglob("*")
        for item in sorted(iterator):
            if item.is_symlink() or not item.is_file() or "__pycache__" in item.parts:
                continue
            relative = item.relative_to(project).as_posix()
            if not directory and relative not in _root_files(project):
                continue
            entries.append({"path": relative, "bytes": item.stat().st_size})
            if len(entries) >= MAX_LIST_ENTRIES:
                break
    return {"ok": True, "action": action, "project": args.get("project"), "kind": _kind(project), "files": entries,
            "has_venv": (project / ".venv").is_dir(),
            "summary": f"{len(entries)} فایل در پروژه «{args.get('project')}» فهرست شد."}


# ----------------------------------------------------------------------------- planning (preview)
def _launcher_text(entry):
    windows_entry = entry.replace("/", "\\")
    return "\r\n".join([
        "@echo off", LAUNCHER_MARK + " (generated; safe to delete)", "chcp 65001 >nul", 'cd /d "%~dp0"',
        'if exist ".venv\\Scripts\\python.exe" (', f'  ".venv\\Scripts\\python.exe" "{windows_entry}" %*', ") else (",
        f'  python "{windows_entry}" %*', ")", "echo.", "pause", ""])


def _run_args(args):
    argv = args.get("args", [])
    if argv is None:
        argv = []
    if (not isinstance(argv, list) or len(argv) > 10
            or any(not isinstance(item, str) or len(item) > 200 or "\n" in item or "\x00" in item for item in argv)):
        raise SystemToolError("آرگومان‌های اجرا معتبر نیستند (حداکثر ۱۰ مورد کوتاه).")
    return argv


def _timeout(args, default, maximum):
    value = args.get("timeout", default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        value = default
    return int(min(max(value, 5), maximum))


def _plan(action, args):
    if action not in SAFE_WRITE_ACTIONS | EXEC_ACTIONS:
        raise SystemToolError("این نوع عملیات در فهرست مجاز نیست.")
    if not isinstance(args, dict):
        raise SystemToolError("پارامترهای عملیات معتبر نیستند.")
    plan = {"action": action, "label": ACTION_LABELS[action],
            "kind": "write" if action in SAFE_WRITE_ACTIONS else "execute"}
    if action == "create_project":
        name = args.get("name")
        kind = args.get("kind") or "python"
        if kind not in toolchains.KINDS:
            raise SystemToolError("kind باید یکی از python, flutter, gradle, node باشد.")
        project = _project_path(name)
        if project.exists():
            raise SystemToolError(f"پروژهٔ «{name}» از قبل وجود دارد؛ برای ادامهٔ کار روی آن از list_files و بقیهٔ ابزارها استفاده کن یا نام دیگری بده.")
        if kind != "python":
            return {**plan, "project": name, "path": str(project), "project_kind": kind,
                    "command": f"ساخت پوشهٔ خالی پروژهٔ {toolchains.KINDS[kind]['label']}؛ اسکلت را بعداً toolchain_run (create/init) می‌سازد"}
        return {**plan, "project": name, "path": str(project), "project_kind": "python",
                "command": "ساخت پوشه‌های src، tests، data، output، logs و فایل‌های اولیهٔ requirements.txt و README.md"}
    if action == "editor_extension":
        extension = args.get("extension")
        if extension not in toolchains.EXTENSION_ALLOWLIST:
            raise SystemToolError("افزونهٔ مجاز: " + ", ".join(sorted(toolchains.EXTENSION_ALLOWLIST)))
        editor = toolfinder.find("code")
        if not editor:
            raise SystemToolError("VS Code (code) پیدا نشد.")
        return {**plan, "extension": extension, "command": f'"{editor}" --install-extension {extension}'}
    name = args.get("project")
    project = _existing_project(name)
    plan["project"] = name
    plan["path"] = str(project)
    if action in {"create_venv", "venv_install", "venv_run", "venv_test", "docker_test", "docker_run", "create_launcher", "screenshot_check"} \
            and _kind(project) != "python":
        raise SystemToolError(f"این پروژه از نوع {toolchains.KINDS[_kind(project)]['label']} است؛ این ابزار فقط برای Python است. "
                              "از toolchain_run استفاده کن.")
    if action == "create_venv":
        if (project / ".venv").exists():
            raise SystemToolError("محیط مجازی از قبل وجود دارد؛ چیزی بازنویسی نمی‌شود.")
        command = system_python()
        return {**plan, "path": str(project / ".venv"), "command": f'{" ".join(command)} -m venv "{project / ".venv"}"'}
    if action in {"write_file", "update_file", "edit_file"}:
        relative = args.get("path")
        if not isinstance(relative, str) or relative == "run.bat":
            raise SystemToolError("مسیر فایل معتبر نیست؛ run.bat را فقط ابزار create_launcher می‌سازد.")
    if action == "write_file":
        relative, content = args.get("path"), args.get("content")
        _check_content(content)
        stub = _is_default_stub(project, relative)
        target = _safe_file(project, relative, must_exist=None if stub else False)
        return {**plan, "path": relative, "content": content, "target": str(target), "overwrite_stub": stub,
                "command": (f"جایگزینی فایل اولیهٔ {relative} که create_project ساخته بود" if stub
                            else f"ایجاد فایل جدید {relative} (فایل موجود بازنویسی نمی‌شود)")}
    if action in {"update_file", "edit_file"}:
        relative = args.get("path")
        target = _safe_file(project, relative, must_exist=True)
        old = _read_text(target)
        if action == "update_file":
            content = args.get("content")
            _check_content(content)
        else:
            old_text, new_text = args.get("old_text"), args.get("new_text")
            if not isinstance(old_text, str) or not old_text or not isinstance(new_text, str):
                raise SystemToolError("old_text (بخش دقیقِ فعلی فایل) و new_text (جایگزین) لازم است.")
            old_text = old_text.replace("\r\n", "\n")
            found = old.count(old_text)
            if found == 0:
                raise SystemToolError("old_text در فایل پیدا نشد؛ ابتدا read_file را بزن و متن را دقیقاً (با فاصله‌ها) کپی کن.")
            if found > 1:
                raise SystemToolError(f"old_text {found} بار در فایل آمده است؛ چند خط اطراف را هم بیاور تا یکتا شود.")
            content = old.replace(old_text, new_text.replace("\r\n", "\n"), 1)
            _check_content(content)
        diff, shown = _diff(old, content, relative)
        if not diff:
            raise SystemToolError("محتوای جدید با فایل فعلی یکسان است؛ چیزی برای تغییر نیست.")
        return {**plan, "path": relative, "content": content, "target": str(target),
                "old_sha256": hashlib.sha256(old.encode("utf-8")).hexdigest(), "diff": shown,
                "command": f"ویرایش {relative}؛ نسخهٔ قبلی در پوشهٔ .history پشتیبان‌گیری می‌شود"}
    if action == "create_launcher":
        entry = args.get("entry", "src/main.py")
        entry_path = _safe_file(project, entry, must_exist=True)
        if not entry.startswith("src/") or entry_path.suffix.lower() != ".py":
            raise SystemToolError("فایل شروع باید یک .py داخل src باشد، مثل src/main.py.")
        target = project / "run.bat"
        if target.exists() and LAUNCHER_MARK not in target.read_text(encoding="utf-8", errors="replace"):
            raise SystemToolError("run.bat از قبل وجود دارد و ساختهٔ این برنامه نیست؛ بازنویسی نمی‌شود.")
        return {**plan, "path": "run.bat", "entry": entry, "content": _launcher_text(entry),
                "command": f"ساخت run.bat؛ با دو بار کلیک {entry} را با venv پروژه (یا python سیستم) اجرا می‌کند"}
    if action == "venv_install":
        _venv_python(project)
        packages = args.get("packages")
        if not isinstance(packages, list) or not 1 <= len(packages) <= 20:
            raise SystemToolError("فهرست نصب باید بین ۱ تا ۲۰ بسته داشته باشد.")
        if any(not isinstance(p, str) or not _PACKAGE.fullmatch(p.strip()) for p in packages):
            raise SystemToolError("نام یا نسخهٔ یکی از بسته‌ها معتبر نیست (فقط نام بستهٔ PyPI و محدودیت نسخه مثل pygame یا requests>=2.31).")
        packages = [p.strip() for p in packages]
        return {**plan, "packages": packages,
                "command": "pip install --only-binary=:all: " + " ".join(packages) + "  (فقط از PyPI، فقط wheel آماده؛ سپس requirements.txt به‌روز می‌شود)"}
    if action == "venv_run":
        relative = args.get("path")
        _venv_python(project)
        target = _safe_file(project, relative, must_exist=True)
        if not relative.startswith("src/") or target.suffix.lower() != ".py":
            raise SystemToolError("فقط فایل .py داخل src اجرا می‌شود.")
        argv, timeout = _run_args(args), _timeout(args, 30, 120)
        return {**plan, "path": relative, "args": argv, "timeout": timeout,
                "command": f"python {relative} {' '.join(argv)}".strip() + f"  (با venv پروژه، حداکثر {timeout} ثانیه؛ برنامه روی همین رایانه اجرا می‌شود و به شبکه و فایل‌های شما دسترسی دارد)"}
    if action == "venv_test":
        _venv_python(project)
        if not (project / "tests").is_dir():
            raise SystemToolError("پوشهٔ tests وجود ندارد؛ ابتدا فایل‌های tests/test_*.py را بساز.")
        return {**plan, "timeout": 180,
                "command": "اجرای تست‌های پروژه (pytest اگر نصب باشد، وگرنه unittest) با venv پروژه، حداکثر ۱۸۰ ثانیه؛ SDL بدون پنجره (dummy)"}
    if action in {"docker_test", "docker_run"}:
        relative = args.get("path")
        if action == "docker_run":
            target = _safe_file(project, relative, must_exist=True)
            if not relative.startswith("src/") or target.suffix.lower() != ".py":
                raise SystemToolError("فقط فایل .py داخل src اجرا می‌شود.")
        try:
            docker_runtime._docker()
            requirements_text, specs = docker_runtime._requirements(project)
        except docker_runtime.DockerRuntimeError as exc:
            raise SystemToolError(str(exc)) from exc
        plan.update(image=docker_runtime.IMAGE, dependency_count=len(specs),
                    requirements_sha256=hashlib.sha256(requirements_text.encode("utf-8")).hexdigest())
        if action == "docker_run":
            return {**plan, "path": relative,
                    "command": f"اجرای {relative} در کانتینر موقت Docker بدون شبکه، پروژه فقط‌خواندنی، حداکثر ۶۰ ثانیه (برنامه‌های گرافیکی/تعاملی اجرا نمی‌شوند)"}
        return {**plan, "command": "دریافت wheelهای باینری از PyPI در کانتینر جدا؛ نصب آفلاین و اجرای تست‌ها در کانتینر بدون شبکه؛ حذف خودکار کانتینرها"}
    if action == "screenshot_check":
        _venv_python(project)
        entry = args.get("entry") or "src/main.py"
        target = _safe_file(project, entry, must_exist=True)
        if not entry.startswith("src/") or target.suffix.lower() != ".py":
            raise SystemToolError("فایل شروع باید یک .py داخل src باشد، مثل src/main.py.")
        clicks = args.get("clicks") or []
        if (not isinstance(clicks, list) or len(clicks) > 12 or any(
                not isinstance(c, (list, tuple)) or len(c) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 5000 for v in c)
                for c in clicks)):
            raise SystemToolError("clicks باید فهرست حداکثر ۱۲ جفت [x, y] با مختصات ۰ تا ۵۰۰۰ باشد.")
        keys = args.get("keys") or []
        if not isinstance(keys, list) or len(keys) > 8 or any(not isinstance(k, str) or k not in SCREENSHOT_KEYS for k in keys):
            raise SystemToolError("keys فقط می‌تواند از این‌ها باشد: " + ", ".join(sorted(SCREENSHOT_KEYS)))
        warmup = args.get("warmup", 10)
        warmup = int(min(max(warmup if isinstance(warmup, (int, float)) and not isinstance(warmup, bool) else 10, 3), 120))
        spec = {"clicks": [[int(c[0]), int(c[1])] for c in clicks], "keys": list(keys), "warmup": warmup, "settle": 4}
        return {**plan, "entry": entry, "spec": spec, "timeout": 60,
                "command": f"python {entry} را بدون پنجره (SDL dummy) اجرا می‌کنم، {len(clicks)} کلیک و {len(keys)} کلید شبیه‌سازی می‌شود و از صفحه عکس می‌گیرم؛ "
                           "حداکثر ۶۰ ثانیه. برنامه روی همین رایانه اجرا می‌شود (فقط pygame)."}
    if action == "open_in_editor":
        editor = toolfinder.find("code")
        if not editor:
            raise SystemToolError("VS Code (code) پیدا نشد؛ نه در PATH و نه در پوشه‌های معمول. مسیر code.cmd را در tool_paths.json بنویس یا خودت پوشه را باز کن.")
        relative = args.get("path")
        line = args.get("line")
        goto = ""
        if relative:
            _safe_file(project, relative, must_exist=True)
            if line is not None and (isinstance(line, bool) or not isinstance(line, int) or not 1 <= line <= 100000):
                raise SystemToolError("line باید عدد صحیح ۱ تا ۱۰۰۰۰۰ باشد.")
            goto = f"{relative}:{line or 1}"
        return {**plan, "editor": editor, "goto": goto,
                "command": f'"{editor}" "{project}"' + (f' --goto "{goto}"' if goto else "") + "  (باز شدن در VS Code)"}
    if action == "editor_setup":
        kind = _kind(project)
        files = toolchains.editor_files(project, kind)
        fresh = [rel for rel in files if not (project / rel).exists()]
        return {**plan, "kind": "write", "files": sorted(files), "new_files": sorted(fresh),
                "command": "نوشتن تنظیمات ثابت VS Code برای پروژهٔ " + toolchains.KINDS[kind]["label"] + ": " + ", ".join(sorted(files)) +
                           "؛ فایل‌های موجودِ ساختهٔ دست خودتان بازنویسی نمی‌شوند."}
    if action == "toolchain_run":
        kind = _kind(project)
        try:
            spec = toolchains.build(project, kind, args.get("task"), args)
        except toolchains.ToolchainError as exc:
            raise SystemToolError(str(exc)) from exc
        return {**plan, "task": args.get("task"), "toolchain": spec["tool"], "timeout": spec["timeout"], "args": {
                k: v for k, v in args.items() if k in {"platforms", "packages", "type", "dsl"}},
                "command": " ".join(f'"{part}"' if " " in part else part for part in spec["argv"]) + f"  ({spec['text']}؛ حداکثر {spec['timeout']} ثانیه)"}
    raise SystemToolError("این نوع عملیات در فهرست مجاز نیست.")


def prepare_action(action, args, session_id):
    if not isinstance(session_id, str) or len(session_id) < 32:
        raise SystemToolError("نشست معتبر برای آماده‌سازی عملیات لازم است.")
    plan = _plan(action, args)
    plan["purpose"] = str(args.get("purpose") or "")[:300]
    plan["step"] = str(args.get("step") or "")[:40]
    token = secrets.token_urlsafe(24)
    now = time.monotonic()
    with _lock:
        for key, item in list(_pending.items()):
            if item["expires"] < now:
                _pending.pop(key, None)
        _pending[token] = {"plan": plan, "expires": now + APPROVAL_TTL, "session_id": session_id}
    preview = {key: value for key, value in plan.items() if key != "content"}
    if "content" in plan and "diff" not in plan:
        preview["content_preview"] = plan["content"]
    if "content" in plan:
        preview["content_bytes"] = len(plan["content"].encode("utf-8"))
    return {"approval_id": token, "expires_in": APPROVAL_TTL, "plan": preview,
            "message": "این عملیات هنوز اجرا نشده است؛ فقط پس از تأیید صریح کاربر اجرا می‌شود."}


def reject_action(token, session_id):
    with _lock:
        item = _pending.get(token)
        if not item or not secrets.compare_digest(item["session_id"], str(session_id or "")):
            return False
        _pending.pop(token, None)
        return item["expires"] >= time.monotonic()


# ----------------------------------------------------------------------------- execution (after approval)
def _result(ok, plan, summary, **extra):
    return {"ok": ok, "action": plan["action"], "project": plan.get("project"), "summary": summary, **extra}


def _run_result(plan, completed, success_text, failure_text, timeout):
    output = _tail((completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else ""))
    if completed.cancelled:
        return _result(False, plan, "عملیات توسط کاربر متوقف شد.", cancelled=True, output=output)
    if completed.timed_out:
        return _result(False, plan, f"{failure_text}: زمان {timeout} ثانیه تمام شد و فرایند متوقف شد.",
                       returncode=completed.returncode, timed_out=True, output=output)
    ok = completed.returncode == 0
    return _result(ok, plan, success_text if ok else f"{failure_text} (کد خروجی {completed.returncode}).",
                   returncode=completed.returncode, output=output)


def _approve_inner(token, session_id, cancelled=None):
    with _lock:
        item = _pending.get(token)
        if not item or not secrets.compare_digest(item["session_id"], str(session_id or "")):
            raise SystemToolError("این درخواست متعلق به نشست فعلی نیست یا وجود ندارد.")
        _pending.pop(token, None)
    if item["expires"] < time.monotonic():
        raise SystemToolError("درخواست تأیید منقضی شده است؛ دوباره درخواست بدهید.")
    plan = item["plan"]
    action = plan["action"]
    if action == "editor_extension":
        editor = toolfinder.find("code")
        if not editor or plan.get("extension") not in toolchains.EXTENSION_ALLOWLIST:
            raise SystemToolError("افزونه یا VS Code معتبر نیست.")
        completed = run_capture([editor, "--install-extension", plan["extension"]], timeout=300, env=toolfinder.env_with_dir(editor),
                                cancelled=cancelled)
        return _run_result(plan, completed, f"افزونهٔ {plan['extension']} نصب شد (یا از قبل نصب بود).", "نصب افزونه ناموفق بود", 300)
    project = _project_path(plan["project"])
    if action != "create_project" and (project.is_symlink() or not project.is_dir()):
        raise SystemToolError("مسیر پروژه پس از تأیید تغییر کرده است.")

    if action == "create_project" and plan.get("project_kind", "python") != "python":
        project.mkdir(parents=True, exist_ok=False)
        _set_kind(project, plan["project_kind"])
        return _result(True, plan, f"پروژهٔ {toolchains.KINDS[plan['project_kind']]['label']} «{plan['project']}» ساخته شد (پوشهٔ خالی).",
                       path=str(project), kind=plan["project_kind"])
    if action == "create_project":
        project.mkdir(parents=True, exist_ok=False)
        _set_kind(project, "python")
        for folder in ("src", "tests", "data", "output", "logs"):
            (project / folder).mkdir()
        (project / "requirements.txt").write_text("# Project dependencies\n", encoding="utf-8", newline="\n")
        (project / "README.md").write_text(
            f"# {plan['project']}\n\nPython project managed by LocalChat.\n\n"
            "Source: src/ | Tests: tests/ | Data: data/ | Output: output/ | Logs: logs/\n",
            encoding="utf-8", newline="\n")
        return _result(True, plan, f"پروژهٔ «{plan['project']}» ساخته شد.", path=str(project))

    if action == "create_venv":
        venv_path = project / ".venv"
        if venv_path.exists() or venv_path.is_symlink():
            raise SystemToolError("مسیر محیط مجازی پس از پیش‌نمایش ایجاد شده است؛ برای جلوگیری از دست‌کاری متوقف شدم.")
        completed = run_capture([*system_python(), "-m", "venv", str(venv_path)], cwd=project, timeout=300,
                                env=_clean_env(), cancelled=cancelled)
        if completed.cancelled or completed.timed_out or completed.returncode:
            return _run_result(plan, completed, "", "ساخت محیط مجازی ناموفق بود", 300)
        return _result(True, plan, "محیط مجازی ساخته شد.", python=str(_venv_python(project)))

    if action in {"write_file", "update_file", "edit_file", "create_launcher"}:
        relative = plan["path"]
        if action == "write_file":
            stub = bool(plan.get("overwrite_stub")) and _is_default_stub(project, relative)
            target = _safe_file(project, relative, must_exist=None if stub else False)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not stub and target.exists():
                raise SystemToolError("فایل پس از نمایش تأیید ایجاد شده است؛ هیچ فایلی بازنویسی نشد.")
            _write_atomic(target, plan["content"])
            size = len(plan["content"].encode("utf-8"))
            return _result(True, plan, f"فایل {relative} ساخته شد ({_kb(size)}).", path=relative, bytes=size)
        if action == "create_launcher":
            target = project / "run.bat"
            if target.exists() and LAUNCHER_MARK not in target.read_text(encoding="utf-8", errors="replace"):
                raise SystemToolError("run.bat پس از پیش‌نمایش تغییر کرده است؛ بازنویسی نشد.")
            _write_atomic(target, plan["content"], newline="")
            return _result(True, plan, f"run.bat ساخته شد؛ با دو بار کلیک {plan['entry']} اجرا می‌شود.", path="run.bat")
        target = _safe_file(project, relative, must_exist=True)
        old = _read_text(target)
        if hashlib.sha256(old.encode("utf-8")).hexdigest() != plan["old_sha256"]:
            raise SystemToolError("فایل پس از نمایش تأیید تغییر کرده است؛ دوباره read_file بزن و تغییر را از نو آماده کن.")
        backup = _backup(project, relative, old)
        _write_atomic(target, plan["content"])
        return _result(True, plan, f"فایل {relative} به‌روز شد (پشتیبان: {backup}).", path=relative, backup=backup)

    if action == "venv_install":
        python = _venv_python(project)
        command = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
                   "--only-binary=:all:", *plan["packages"]]
        completed = run_capture(command, cwd=project, timeout=900, env=_clean_env(), cancelled=cancelled)
        if completed.cancelled or completed.timed_out or completed.returncode:
            result = _run_result(plan, completed, "", "نصب کتابخانه ناموفق بود", 900)
            if not completed.cancelled and not completed.timed_out:
                result["hint"] = ("اگر پیام «No matching distribution» است، برای این بسته wheel آماده‌ای متناسب با نسخهٔ Python شما نیست؛ "
                                  "نسخهٔ دیگر یا کتابخانهٔ جایگزین انتخاب کن. نصب از سورس عمداً مجاز نیست.")
            return result
        versions = {}
        listing = run_capture([str(python), "-m", "pip", "list", "--format=json"], cwd=project, timeout=60, env=_clean_env())
        if listing.returncode == 0:
            try:
                versions = {_package_name(entry["name"]): entry["version"] for entry in json.loads(listing.stdout)}
            except (ValueError, KeyError, TypeError):
                versions = {}
        specs = []
        for package in plan["packages"]:
            plain = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", package)
            pinned = versions.get(_package_name(package)) if plain else None
            specs.append(f"{package}=={pinned}" if pinned else package)
        _merge_requirements(project / "requirements.txt", specs)
        return _result(True, plan, "کتابخانه‌ها نصب شدند و requirements.txt به‌روز شد: " + ", ".join(specs) + ".",
                       packages=specs, output=_tail(completed.stdout, 3000))

    if action == "venv_run":
        python = _venv_python(project)
        target = _safe_file(project, plan["path"], must_exist=True)
        env = _clean_env({"PYTHONPATH": os.pathsep.join([str(project), str(project / "src")])})
        completed = run_capture([str(python), "-X", "utf8", "-u", str(target.relative_to(project)), *plan["args"]],
                                cwd=project, timeout=plan["timeout"], env=env, cancelled=cancelled)
        result = _run_result(plan, completed, f"{plan['path']} بدون خطا اجرا و تمام شد (فقط یعنی کرش نکرد، نه اینکه درست کار می‌کند).",
                             f"اجرای {plan['path']} ناموفق بود", plan["timeout"])
        try:
            source = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            source = ""
        if result.get("ok") and re.search(r"^\s*(?:import|from)\s+(?:pygame|tkinter)", source, re.M):
            result["note"] = ("این برنامه گرافیکی است؛ اجرای بدون خطا چیزی دربارهٔ ظاهر یا کار کردن آن نمی‌گوید. "
                              "برای دیدن واقعی صفحه از screenshot_check استفاده کن.")
        return result

    if action == "venv_test":
        python = _venv_python(project)
        env = _clean_env({"PYTHONPATH": os.pathsep.join([str(project), str(project / "src")]),
                          "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYTHONDONTWRITEBYTECODE": "1"})
        has_pytest = run_capture([str(python), "-c", "import pytest"], cwd=project, timeout=30, env=env).returncode == 0
        command = ([str(python), "-m", "pytest", "-q", "-p", "no:cacheprovider"] if has_pytest
                   else [str(python), "-m", "unittest", "discover", "-s", "tests", "-v"])
        completed = run_capture(command, cwd=project, timeout=plan["timeout"], env=env, cancelled=cancelled)
        result = _run_result(plan, completed, "همهٔ تست‌ها موفق بودند.", "تست‌ها شکست خوردند", plan["timeout"])
        low = ((completed.stdout or "") + (completed.stderr or "")).lower()
        if result["ok"] and ("ran 0 tests" in low or "no tests ran" in low or "collected 0 items" in low):
            result["ok"] = False
            result["summary"] = "هیچ تستی پیدا نشد؛ فایل‌های tests/test_*.py با کلاس unittest.TestCase بنویس."
        result["runner"] = "pytest" if has_pytest else "unittest"
        return result

    if action in {"docker_test", "docker_run"}:
        script = plan.get("path") if action == "docker_run" else None
        try:
            outcome = docker_runtime.run_project_tests(str(project), plan["requirements_sha256"], script=script,
                                                       cancelled=cancelled)
        except docker_runtime.DockerRuntimeError as exc:
            raise SystemToolError(str(exc)) from exc
        what = f"اجرای {script}" if script else "تست‌های پروژه"
        summary = (f"{what} در Docker موفق بود." if outcome["ok"]
                   else (f"{what} در Docker زمان‌بر شد و متوقف شد." if outcome.get("timed_out") else f"{what} در Docker ناموفق بود."))
        return {"action": action, "project": plan["project"], "summary": summary, **outcome}
    if action == "screenshot_check":
        python = _venv_python(project)
        target = _safe_file(project, plan["entry"], must_exist=True)
        out_dir = project / "output" / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.png"):
            try:
                old.unlink()
            except OSError:
                pass
        harness = project / MANIFEST_DIR / "screenshot_harness.py"
        harness.parent.mkdir(exist_ok=True)
        _write_atomic(harness, project_analysis.HARNESS)
        env = _clean_env({"PYTHONPATH": os.pathsep.join([str(project), str(project / "src")]),
                          "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYTHONDONTWRITEBYTECODE": "1"})
        completed = run_capture([str(python), "-X", "utf8", "-u", str(harness), str(target.relative_to(project)), str(out_dir),
                                 json.dumps(plan["spec"])], cwd=project, timeout=plan["timeout"], env=env, cancelled=cancelled)
        return _screenshot_result(plan, completed, project, out_dir)

    if action == "open_in_editor":
        import subprocess
        editor = toolfinder.find("code")
        if not editor:
            raise SystemToolError("VS Code پیدا نشد.")
        goto_abs = str(project / plan["goto"].rsplit(":", 1)[0]) + ":" + plan["goto"].rsplit(":", 1)[1] if plan.get("goto") else ""
        command = [editor, str(project)] + (["--goto", goto_abs] if goto_abs else [])
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,  # noqa: S603 - fixed args
                             stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True)
        except OSError as exc:
            return _result(False, plan, f"باز کردن VS Code ناموفق بود: {str(exc)[:200]}")
        where = f" روی {plan['goto']}" if plan.get("goto") else ""
        return _result(True, plan, f"پروژهٔ «{plan['project']}» در VS Code باز شد{where}.", path=str(project))

    if action == "editor_setup":
        kind = _kind(project)
        written, skipped = [], []
        for rel, text in toolchains.editor_files(project, kind).items():
            target = project / rel
            if target.is_symlink() or (target.parent.exists() and target.parent.is_symlink()):
                raise SystemToolError("پیوند نمادین در .vscode مجاز نیست.")
            if target.exists() and toolchains.MARK not in target.read_text(encoding="utf-8", errors="replace"):
                skipped.append(rel)  # a file the user wrote: never overwritten
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(target, text)
            written.append(rel)
        note = f" (بدون دست‌زدن به فایل‌های شما: {', '.join(skipped)})" if skipped else ""
        return _result(True, plan, f"تنظیمات VS Code نوشته شد: {', '.join(written) or 'چیزی تغییر نکرد'}{note}.", files=written, skipped=skipped)

    if action == "toolchain_run":
        kind = _kind(project)
        try:
            spec = toolchains.build(project, kind, plan["task"], plan.get("args") or {})
        except toolchains.ToolchainError as exc:
            raise SystemToolError(str(exc)) from exc
        env = _clean_env(spec["env"])
        completed = run_capture(spec["argv"], cwd=project, timeout=spec["timeout"], env=env, cancelled=cancelled)
        if spec["ok_on_timeout"] and completed.timed_out and not completed.cancelled:
            output = _tail((completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else ""))
            return _result(True, plan, f"{spec['text']}: برنامه تا {spec['timeout']} ثانیه بدون خطا اجرا بود و بعد متوقف شد.", output=output)
        result = _run_result(plan, completed, spec["text"] + " — موفق.", spec["text"] + " — ناموفق", spec["timeout"])
        if result.get("ok") and plan["task"] in {"create", "init"}:
            _set_kind(project, kind)
        return result

    raise SystemToolError("عملیات مجاز نیست.")


def _screenshot_result(plan, completed, project, out_dir):
    output = _tail((completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else ""), 3000)
    if completed.cancelled:
        return _result(False, plan, "عملیات توسط کاربر متوقف شد.", cancelled=True)
    data = None
    for line in reversed((completed.stdout or "").splitlines()):
        if line.startswith("LCRESULT:"):
            try:
                data = json.loads(line[len("LCRESULT:"):])
            except ValueError:
                data = None
            break
    if data is None:
        reason = "زمان ۶۰ ثانیه تمام شد؛ شاید برنامه منتظر ورودی است یا حلقهٔ بی‌پایان بدون flip دارد." if completed.timed_out \
            else f"برنامه پیش از رسم صفحه با کد {completed.returncode} تمام شد."
        return _result(False, plan, "اسکرین‌شات گرفته نشد: " + reason, output=output)
    shots = data.get("shots") or []
    observations = []
    if data.get("error"):
        return _result(False, plan, "برنامه هنگام اجرا خطا داد.", output=data["error"], frames=data.get("frames", 0))
    if not data.get("frames"):
        return _result(False, plan, "برنامه هیچ فریمی رسم نکرد (pygame.display.flip/update هرگز صدا زده نشد)؛ فایل شروع یا حلقهٔ اصلی ناقص است.",
                       output=output, frames=0)
    if data.get("exited_early"):
        observations.append("برنامه پیش از پایان سناریو خودش بسته شد.")
    for shot in shots:
        change = shot.get("changed_vs_previous_pct")
        if shot["name"].startswith("after_click") and change == 0:
            observations.append(f"پس از «{shot['name']}» هیچ تغییری در تصویر دیده نشد (کلیک اثری نداشت یا در جای اشتباه بود).")
        elif change is not None and shot["name"].startswith("after"):
            observations.append(f"پس از «{shot['name']}» حدود {change}٪ از تصویر تغییر کرد.")
    window = data.get("window")
    files = [f"output/screenshots/{shot['file']}" for shot in shots]
    summary = (f"{data.get('frames')} فریم رسم شد؛ اندازهٔ پنجره {window[0]}×{window[1]}؛ {len(shots)} اسکرین‌شات ذخیره شد." if window
               else f"{data.get('frames')} فریم رسم شد؛ {len(shots)} اسکرین‌شات ذخیره شد.")
    return _result(bool(shots), plan, summary, window=window, frames=data.get("frames"), shots=shots, files=files,
                   observations=observations, events=data.get("events", []),
                   folder=str(out_dir), output=output if not shots else "")


def approve_action(token, session_id, cancelled=None):
    """Run an approved action and record what really happened in the project's own manifest."""
    with _lock:
        item = _pending.get(token)
        plan = dict(item["plan"]) if item else None
    result = _approve_inner(token, session_id, cancelled=cancelled)
    if plan and isinstance(result, dict) and plan.get("project") and plan.get("action") != "open_in_editor":
        try:
            _log_action(_project_path(plan["project"]), plan, result)
        except SystemToolError:
            pass
    return result


@register_capability(
    id="approved_system_tools",
    name="برنامه‌نویسی محلی با تأیید شما",
    description="حالت برنامه‌نویسی: طرح مرحله‌به‌مرحله، ساخت پروژه و فایل، ویرایش با پشتیبان، محیط مجازی، نصب کتابخانه، اجرای برنامه و تست روی venv یا در Docker ایزوله.",
    purpose="انجام کارهای برنامه‌نویسی محلی در پوشه‌های مستقل و بدون اجرای فرمان دلخواه shell.",
    activation_conditions=("محیط مجازی", "venv", "virtualenv", "نصب کتابخانه", "pip install", "فایل پایتون",
                           "اجرای پایتون", "ساخت پروژه پایتون", "docker", "داکر", "create python file",
                           "run python", "install package"),
    inputs=("درخواست برنامه‌نویسی", "تأیید طرح", "تأیید هر عملیات اجرایی"),
    outputs=("پروژهٔ مرتب", "فایل‌های کد و تست", "محیط مجازی", "گزارش اجرا و تست"),
    dependencies=("Python 3.10+ روی سیستم (برای venv)", "Docker Desktop (اختیاری، برای تست ایزوله)",
                  "مدل Ollama با پشتیبانی ابزار (مثل qwen3-coder یا qwen2.5-coder)"),
    limitations=("هیچ فرمان shell دلخواهی اجرا نمی‌شود.",
                 "ساخت/ویرایش فایل فقط در پوشهٔ پروژه‌های برنامه و با پسوند متنی مجاز است.",
                 "نصب و اجرا و Docker همیشه تأیید جداگانه می‌خواهند.",
                 "venv و اجرای مستقیم sandbox امنیتی نیستند؛ Docker ایزوله‌تر است.",
                 "برنامه‌های گرافیکی در Docker اجرا نمی‌شوند؛ منطق آن‌ها تست و با run.bat اجرا می‌شود."),
    documentation="system_tools.md",
)
def _registered_system_tools_capability():
    return True
