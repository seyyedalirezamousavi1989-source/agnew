"""Managed, approval-gated Docker runtime for project tests and scripts.

Only two fixed workflows are exposed (test the project / run one console script).
No model-supplied shell, image, Docker flags, host paths or commands are accepted.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import uuid
from pathlib import Path

if __package__:
    from . import toolfinder
    from .procutil import run_capture as _run_capture
else:
    import toolfinder
    from procutil import run_capture as _run_capture


def run_capture(command, **kwargs):
    """Docker commands get the docker folder on PATH (credential helpers live next to docker.exe)."""
    if "env" not in kwargs and command and str(command[0]).lower().endswith(("docker", "docker.exe")):
        kwargs["env"] = toolfinder.env_with_dir(str(command[0]))
    return _run_capture(command, **kwargs)

IMAGE = "python:3.12-slim"
LABEL = "localchat.managed=true"
MAX_LOG = 12000
MAX_REQUIREMENTS = 64 * 1024
MAX_PACKAGES = 80
TEST_TIMEOUT = 600
SCRIPT_TIMEOUT = 60
PACKAGE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}(?:\[[A-Za-z0-9_,.-]+\])?"
    r"(?:\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+"
    r"(?:\s*,\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+)*)?$"
)
SCRIPT_PATH = re.compile(r"^src/(?:[A-Za-z0-9][A-Za-z0-9_.-]{0,59}/){0,3}[A-Za-z0-9][A-Za-z0-9_.-]{0,59}\.py$")

# Runs INSIDE the network-less container. It is mounted read-only as /runner.py
# (no quoting problems on Windows, unlike a very long `python -c` argument).
RUNNER = r'''
import importlib.util, os, pathlib, shutil, subprocess, sys

shutil.copytree("/workspace", "/tmp/project", dirs_exist_ok=True, symlinks=False,
                ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", ".pytest_cache",
                                              ".mypy_cache", "node_modules", ".history"))
os.chdir("/tmp/project")
sys.path.insert(0, "/tmp/deps")
env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join(["/tmp/deps", "/tmp/project", "/tmp/project/src"])
env["PYTHONDONTWRITEBYTECODE"] = "1"
req = pathlib.Path("requirements.txt")
wanted = [x.strip() for x in req.read_text(encoding="utf-8").splitlines()
          if x.strip() and not x.strip().startswith("#")] if req.is_file() else []
if wanted:
    code = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                           "--no-index", "--find-links", "/wheelhouse", "--target", "/tmp/deps",
                           "-r", "requirements.txt"], env=env).returncode
    if code:
        sys.exit(code)

def run(args):
    return subprocess.run(args, env=env).returncode

script = os.environ.get("LC_SCRIPT", "")
if script:
    sys.exit(run([sys.executable, script]))
if pathlib.Path("manage.py").is_file():
    sys.exit(run([sys.executable, "manage.py", "check"]) or run([sys.executable, "manage.py", "test"]))
if not pathlib.Path("tests").is_dir():
    print("پوشهٔ tests وجود ندارد؛ تستی برای اجرا پیدا نشد.")
    sys.exit(5)
if importlib.util.find_spec("pytest"):
    sys.exit(run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]))
sys.exit(run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]))
'''


class DockerRuntimeError(RuntimeError):
    pass


def _docker_exe():
    exe = toolfinder.find("docker")
    if not exe:
        raise DockerRuntimeError("Docker پیدا نشد (نه در PATH و نه در پوشه‌های معمول)؛ Docker Desktop را نصب و اجرا کنید (یا از venv_test / venv_run استفاده شود).")
    return exe


def _docker():
    exe = _docker_exe()
    try:
        probe = run_capture([exe, "info", "--format", "{{.ServerVersion}}"], timeout=25)
    except OSError as exc:
        raise DockerRuntimeError(f"اجرای Docker ممکن نشد: {exc}") from exc
    if probe.timed_out or probe.returncode or not probe.stdout.strip():
        raise DockerRuntimeError("Docker Engine در دسترس نیست؛ Docker Desktop را اجرا کنید و چند ثانیه صبر کنید.")
    return exe


def status():
    """Read-only Docker report for the environment check tool."""
    exe = toolfinder.find("docker")
    info = {"installed": bool(exe), "path": exe or "", "running": False, "version": "", "image_present": False, "image": IMAGE}
    if not exe:
        info["message"] = "Docker روی این رایانه پیدا نشد (PATH و پوشه‌های معمول بررسی شد)."
        return info
    try:
        probe = run_capture([exe, "info", "--format", "{{.ServerVersion}}"], timeout=25)
        if probe.returncode == 0 and probe.stdout.strip():
            info["running"] = True
            info["version"] = probe.stdout.strip()
            image = run_capture([exe, "image", "inspect", IMAGE], timeout=30)
            info["image_present"] = image.returncode == 0
            info["message"] = "Docker آماده است." + ("" if info["image_present"] else " image پایه در اولین اجرا دانلود می‌شود.")
        else:
            info["message"] = "Docker نصب است ولی Engine اجرا نیست؛ Docker Desktop را باز کنید."
    except (OSError, ValueError):
        info["message"] = "بررسی Docker ناموفق بود."
    return info


def _requirements(project: Path):
    path = project / "requirements.txt"
    if not path.exists():
        return "", []
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_REQUIREMENTS:
        raise DockerRuntimeError("requirements.txt باید فایل معمولی حداکثر ۶۴ کیلوبایت باشد.")
    try:
        raw = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DockerRuntimeError("requirements.txt باید با کدگذاری UTF-8 ذخیره شود.") from exc
    specs = []
    for row in raw.splitlines():
        line = row.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-", ".")) or not PACKAGE.fullmatch(line):
            raise DockerRuntimeError(
                f"خط «{line[:60]}» در requirements.txt پذیرفته نیست؛ فقط نام بسته و محدودیت نسخه مجاز است "
                "(URL، مسیر محلی و optionهای pip مجاز نیستند)."
            )
        specs.append(line)
    if len(specs) > MAX_PACKAGES:
        raise DockerRuntimeError("تعداد وابستگی‌ها از حد مجاز بیشتر است.")
    return raw, specs


def _run_container(docker, args, cidfile: Path, timeout: int, label: str, cancelled=None):
    """Run a fixed docker invocation and forcibly remove its container on every exit path."""
    command = [docker, "run", "--cidfile", str(cidfile), "--label", LABEL, "--label", "localchat.run=" + label, *args]
    try:
        return run_capture(command, timeout=timeout, cancelled=cancelled)
    finally:
        try:
            cid = cidfile.read_text(encoding="ascii").strip() if cidfile.exists() else ""
            if cid:
                run_capture([docker, "rm", "-f", cid], timeout=30)
        except (OSError, ValueError):
            pass  # the label-based cleanup below catches leftovers


def cleanup_managed(run_id=None):
    """Remove managed containers, optionally scoped to one run."""
    docker = _docker()
    selector = "localchat.run=" + run_id if run_id else LABEL
    listed = run_capture([docker, "ps", "-aq", "--filter", f"label={selector}"], timeout=30)
    if listed.returncode:
        raise DockerRuntimeError((listed.stderr or "docker ps failed")[-MAX_LOG:])
    removed = []
    for cid in (item.strip() for item in listed.stdout.splitlines() if item.strip()):
        result = run_capture([docker, "rm", "-f", cid], timeout=30)
        if result.returncode:
            raise DockerRuntimeError((result.stderr or f"Could not remove {cid}")[-MAX_LOG:])
        removed.append(cid[:12])
    return {"removed": removed, "count": len(removed)}


def list_managed():
    docker = _docker()
    result = run_capture([docker, "ps", "-a", "--filter", f"label={LABEL}",
                          "--format", "{{.ID}} {{.Status}} {{.Image}}"], timeout=30)
    if result.returncode:
        raise DockerRuntimeError((result.stderr or "docker ps failed")[-MAX_LOG:])
    return {"containers": result.stdout[-MAX_LOG:], "count": len(result.stdout.splitlines())}


def _hint(project: Path, output: str, script, no_tests):
    """A short Persian hint for the most common reasons a test run fails."""
    low = output.lower()
    if "modulenotfounderror" in low or "no module named" in low:
        match = re.search(r"no module named ['\"]([\w.]+)['\"]", output, re.I)
        name = match.group(1).split(".")[0] if match else "?"
        return (f"ماژول «{name}» پیدا نشد: یا در requirements.txt نیست، یا import نادرستی در کد است. "
                "اگر ماژول متعلق به پروژه است، import را نسبت به src یا ریشهٔ پروژه بررسی کن.")
    if no_tests:
        tests = project / "tests"
        uses_pytest = False
        if tests.is_dir():
            for item in list(tests.rglob("*.py"))[:30]:
                try:
                    if re.search(r"^\s*(?:import|from)\s+pytest", item.read_text(encoding="utf-8", errors="replace"), re.M):
                        uses_pytest = True
                        break
                except OSError:
                    pass
        if uses_pytest:
            return "تست‌ها pytest می‌خواهند ولی pytest در requirements.txt نیست؛ pytest را اضافه کن یا تست‌ها را با unittest بنویس."
        return "هیچ تستی پیدا نشد: فایل‌های tests/test_*.py باید کلاس‌هایی از unittest.TestCase (یا تابع‌های pytest همراه pytest در requirements.txt) داشته باشند."
    if "no matching distribution" in low or "could not find a version" in low:
        return "یکی از بسته‌ها wheel باینری برای Python 3.12 لینوکس ندارد؛ نسخه یا بستهٔ دیگری انتخاب کن."
    return ""


def run_project_tests(project_path, expected_requirements_sha256, script=None, cancelled=None):
    """Download wheels in a separate networked container, then test (or run one script) offline."""
    if script is not None and not SCRIPT_PATH.fullmatch(script):
        raise DockerRuntimeError("مسیر اسکریپت برای اجرا معتبر نیست.")
    docker = _docker()
    project = Path(project_path).resolve(strict=True)
    if not project.is_dir() or project.is_symlink():
        raise DockerRuntimeError("مسیر پروژه معتبر نیست.")
    req_text, specs = _requirements(project)
    digest = hashlib.sha256(req_text.encode("utf-8")).hexdigest()
    if digest != expected_requirements_sha256:
        raise DockerRuntimeError("requirements.txt پس از تأیید تغییر کرده؛ دوباره پیش‌نمایش بگیرید.")

    if run_capture([docker, "image", "inspect", IMAGE], timeout=30).returncode:
        pulled = run_capture([docker, "pull", IMAGE], timeout=600, cancelled=cancelled)
        if pulled.cancelled:
            raise DockerRuntimeError("عملیات توسط کاربر متوقف شد.")
        if pulled.returncode:
            raise DockerRuntimeError("دریافت image پایهٔ Python ناموفق بود (اینترنت لازم است)؛\n" +
                                     (pulled.stderr or pulled.stdout)[-MAX_LOG:])

    logs = []
    run_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="localchat-docker-") as tmp:
        temp = Path(tmp)
        wheelhouse = temp / "wheelhouse"
        wheelhouse.mkdir()
        runner = temp / "runner.py"
        runner.write_text(RUNNER, encoding="utf-8")
        if specs:
            req = project / "requirements.txt"
            download = _run_container(
                docker,
                [
                    "--pull=missing", "--network", "bridge", "--read-only",
                    "--env", "HOME=/tmp", "--env", "PIP_NO_CACHE_DIR=1",
                    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                    "--pids-limit", "48", "--memory", "512m", "--cpus", "1",
                    "--tmpfs", "/tmp:rw,nosuid,size=128m",
                    "--mount", f"type=bind,src={req.resolve()},dst=/requirements.txt,readonly",
                    "--mount", f"type=bind,src={wheelhouse.resolve()},dst=/wheelhouse",
                    IMAGE, "python", "-m", "pip", "download",
                    "--disable-pip-version-check", "--only-binary=:all:",
                    "--index-url", "https://pypi.org/simple",
                    "--dest", "/wheelhouse", "-r", "/requirements.txt",
                ],
                temp / "download.cid", 300, run_id, cancelled,
            )
            if download.cancelled:
                raise DockerRuntimeError("عملیات توسط کاربر متوقف شد.")
            logs.append("Dependency download (PyPI wheels only):\n" + (download.stdout + "\n" + download.stderr)[-MAX_LOG:])
            if download.returncode or download.timed_out:
                raise DockerRuntimeError(
                    "دریافت wheelهای وابستگی از PyPI ناموفق بود؛ هیچ بسته‌ای روی ویندوز نصب نشد.\n" + logs[-1][-MAX_LOG:]
                )

        # Test phase: no network, project mounted read-only, copied to container-local tmpfs.
        timeout = SCRIPT_TIMEOUT if script else TEST_TIMEOUT
        tested = _run_container(
            docker,
            [
                "--pull=never", "--network", "none", "--read-only",
                "--env", "HOME=/tmp", "--env", "PIP_NO_CACHE_DIR=1",
                "--env", "SDL_VIDEODRIVER=dummy", "--env", "SDL_AUDIODRIVER=dummy",
                "--env", "PYTHONUNBUFFERED=1", "--env", "PYTHONIOENCODING=utf-8",
                "--env", "LC_SCRIPT=" + (script or ""),
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "64", "--memory", "768m", "--cpus", "1.5",
                "--tmpfs", "/tmp:rw,nosuid,size=256m",
                "--mount", f"type=bind,src={project},dst=/workspace,readonly",
                "--mount", f"type=bind,src={wheelhouse.resolve()},dst=/wheelhouse,readonly",
                "--mount", f"type=bind,src={runner.resolve()},dst=/runner.py,readonly",
                "--workdir", "/tmp", IMAGE, "python", "/runner.py",
            ],
            temp / "tests.cid", timeout, run_id, cancelled,
        )
        if tested.cancelled:
            cleanup_managed(run_id)
            raise DockerRuntimeError("عملیات توسط کاربر متوقف شد.")
        output = (tested.stdout or "") + "\n" + (tested.stderr or "")
        if tested.timed_out:
            output += f"\n[زمان اجرا بعد از {timeout} ثانیه تمام شد و کانتینر متوقف شد؛ برنامه‌های بی‌پایان یا تعاملی در Docker اجرا نمی‌شوند.]"
        logs.append("Offline test run:\n" + output[-MAX_LOG:])
        low = output.lower()
        no_tests = False if script else ("ran 0 tests" in low or "no tests ran" in low or "collected 0 items" in low)
        cleanup = cleanup_managed(run_id)  # remove anything a killed docker CLI left behind
        ok = tested.returncode == 0 and not tested.timed_out and not no_tests and cleanup["count"] == 0
        return {
            "ok": ok,
            "runtime": "Docker",
            "returncode": tested.returncode,
            "timed_out": tested.timed_out,
            "tests_discovered": not no_tests,
            "dependency_count": len(specs),
            "dependency_source": "PyPI binary wheels; downloaded in separate container",
            "test_network": "disabled",
            "workspace_mount": "read-only",
            "container_cleanup": cleanup,
            "hint": "" if ok else _hint(project, output, script, no_tests),
            "output": "\n\n".join(logs)[-MAX_LOG:],
        }
