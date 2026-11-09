import ctypes
import json
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


# PyInstaller executes the entry script as __main__, so it has no package
# context and relative imports fail. Add the project/bundle root explicitly,
# then always import through the real local_chat package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_chat import settings as app_settings
from local_chat.version import APP_VERSION

APP_DIR = Path(__file__).resolve().parent
SERVER_FILE = APP_DIR / "server.py"
CHAT_URL = "http://127.0.0.1:8000/"
STATUS_URL = "http://127.0.0.1:8000/api/status"
OLLAMA_URL = "http://127.0.0.1:11434/api/tags"
STARTUP_TIMEOUT = 60


def server_log_path():
    return app_settings.app_data_dir() / "server.log"


def read_log_tail(path, max_chars=800):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text.strip()[-max_chars:]


def show_error(message):
    ctypes.windll.user32.MessageBoxW(None, message, "گفت‌وگوی محلی", 0x10)


def chat_server_is_running():
    try:
        with urlopen(STATUS_URL, timeout=2) as response:
            return json.loads(response.read().decode("utf-8")).get("version") == APP_VERSION
    except (OSError, URLError, TimeoutError, ValueError):
        return False


def port_is_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        return connection.connect_ex(("127.0.0.1", port)) != 0


def find_ollama():
    executable = shutil.which("ollama")
    if executable:
        return executable
    candidates = [
        Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
    ]
    return next((str(path) for path in candidates if path.is_file()), None)


def ollama_is_running():
    try:
        with urlopen(OLLAMA_URL, timeout=2) as response:
            return response.status == 200
    except (OSError, URLError, TimeoutError):
        return False


def wait_for_ollama():
    if ollama_is_running():
        return

    executable = find_ollama()
    if not executable:
        raise RuntimeError(
            "Ollama پیدا نشد. ابتدا Ollama را نصب کنید یا برنامهٔ Ollama را باز کنید."
        )

    subprocess.Popen(
        [executable, "serve"],
        env=app_settings.ollama_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if ollama_is_running():
            return
        time.sleep(1)
    raise RuntimeError(
        "Ollama در زمان مناسب اجرا نشد. برنامهٔ Ollama را باز کنید و دوباره تلاش کنید."
    )


def start_chat_server():
    if chat_server_is_running():
        return
    if not port_is_available(8000):
        raise RuntimeError(
            "درگاه ۸۰۰۰ توسط برنامهٔ دیگری استفاده شده است. آن برنامه را ببندید و دوباره تلاش کنید."
        )

    # In a PyInstaller build, sys.executable is LocalChat.exe, not python.exe.
    # Relaunch the frozen application in its server mode instead of passing
    # server.py as a script (which would simply reopen the GUI launcher).
    command = (
        [sys.executable, "--localchat-server"]
        if getattr(sys, "frozen", False)
        else [sys.executable, str(SERVER_FILE)]
    )
    log_path = server_log_path()
    log_handle = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "ab")
    except OSError:
        log_handle = None
    process = subprocess.Popen(
        command,
        cwd=APP_DIR,
        stdin=subprocess.DEVNULL,
        stdout=log_handle or subprocess.DEVNULL,
        stderr=subprocess.STDOUT if log_handle else subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if log_handle:
        log_handle.close()
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if chat_server_is_running():
            return process
        if process is not None and process.poll() is not None:
            break  # the server process already died: no point waiting any longer
        time.sleep(0.25)
    detail = ""
    if process is not None and process.poll() is not None:
        detail = f"\n\nپردازش سرور با کد {process.returncode} بسته شد."
    tail = read_log_tail(log_path)
    if tail:
        detail += "\n\nآخرین خطای ثبت‌شده:\n" + tail
    raise RuntimeError(
        "سرور چت اجرا نشد. مطمئن شوید فایل‌های برنامه کامل از ZIP استخراج شده‌اند."
        f"{detail}\n\nفایل گزارش: {log_path}"
    )


def open_app_window(server_process=None):
    """Show LocalChat in its own desktop window (WebView2). Returns False if that is impossible."""
    try:
        import webview
    except Exception:
        return False
    try:
        window = webview.create_window(
            "گفت‌وگوی محلی", CHAT_URL, width=1360, height=900, min_size=(900, 600)
        )

        def close_when_server_stops():
            # The in-app "shutdown" button stops the server process: close the window with it.
            misses = 0
            while True:
                time.sleep(3)
                misses = 0 if chat_server_is_running() else misses + 1
                if misses >= 2:
                    try:
                        window.destroy()
                    except Exception:
                        pass
                    return

        storage = app_settings.app_data_dir() / "webview"
        storage.mkdir(parents=True, exist_ok=True)
        webview.start(close_when_server_stops, storage_path=str(storage), private_mode=False)
    except Exception:
        return False
    # The window was closed by the user: stop the server this launcher started.
    if server_process is not None and server_process.poll() is None:
        server_process.terminate()
    return True


def main():
    try:
        wait_for_ollama()
        server_process = start_chat_server()
        if not open_app_window(server_process):
            webbrowser.open(CHAT_URL)
    except (OSError, RuntimeError) as error:
        show_error(str(error))
        return 1
    return 0


if __name__ == "__main__":
    if "--localchat-server" in sys.argv:
        # This path is used only by the frozen executable's child process.
        # A windowed (console=False) executable has sys.stdout/sys.stderr == None, which makes
        # http.server's request logging raise AttributeError and drop every response.
        import os
        for _name in ("stdout", "stderr"):
            if getattr(sys, _name) is None:
                setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))
        from local_chat.server import main as server_main

        server_main()
    else:
        raise SystemExit(main())
