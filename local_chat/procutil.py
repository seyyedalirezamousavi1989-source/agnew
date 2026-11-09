"""Small subprocess helper shared by the programming tools.

* captured stdout/stderr (bounded), UTF-8 decoding that never raises
* hard timeout and cooperative cancellation (the stop button)
* the whole process tree is killed, and no console window pops up on Windows
"""
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from collections import namedtuple

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
MAX_CAPTURE_BYTES = 256 * 1024

RunResult = namedtuple("RunResult", "returncode stdout stderr timed_out cancelled")


def _windows_command(command):
    """Make Windows command scripts (.bat/.cmd) safely executable with CreateProcess."""
    if os.name != "nt" or not isinstance(command, (list, tuple)) or not command:
        return command
    executable = os.fspath(command[0])
    if Path(executable).suffix.lower() not in {".bat", ".cmd"}:
        return command
    comspec = os.environ.get("COMSPEC") or os.path.join(
        os.environ.get("SystemRoot", r"C:\\Windows"), "System32", "cmd.exe"
    )
    # For batch files, pass one deliberately constructed command-line string.
    # Passing [cmd.exe, ..., command_string] makes Popen/list2cmdline escape the
    # embedded quotes a second time; cmd.exe then receives malformed quoting
    # (notably when the .cmd path or arguments contain spaces). The canonical
    # cmd.exe form is: "cmd.exe" /d /s /c ""script.cmd" arg1 arg2".
    command_line = subprocess.list2cmdline([executable, *map(str, command[1:])])
    return f'"{comspec}" /d /s /c "{command_line}"'


def kill_tree(process):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=10, creationflags=CREATE_NO_WINDOW)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _drain(stream, sink):
    total = 0
    try:
        for chunk in iter(lambda: stream.read(4096), b""):
            sink.append(chunk)
            total += len(chunk)
            while total > MAX_CAPTURE_BYTES and len(sink) > 1:
                total -= len(sink.pop(0))
    except (OSError, ValueError):
        pass


def run_capture(command, cwd=None, timeout=60, env=None, cancelled=None):
    """Run ``command`` (a list, never a shell string) and return a RunResult.

    Raises OSError (e.g. FileNotFoundError) when the executable cannot be started.
    """
    options = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
               "cwd": str(cwd) if cwd else None, "env": env}
    if os.name == "nt":
        options["creationflags"] = CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    process = subprocess.Popen(_windows_command(command), **options)
    out_chunks, err_chunks = [], []
    readers = [threading.Thread(target=_drain, args=(process.stdout, out_chunks), daemon=True),
               threading.Thread(target=_drain, args=(process.stderr, err_chunks), daemon=True)]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    timed_out = was_cancelled = False
    while True:
        try:
            process.wait(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if cancelled is not None and cancelled():
                was_cancelled = True
            elif time.monotonic() > deadline:
                timed_out = True
            else:
                continue
            kill_tree(process)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            break
    for reader in readers:
        reader.join(timeout=5)
    for stream in (process.stdout, process.stderr):
        try:
            stream.close()
        except (OSError, ValueError):
            pass
    return RunResult(
        process.returncode if process.returncode is not None else -1,
        b"".join(out_chunks).decode("utf-8", "replace"),
        b"".join(err_chunks).decode("utf-8", "replace"),
        timed_out, was_cancelled,
    )
