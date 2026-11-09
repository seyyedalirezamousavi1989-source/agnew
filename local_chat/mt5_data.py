"""Read-only access to MetaTrader 5 for LocalChat's forex mode.

The official ``MetaTrader5`` Python package talks to a MetaTrader 5 terminal that is
installed on the same Windows machine. It needs a 64-bit Python and a terminal that is
logged in to a broker (a free demo account is enough).

Safety: this module only *reads* (terminal/account summary, quotes, candles). It never
places, modifies or closes orders, and it never stores a broker login or password —
the terminal keeps its own session.
"""
import atexit
import glob
import os
import subprocess
import math
import queue
import re
import sys
import threading
import time

TIMEFRAME_ATTRS = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1", "W1": "TIMEFRAME_W1",
}
INIT_TIMEOUT_MS = 30000     # a terminal that is not running yet has to be started first
CONNECT_WAIT_SECONDS = 15   # time the terminal gets to reach the broker's server
TICK_WAIT_SECONDS = 3.0
RATES_RETRIES = 4
DEFAULT_BARS = 500
DEEP_BARS = 6000            # history of the trading timeframe for the empirical probability model
MIN_BARS = 50
ACCOUNT_MODES = {0: "دمو", 1: "مسابقه", 2: "واقعی"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._#\-+]{1,32}$")

CALL_TIMEOUT_SECONDS = 120
_LOCK = threading.RLock()
_STATE = {"ready": False, "path": "", "atexit": False}


class MT5Error(Exception):
    """Problem with MetaTrader 5. ``message`` is Persian and safe to show to the user."""

    def __init__(self, message, code="error"):
        super().__init__(message)
        self.message = message
        self.code = code


class _Worker:
    """Runs every MetaTrader5 call on one dedicated daemon thread.

    The MetaTrader5 package talks to the terminal over a single IPC pipe and is not
    documented as thread-safe, while the chat server answers each request on its own
    thread. One worker thread gives every call the same thread and serialises them.
    A daemon thread also means a hung terminal can never block closing the app.
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._thread = None
        self._start_lock = threading.Lock()

    def _ensure_thread(self):
        with self._start_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="mt5-worker", daemon=True)
                self._thread.start()

    def _run(self):
        while True:
            function, args, box = self._queue.get()
            try:
                box["value"] = function(*args)
            except BaseException as exc:  # noqa: BLE001 - handed back to the caller
                box["error"] = exc
            finally:
                box["done"].set()

    def call(self, function, *args, timeout=CALL_TIMEOUT_SECONDS):
        self._ensure_thread()
        box = {"done": threading.Event()}
        self._queue.put((function, args, box))
        if not box["done"].wait(timeout):
            raise MT5Error("MetaTrader 5 به‌موقع پاسخ نداد. ترمینال را بررسی کن (ممکن است در حال به‌روزرسانی یا قفل‌شده باشد).",
                           "timeout")
        if "error" in box:
            raise box["error"]
        return box["value"]


_WORKER = _Worker()


def _import_mt5():
    """Import the MetaTrader5 package (tests replace this function)."""
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ModuleNotFoundError as exc:
        # Distinguish the package itself from a missing transitive dependency.
        missing = getattr(exc, "name", None) or "نامشخص"
        if missing != "MetaTrader5":
            raise MT5Error(
                f"یکی از وابستگی‌های MetaTrader5 بارگذاری نشد ({missing}). "
                "نصب Python و فایل‌های همراه برنامه را بررسی کنید.", "dependency_missing") from None
        if sys.platform != "win32":
            raise MT5Error("پکیج MetaTrader5 فقط روی ویندوز کار می‌کند.", "not_supported") from None
        if getattr(sys, "frozen", False):
            raise MT5Error(
                "پکیج MetaTrader5 داخل نسخهٔ نصب‌شده پیدا نشد. نسخهٔ جدید را با "
                "build_installer.bat بسازید؛ بسته‌بندی باید ماژول‌ها و فایل‌های باینری "
                "MetaTrader5 را نیز همراه برنامه قرار دهد.", "not_installed") from None
        raise MT5Error(
            "پکیج MetaTrader5 نصب نیست. در CMD اجرا کنید: "
            "python -m pip install MetaTrader5 (با Python ویندوز ۶۴ بیتی).", "not_installed") from None
    except OSError as exc:
        raise MT5Error(
            "فایل باینری MetaTrader5 بارگذاری نشد؛ احتمالاً DLL وابسته موجود نیست "
            "یا معماری Python/برنامه ناسازگار است. نسخهٔ ۶۴ بیتی Python و فایل‌های "
            f"همراه نصب را بررسی کنید. جزئیات: {exc}", "binary_load_failed") from None
    except ImportError as exc:
        raise MT5Error(
            f"MetaTrader5 پیدا شد اما یکی از اجزای آن قابل بارگذاری نیست: {exc}. "
            "بسته‌بندی یا وابستگی‌های DLL را بررسی کنید.", "import_failed") from None
    return mt5


# Stage of the MetaTrader 5 stack that failed (so the UI/log can tell three different
# problems apart): the Python package, its native DLL, the terminal, the broker link.
ERROR_STAGES = {
    "not_installed": "package", "dependency_missing": "package", "not_supported": "package",
    "binary_load_failed": "dll", "import_failed": "dll",
    "init_failed": "terminal", "timeout": "terminal",
    "not_connected": "connection", "no_tick": "connection",
    "symbol_not_found": "symbol", "symbol_select_failed": "symbol", "bad_symbol": "symbol",
    "no_rates": "data",
}


def error_stage(code):
    return ERROR_STAGES.get(code, "unknown")


def diagnose(terminal_path=""):
    """Staged health check: [{"stage", "ok", "message"}]. Stops at the first failing stage."""
    steps = []
    try:
        _import_mt5()
        steps.append({"stage": "package+dll", "ok": True, "message": "پکیج MetaTrader5 و DLL آن بارگذاری شد."})
    except MT5Error as exc:
        steps.append({"stage": error_stage(exc.code), "ok": False, "code": exc.code, "message": exc.message})
        return steps
    try:
        connect(terminal_path)
        steps.append({"stage": "terminal+connection", "ok": True, "message": "ترمینال در دسترس و به بروکر متصل است."})
    except MT5Error as exc:
        steps.append({"stage": error_stage(exc.code), "ok": False, "code": exc.code, "message": exc.message})
    return steps


def is_installed():
    try:
        _import_mt5()
    except MT5Error:
        return False
    return True


def _last_error(mt5):
    try:
        code, text = mt5.last_error()
        return f"{code}: {text}"
    except Exception:  # noqa: BLE001 - diagnostics only
        return "نامشخص"


def _quiet_shutdown(mt5):
    try:
        mt5.shutdown()
    except Exception:  # noqa: BLE001
        pass
    _STATE["ready"] = False


def _shutdown_impl():
    with _LOCK:
        if not _STATE["ready"]:
            return
        try:
            _quiet_shutdown(_import_mt5())
        except MT5Error:
            _STATE["ready"] = False


INIT_RETRY_PAUSE = 3.0
_IPC_CODES = (-10005, -10003, -10004)   # IPC timeout / no IPC connection / terminal not found
_CANDIDATES_CACHE = {"paths": None}


def discover_terminals():
    """Paths of installed/running terminal64.exe files, running ones first (Windows; [] elsewhere)."""
    if _CANDIDATES_CACHE["paths"] is not None:
        return list(_CANDIDATES_CACHE["paths"])
    found = []

    def add(path):
        path = str(path).strip().strip('"')
        if path.lower().endswith("terminal64.exe") and path not in found and os.path.isfile(path):
            found.append(path)

    if sys.platform == "win32":
        try:    # a terminal that is already running is the one the user logged in to
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-Process terminal64 -ErrorAction SilentlyContinue).Path"],
                capture_output=True, text=True, timeout=8, creationflags=0x08000000).stdout
            for line in out.splitlines():
                add(line)
        except (OSError, subprocess.SubprocessError):
            pass
        roots = [os.environ.get(k, "") for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA", "APPDATA")]
        roots += [os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")]
        for root in filter(None, roots):
            for pattern in ("*/terminal64.exe", "*/*/terminal64.exe"):
                for hit in sorted(glob.glob(os.path.join(root, pattern))):
                    add(hit)
        try:
            import winreg  # type: ignore
            for hive, key in ((winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                              (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")):
                with winreg.OpenKey(hive, key) as root_key:
                    for index in range(winreg.QueryInfoKey(root_key)[0]):
                        try:
                            with winreg.OpenKey(root_key, winreg.EnumKey(root_key, index)) as sub:
                                name = winreg.QueryValueEx(sub, "DisplayName")[0]
                                if "metatrader" in str(name).lower():
                                    add(os.path.join(winreg.QueryValueEx(sub, "InstallLocation")[0], "terminal64.exe"))
                        except OSError:
                            continue
        except (ImportError, OSError):
            pass
    _CANDIDATES_CACHE["paths"] = found
    return list(found)


def _initialize_auto(mt5, terminal_path):
    """initialize() with automatic recovery: retry on IPC timeouts, then try every discovered terminal."""
    remembered = _STATE.get("found", "")
    attempts = [terminal_path] if terminal_path else ([remembered] if remembered else [""])
    if not terminal_path:
        attempts += [p for p in [""] + discover_terminals() if p not in attempts]
    for path in attempts:
        for attempt in range(2):
            ok = mt5.initialize(path, timeout=INIT_TIMEOUT_MS) if path else mt5.initialize(timeout=INIT_TIMEOUT_MS)
            if ok:
                if path and not terminal_path:
                    _STATE["found"] = path
                return True
            code = None
            try:
                code = mt5.last_error()[0]
            except Exception:  # noqa: BLE001
                pass
            _quiet_shutdown(mt5)
            if code not in _IPC_CODES:
                break                       # a different failure: another attempt on this path will not help
            time.sleep(INIT_RETRY_PAUSE)    # the terminal may still be starting up
    return False


def _connect_impl(terminal_path=""):
    with _LOCK:
        mt5 = _import_mt5()
        if _STATE["ready"] and _STATE["path"] == terminal_path:
            info = mt5.terminal_info()
            if info is not None and getattr(info, "connected", False):
                return info
            _quiet_shutdown(mt5)  # terminal was closed or lost the broker: start over
        elif _STATE["ready"]:
            _quiet_shutdown(mt5)
        started = _initialize_auto(mt5, terminal_path)
        if not started:
            raise MT5Error(
                "اتصال به ترمینال MetaTrader 5 برقرار نشد. ترمینال را باز کنید و وارد حساب (دمو هم کافی است) شوید. "
                "برنامه همهٔ ترمینال‌های نصب‌شده را هم خودکار امتحان کرد. اگر باز هم نشد: ترمینال و برنامه باید با سطح دسترسی یکسان "
                "(هر دو عادی یا هر دو «Run as administrator») اجرا شوند و Python/MetaTrader5 هم ۶۴بیتی باشند؛ "
                "یا مسیر terminal64.exe را در «تنظیمات» بدهید. "
                f"(جزئیات: {_last_error(mt5)})", "init_failed")
        _STATE["ready"] = True
        _STATE["path"] = terminal_path
        if not _STATE["atexit"]:
            atexit.register(shutdown)
            _STATE["atexit"] = True
        deadline = time.monotonic() + CONNECT_WAIT_SECONDS
        info = mt5.terminal_info()
        while (info is None or not getattr(info, "connected", False)) and time.monotonic() < deadline:
            time.sleep(0.5)
            info = mt5.terminal_info()
        if info is None or not getattr(info, "connected", False):
            _quiet_shutdown(mt5)
            raise MT5Error(
                "ترمینال MetaTrader 5 باز است ولی به سرور بروکر وصل نیست. "
                "اینترنت و ورود به حساب را در خود ترمینال بررسی کنید.", "not_connected")
        return info


def _terminal_connected():
    if not _STATE["ready"]:
        return False
    info = _import_mt5().terminal_info()
    return bool(info is not None and getattr(info, "connected", False))


def peek():
    """Cheap state for the UI. Never starts the terminal."""
    installed = is_installed()
    state = {"installed": installed, "connected": False, "busy": False}
    if not installed or not _STATE["ready"]:
        return state
    try:
        state["connected"] = _WORKER.call(_terminal_connected, timeout=0.5)
    except MT5Error as exc:
        state["busy"] = exc.code == "timeout"   # the worker is busy with a long request
    return state


def _clean_symbol(name):
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        return None
    return name


def resolve_symbol(mt5, base, suffix=""):
    """Broker symbol name for a standard symbol such as EURUSD.

    Brokers add suffixes or prefixes (EURUSD.a, EURUSDm, EURUSD#, …). Order of tries:
    the configured suffix, the plain name, then the shortest name containing it.
    """
    for candidate in ([base + suffix] if suffix else []) + [base]:
        if mt5.symbol_info(candidate) is not None:
            return candidate
    found = mt5.symbols_get(f"*{base}*") or ()
    names = [getattr(item, "name", "") for item in found]
    names = [n for n in names if _clean_symbol(n) and base in n.upper()]
    if not names:
        raise MT5Error(
            f"نماد {base} در این بروکر پیدا نشد. اگر بروکرت پسوند دارد (مثل EURUSD.a) آن را در «تنظیمات» وارد کن "
            "یا نماد را در Market Watch ترمینال فعال کن.", "symbol_not_found")
    names.sort(key=lambda n: (not n.upper().startswith(base), len(n), n))
    return names[0]


def _wait_for_tick(mt5, name):
    deadline = time.monotonic() + TICK_WAIT_SECONDS
    while True:
        tick = mt5.symbol_info_tick(name)
        if tick is not None and getattr(tick, "time", 0):
            return tick
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.2)


def _columns(rates):
    """numpy structured array (or similar) → plain lists, dropping corrupt candles."""
    try:
        raw = {key: list(rates[key]) for key in ("time", "open", "high", "low", "close")}
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    clean = {key: [] for key in raw}
    for i in range(len(raw["time"])):
        try:
            t = int(raw["time"][i])
            o, h, l, c = (float(raw[key][i]) for key in ("open", "high", "low", "close"))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) and v > 0 for v in (o, h, l, c)) or h < l:
            continue
        if clean["time"] and t <= clean["time"][-1]:
            continue
        clean["time"].append(t)
        clean["open"].append(o)
        clean["high"].append(h)
        clean["low"].append(l)
        clean["close"].append(c)
    return clean


def _copy_rates(mt5, name, tf_name, bars):
    constant = getattr(mt5, TIMEFRAME_ATTRS[tf_name])
    for attempt in range(RATES_RETRIES):
        rates = mt5.copy_rates_from_pos(name, constant, 0, bars)
        if rates is not None and len(rates) > 0:
            columns = _columns(rates)
            if columns and len(columns["time"]) >= MIN_BARS:
                return columns
        time.sleep(0.4 * (attempt + 1))  # history is still being synchronised
    return None


def _fetch_impl(symbol, timeframes, bars, suffix, terminal_path):
    with _LOCK:
        mt5 = _import_mt5()
        _connect_impl(terminal_path)
        name = resolve_symbol(mt5, symbol, suffix)
        info = mt5.symbol_info(name)
        if info is None:
            raise MT5Error(f"اطلاعات نماد {name} از MT5 خوانده نشد.", "symbol_not_found")
        if not getattr(info, "visible", True) and not mt5.symbol_select(name, True):
            raise MT5Error(f"نماد {name} در Market Watch فعال نشد ({_last_error(mt5)}).", "symbol_select_failed")
        tick = _wait_for_tick(mt5, name)
        if tick is None or (not getattr(tick, "bid", 0) and not getattr(tick, "ask", 0)):
            raise MT5Error(f"برای {name} قیمت زنده‌ای از بروکر نرسید (نماد بدون اشتراک قیمت یا بازار بسته).", "no_tick")
        rates, warnings = {}, []
        for tf_name in timeframes:
            count = bars.get(tf_name, DEFAULT_BARS) if isinstance(bars, dict) else bars
            columns = _copy_rates(mt5, name, tf_name, count)
            if columns is None:
                warnings.append(tf_name)
            else:
                rates[tf_name] = columns
        if not rates:
            raise MT5Error(f"کندلی برای {name} از MT5 دریافت نشد؛ چند ثانیه بعد دوباره امتحان کن.", "no_rates")
        return {
            "symbol": symbol,
            "broker_symbol": name,
            "info": {
                "digits": int(getattr(info, "digits", 5)),
                "point": float(getattr(info, "point", 0.0) or 0.0),
            },
            "tick": {
                "time": int(getattr(tick, "time", 0)) or None,
                "bid": float(getattr(tick, "bid", 0.0) or 0.0) or None,
                "ask": float(getattr(tick, "ask", 0.0) or 0.0) or None,
            },
            "rates": rates,
            "failed_timeframes": warnings,
        }


def _raw_columns(rates):
    """Uncleaned lists (incl. spread/tick volume when the terminal provides them) for the backtest's own audit."""
    out = {}
    for key in ("time", "open", "high", "low", "close", "spread", "tick_volume"):
        try:
            out[key] = [float(v) if key not in ("time",) else int(v) for v in rates[key]]
        except (KeyError, IndexError, ValueError, TypeError):
            if key in ("time", "open", "high", "low", "close"):
                return None
    return out


def _history_impl(symbol, bars, suffix, terminal_path):
    with _LOCK:
        mt5 = _import_mt5()
        _connect_impl(terminal_path)
        name = resolve_symbol(mt5, symbol, suffix)
        info = mt5.symbol_info(name)
        if info is None:
            raise MT5Error(f"اطلاعات نماد {name} از MT5 خوانده نشد.", "symbol_not_found")
        if not getattr(info, "visible", True) and not mt5.symbol_select(name, True):
            raise MT5Error(f"نماد {name} در Market Watch فعال نشد ({_last_error(mt5)}).", "symbol_select_failed")
        tick = mt5.symbol_info_tick(name)
        frames, failed = {}, []
        for tf_name, count in bars.items():
            constant = getattr(mt5, TIMEFRAME_ATTRS[tf_name])
            columns = None
            for attempt in range(RATES_RETRIES):
                rates = mt5.copy_rates_from_pos(name, constant, 0, int(count))
                if rates is not None and len(rates) > 0:
                    columns = _raw_columns(rates)
                    if columns and len(columns["time"]) >= MIN_BARS:
                        break
                    columns = None
                time.sleep(0.6 * (attempt + 1))      # history is still being downloaded
            if columns is None:
                failed.append(tf_name)
            else:
                frames[tf_name] = columns
        if not frames:
            raise MT5Error(f"تاریخچهٔ {name} از MT5 دریافت نشد؛ چند ثانیه بعد دوباره امتحان کن.", "no_rates")
        return {
            "symbol": symbol, "broker_symbol": name,
            "info": {"digits": int(getattr(info, "digits", 5)), "point": float(getattr(info, "point", 0.0) or 0.0),
                     "spread_points": float(getattr(info, "spread", 0) or 0)},
            "tick": {"time": int(getattr(tick, "time", 0)) or None if tick is not None else None},
            "fetched_at": time.time(), "frames": frames, "failed_timeframes": failed,
        }


def fetch_history(symbol, bars, suffix="", terminal_path=""):
    """Raw (uncleaned) candle history for the backtest. ``bars``: {timeframe: count}."""
    if not _SAFE_NAME.match(symbol):
        raise MT5Error("نام نماد نامعتبر است.", "bad_symbol")
    return _WORKER.call(_history_impl, symbol, dict(bars), suffix, terminal_path, timeout=600)


def _summary_impl(terminal_path, suffix):
    with _LOCK:
        mt5 = _import_mt5()
        info = _connect_impl(terminal_path)
        account = mt5.account_info()
        result = {
            "terminal": str(getattr(info, "name", "") or "MetaTrader 5")[:60],
            "build": getattr(info, "build", None),
            "connected": bool(getattr(info, "connected", False)),
            "ping_ms": None,
            "server": "",
            "account_type": "",
            "sample": None,
        }
        ping = getattr(info, "ping_last", None)
        if isinstance(ping, (int, float)) and ping > 0:
            result["ping_ms"] = round(ping / 1000.0, 1)  # microseconds → ms
        if account is not None:
            result["server"] = str(getattr(account, "server", "") or "")[:60]
            result["account_type"] = ACCOUNT_MODES.get(getattr(account, "trade_mode", None), "")
        try:
            name = resolve_symbol(mt5, "EURUSD", suffix)
            tick = _wait_for_tick(mt5, name)
            if tick is not None:
                result["sample"] = {"symbol": name, "bid": float(tick.bid), "ask": float(tick.ask)}
        except MT5Error:
            pass
        return result


# ---- public API: everything runs on the worker thread ---------------------------------

def connect(terminal_path=""):
    """Make sure the terminal is reachable and logged in. Returns the terminal info object."""
    return _WORKER.call(_connect_impl, terminal_path)


def shutdown():
    """Release the connection to the terminal (the terminal itself keeps running)."""
    try:
        _WORKER.call(_shutdown_impl, timeout=10)
    except (MT5Error, RuntimeError):
        _STATE["ready"] = False


def fetch_symbol(symbol, timeframes, bars=DEFAULT_BARS, suffix="", terminal_path=""):
    """Raw quote + candles of one symbol: the input of ``forex_analysis.build_snapshot``.

    ``bars`` is one number for every timeframe or a ``{timeframe: count}`` mapping (the
    trading timeframe is fetched deeper because the probability model learns from it).
    """
    return _WORKER.call(_fetch_impl, symbol, list(timeframes), bars, suffix, terminal_path)


def summary(terminal_path="", suffix=""):
    """Connection report for the settings page: no login number, no balance."""
    return _WORKER.call(_summary_impl, terminal_path, suffix)
