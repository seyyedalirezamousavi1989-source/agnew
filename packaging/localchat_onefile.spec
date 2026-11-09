# Single-file variant: one portable LocalChat.exe (slower first start, extracts to temp)
# PyInstaller spec — run from the project root via build_installer.bat
import os
from PyInstaller.utils.hooks import collect_all

try:
    mt5_datas, mt5_binaries, mt5_hiddenimports = collect_all("MetaTrader5")
    # Rely on PyInstaller's built-in hook-numpy.py for NumPy's native runtime.
    # recursive NumPy collection over-collects tests/docs and may break COLLECT.
except Exception as exc:
    raise RuntimeError(
        "MetaTrader5 or NumPy could not be collected. Install both in the "
        "64-bit build Python environment. "
        f"Original error: {type(exc).__name__}: {exc}"
    ) from exc

# --- Workaround for Python 3.10.0's broken `dis` module -----------------------
# On CPython 3.10.0, dis.get_instructions() can raise IndexError for some
# library bytecode, which aborts PyInstaller's analysis. Skip the instructions
# of such a code object instead of crashing (hiddenimports below cover the rest).
import sys as _sys
if _sys.version_info[:3] == (3, 10, 0):
    import PyInstaller.lib.modulegraph.util as _mg_util
    _orig_iter = _mg_util.iterate_instructions

    def _safe_iter(code_object):
        try:
            for inst in _orig_iter(code_object):
                yield inst
        except IndexError:
            return

    _mg_util.iterate_instructions = _safe_iter
# -------------------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
PKG = os.path.join(ROOT, "local_chat")

# Edge TTS (voice replies): without this the installed EXE cannot import edge_tts/aiohttp/certifi and speech fails.
# Collected defensively: a missing optional package must not break the build (the app then reports it clearly).
edge_datas, edge_hiddenimports = [], []
try:
    from PyInstaller.utils.hooks import collect_data_files, collect_submodules
    for _pkg in ("edge_tts", "certifi"):
        try:
            edge_datas += collect_data_files(_pkg)
            edge_hiddenimports += collect_submodules(_pkg)
        except Exception:
            pass
    edge_hiddenimports += ["aiohttp", "tabulate", "srt", "typing_extensions"]
except Exception:
    pass


a = Analysis(
    [os.path.join(ROOT, "local_chat", "launcher.py")],
    pathex=[ROOT],
    datas=[
        *edge_datas,
        *mt5_datas,
        (os.path.join(PKG, "static"), os.path.join("local_chat", "static")),
        (os.path.join(PKG, "capability_docs"), os.path.join("local_chat", "capability_docs")),
        (os.path.join(PKG, "smtp_settings.example.json"), "local_chat"),
        (os.path.join(PKG, "tts_worker.py"), "local_chat"),
        (os.path.join(PKG, "stt_worker.py"), "local_chat"),
    ],
    binaries=[*mt5_binaries],
    hiddenimports=[
        *mt5_hiddenimports,
        *edge_hiddenimports,
        "local_chat.version", "local_chat.search_intent", "local_chat.media_search", "local_chat.search_engine",
        "numpy", "numpy.core", "numpy.core.multiarray", "numpy.core._multiarray_umath",
        "local_chat.mt5_data", "local_chat.mt5_trade",
        "local_chat.server", "local_chat.capabilities", "local_chat.memory",
        "local_chat.settings", "local_chat.launcher", "local_chat.files", "local_chat.models_catalog", "local_chat.search_providers", "pypdf", "webview", "webview.platforms.edgechromium", "clr", "tkinter", "tkinter.filedialog",
    ],
    excludes=["pytest", "unittest.mock", "pandas", "matplotlib"],
)
pyz = PYZ(a.pure)
# PyInstaller 6.x may attempt to copy a nested binary (notably NumPy's
# sibling numpy.libs directory) before creating its destination parent.
# Ensure the parent exists at copy time; this does not change bundle paths
# or which binaries are included.
import shutil as _shutil
_copyfile_original = _shutil.copyfile
_copystat_original = _shutil.copystat


def _long_path(p):
    """Return a Windows extended-length path (\\\\?\\...) so that paths longer
    than MAX_PATH (260 chars) can be opened, even when the OS 'LongPathsEnabled'
    policy is off. On non-Windows systems the path is returned unchanged."""
    p = os.fspath(p)
    if os.name != "nt" or p.startswith("\\\\?\\"):
        return p
    p = os.path.abspath(p)
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


def _copyfile_with_parent(src, dst, *args, **kwargs):
    """Copy with long-path support and retry on a vanished parent directory.

    Root cause of the observed COLLECT failure: the destination
    (...\\_internal\\numpy.libs\\libopenblas64__...dll) was longer than 260
    characters, so open(dst, 'wb') raised FileNotFoundError even though the
    parent directory existed. Extended-length paths fix that. The retry loop is
    kept for the (separate) case of a vanished parent directory.
    """
    _dst = _long_path(dst)
    _src = _long_path(src)
    _dst_parent = os.path.dirname(_dst)
    for _attempt in range(4):
        if _dst_parent:
            os.makedirs(_dst_parent, exist_ok=True)
        try:
            return _copyfile_original(_src, _dst, *args, **kwargs)
        except FileNotFoundError:
            if _attempt >= 3 or not _dst_parent or not os.path.isfile(_src):
                raise
            os.makedirs(_dst_parent, exist_ok=True)
    raise RuntimeError("Unreachable PyInstaller copy retry state")


def _copystat_long(src, dst, *args, **kwargs):
    return _copystat_original(_long_path(src), _long_path(dst), *args, **kwargs)


_shutil.copyfile = _copyfile_with_parent
_shutil.copystat = _copystat_long

exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="LocalChat",
    icon=os.path.join(ROOT, "packaging", "icon.ico"),
    console=False,
    disable_windowed_traceback=False,
    upx=False,
)
