"""Static check of the PyInstaller output (run by build_installer.bat after the build).

It proves the files MT5 charts need were COPIED into the app folder. It does NOT prove
that MetaTrader 5 works: that needs a Windows PC with a logged-in terminal
(see local_chat /api/diagnostics/mt5 or the in-app MT5 status).
"""
import sys
from pathlib import Path


def check(app_dir):
    app = Path(app_dir)
    internal = app / "_internal"
    base = internal if internal.is_dir() else app
    problems = []
    if not (app / "LocalChat.exe").is_file():
        problems.append("LocalChat.exe پیدا نشد.")
    mt5 = base / "MetaTrader5"
    if not mt5.is_dir():
        problems.append("پوشهٔ MetaTrader5 در خروجی نیست (پکیج بسته‌بندی نشده).")
    elif not list(mt5.rglob("*.pyd")):
        problems.append("هیچ ماژول بومی (.pyd) در MetaTrader5 نیست؛ بارگذاری DLL در برنامه شکست می‌خورد.")
    numpy_dir = base / "numpy"
    if not numpy_dir.is_dir():
        problems.append("NumPy در خروجی نیست؛ MetaTrader5 بدون آن بارگذاری نمی‌شود.")
    elif not list(numpy_dir.rglob("_multiarray_umath*.pyd")):
        problems.append("هستهٔ بومی NumPy (_multiarray_umath) در خروجی نیست.")
    libs = base / "numpy.libs"
    if not libs.is_dir() or not list(libs.glob("*.dll")):
        problems.append("DLLهای numpy.libs (مثل OpenBLAS) در خروجی نیست.")
    if not list(base.glob("python3*.dll")):
        problems.append("python3xx.dll در خروجی نیست.")
    return problems


def main(argv):
    if len(argv) != 2:
        print("usage: verify_bundle.py <app_dir>")
        return 2
    problems = check(argv[1])
    if problems:
        print("[ERROR] بررسی بستهٔ خروجی ناموفق بود:")
        for item in problems:
            print("  -", item)
        return 1
    print("Bundle check OK (files present). This does not test MT5 connectivity.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
