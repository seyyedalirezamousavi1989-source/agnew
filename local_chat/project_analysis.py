"""Read-only static analysis of a managed project + the fixed screenshot harness.

``analyze`` never imports or runs project code (it only parses it with ``ast``), so it needs no approval.
It exists because a weak local model often "finishes" a project whose modules are never wired together
(e.g. ``main.py`` opens a window but never imports the board/rules modules) and then claims success.
"""
import ast
import json
import re
import sys
from pathlib import Path

SKIP_PARTS = {".venv", ".history", "__pycache__", ".git", ".localchat", "node_modules"}
STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__", "typing_extensions"}
IMPORT_TO_PACKAGE = {"pil": "pillow", "cv2": "opencv-python", "yaml": "pyyaml", "sklearn": "scikit-learn",
                     "bs4": "beautifulsoup4", "dateutil": "python-dateutil", "dotenv": "python-dotenv",
                     "serial": "pyserial", "skimage": "scikit-image"}
MAX_FILES = 80


def _py_files(project, folder):
    base = project / folder
    if not base.is_dir() or base.is_symlink():
        return []
    out = []
    for path in sorted(base.rglob("*.py")):
        if path.is_symlink() or SKIP_PARTS & set(path.relative_to(project).parts):
            continue
        out.append(path)
        if len(out) >= MAX_FILES:
            break
    return out


def _module_name(project, path):
    relative = path.relative_to(project / "src").with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _is_stub(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        return True
    if len(body) == 1:
        item = body[0]
        if isinstance(item, ast.Pass):
            return True
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and item.value.value is Ellipsis:
            return True
        if isinstance(item, ast.Raise) and "NotImplementedError" in ast.dump(item):
            return True
    return False


def _has_main_guard(tree):
    for node in tree.body:
        if isinstance(node, ast.If) and "__name__" in ast.dump(node.test) and "__main__" in ast.dump(node.test):
            return True
    return False


def _local_imports(tree, module, is_package, modules):
    """Names of local src modules imported by this file."""
    found = set()

    def resolve(name):
        if name.startswith("src."):
            name = name[4:]
        while name:
            if name in modules:
                return name
            name = name.rpartition(".")[0]
        return None

    package = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = resolve(alias.name)
                if target:
                    found.add(target)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".") if package else []
                base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
                prefix = ".".join(base)
                full = ".".join(part for part in (prefix, node.module or "") if part)
            else:
                full = node.module or ""
            target = resolve(full) if full else None
            if target:
                found.add(target)
            for alias in node.names:  # `from src import chessboard` / `from . import chessboard`
                sub = resolve(f"{full}.{alias.name}" if full else alias.name)
                if sub:
                    found.add(sub)
    return found


def _third_party(tree, modules):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.add(node.module.split(".")[0])
    top_local = {m.split(".")[0] for m in modules} | {"src", "tests"}
    return {n for n in names if n not in STDLIB and n not in top_local}


def _requirements(project):
    path = project / "requirements.txt"
    names = set()
    if path.is_file() and not path.is_symlink():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-")):
                names.add(_norm(re.split(r"[\[<>=!~;\s]", line, maxsplit=1)[0]))
    return names


def _signatures(tree):
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            out.append(f"class {node.name}: " + ", ".join(methods[:14]))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"def {node.name}(...)")
    return out


def analyze(project):
    """Return a dict describing the structure and the obvious problems of the project."""
    project = Path(project)
    src_files = _py_files(project, "src")
    test_files = _py_files(project, "tests")
    modules = {_module_name(project, p): p for p in src_files}
    problems, warnings = [], []
    trees = {}
    for path in src_files + test_files:
        relative = path.relative_to(project).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            trees[path] = ast.parse(text, filename=relative)
        except SyntaxError as exc:
            problems.append(f"خطای نحوی در {relative} خط {exc.lineno}: {exc.msg}")
        except (OSError, ValueError) as exc:
            problems.append(f"{relative} خوانده نشد: {str(exc)[:80]}")
    report = {"files": {"src": [p.relative_to(project).as_posix() for p in src_files],
                        "tests": [p.relative_to(project).as_posix() for p in test_files]}}
    if not src_files:
        problems.append("هیچ فایل پایتون در src نیست.")

    # --- entry point and import graph
    entry = project / "src" / "main.py"
    entries = [entry] if entry in trees else [p for p in src_files if p in trees and _has_main_guard(trees[p])]
    report["entry"] = [p.relative_to(project).as_posix() for p in entries]
    if src_files and not entries:
        problems.append("هیچ فایل شروعی (src/main.py یا فایلی با `if __name__ == \"__main__\"`) پیدا نشد.")
    for path in entries:
        tree = trees[path]
        if not _has_main_guard(tree) and not any(isinstance(n, (ast.While, ast.For, ast.Expr)) and
                                                  not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))
                                                  for n in tree.body):
            problems.append(f"{path.relative_to(project).as_posix()} فقط تعریف می‌کند و هیچ‌چیز را اجرا نمی‌کند "
                            "(نبودن `if __name__ == \"__main__\": main()`)؛ با اجرای فایل هیچ اتفاقی نمی‌افتد.")
    graph = {}
    for path, tree in trees.items():
        if path in src_files:
            module = _module_name(project, path)
            graph[module] = _local_imports(tree, module, path.name == "__init__.py", set(modules))
    reachable, queue = set(), [_module_name(project, p) for p in entries]
    while queue:
        current = queue.pop()
        if current in reachable:
            continue
        reachable.add(current)
        queue.extend(graph.get(current, ()))
    # parent packages count as reachable through their __init__
    orphans = sorted(m for m in modules if m not in reachable and m and not m.endswith("__init__"))
    if entries and orphans:
        problems.append("این ماژول‌ها از فایل شروع (main.py) هیچ‌وقت وارد نمی‌شوند و در برنامه نقشی ندارند: " + ", ".join(orphans) +
                        " — آن‌ها را در main.py import و استفاده کن یا حذفشان کن.")
    report["import_graph"] = {m: sorted(t) for m, t in graph.items() if t}

    # --- stubs and TODOs
    stubs = []
    for path, tree in trees.items():
        if path not in src_files:
            continue
        relative = path.relative_to(project).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_stub(node) \
                    and not node.name.startswith("__") and not _is_abstract(node):
                stubs.append(f"{relative}:{node.lineno} {node.name}")
    if stubs:
        warnings.append("تابع‌های خالی (فقط pass/...): " + ", ".join(stubs[:8]) + (" …" if len(stubs) > 8 else ""))
    report["stubs"] = stubs[:20]

    # --- third-party imports vs requirements
    needed = set()
    for path, tree in trees.items():
        needed |= _third_party(tree, set(modules))
    required = _requirements(project)
    missing = sorted(n for n in needed if IMPORT_TO_PACKAGE.get(n.lower(), _norm(n)) not in required)
    if missing:
        problems.append("کتابخانه‌هایی import شده‌اند ولی در requirements.txt نیستند: " + ", ".join(missing))
    report["third_party"] = sorted(needed)
    report["requirements"] = sorted(required)

    # --- tests
    case_count = 0
    for path in test_files:
        tree = trees.get(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                case_count += 1
    report["tests"] = {"files": len(test_files), "cases": case_count}
    if not test_files:
        warnings.append("هیچ فایل تستی در tests نیست.")
    elif case_count == 0:
        warnings.append("در tests هیچ تابع test* پیدا نشد.")

    # --- GUI hints (pygame)
    gui = None
    uses_pygame = any("pygame" in needed_name.lower() for needed_name in needed)
    if uses_pygame:
        joined = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in src_files if p in trees)
        gui = {"library": "pygame",
               "set_mode": "display.set_mode" in joined,
               "event_loop": bool(re.search(r"pygame\.event\.(?:get|poll|wait)", joined)),
               "flip_or_update": bool(re.search(r"display\.(?:flip|update)\(", joined)),
               "mouse_handling": "MOUSEBUTTON" in joined or "mouse.get_pos" in joined,
               "window_size_literal": _window_size(trees, src_files)}
        if not gui["set_mode"]:
            problems.append("برنامهٔ pygame هیچ‌جا pygame.display.set_mode نمی‌زند؛ پنجره‌ای ساخته نمی‌شود.")
        if gui["set_mode"] and not (gui["event_loop"] and gui["flip_or_update"]):
            problems.append("حلقهٔ اصلی pygame کامل نیست (باید pygame.event.get و pygame.display.flip در یک while باشد).")
    elif any(("tkinter" in str(ast.dump(t))) for t in trees.values()):
        gui = {"library": "tkinter"}
    report["gui"] = gui
    report["problems"] = problems
    report["warnings"] = warnings
    report["healthy"] = not problems
    return report


def _is_abstract(node):
    return any("abstractmethod" in ast.dump(d) for d in node.decorator_list)


def _window_size(trees, src_files):
    constants = {}
    for path in src_files:
        tree = trees.get(path)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                constants[node.targets[0].id] = node.value.value
    for path in src_files:
        tree = trees.get(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.args and getattr(node.func, "attr", "") == "set_mode":
                first = node.args[0]
                if isinstance(first, (ast.Tuple, ast.List)) and len(first.elts) == 2:
                    values = []
                    for item in first.elts:
                        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
                            values.append(item.value)
                        elif isinstance(item, ast.Name) and item.id in constants:
                            values.append(constants[item.id])
                        else:
                            values.append(None)
                    return values
    return None


# ----------------------------------------------------------------------------- prompt snapshot
CODE_SUFFIXES = {".py", ".dart", ".kt", ".java", ".kts", ".gradle", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".vue", ".html", ".css", ".yaml", ".yml", ".json", ".properties"}
KIND_ENTRY = {"python": "src/main.py", "flutter": "lib/main.dart", "gradle": "app/src/main", "node": "index.js"}
KIND_DIRS = {"python": ("src", "tests"), "flutter": ("lib", "test"), "gradle": ("src", "app"), "node": ("src", "test", "public")}
KIND_ROOT = {"python": ("requirements.txt", "README.md"), "flutter": ("pubspec.yaml",), "gradle": ("build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"),
             "node": ("package.json", "index.js", "index.html")}


def snapshot(project, budget_chars=18000, kind="python"):
    """Text with the real current code (entry first). Files that do not fit are listed with their signatures only."""
    project = Path(project)
    files = []
    for folder in KIND_DIRS.get(kind, KIND_DIRS["python"]):
        base = project / folder
        if base.is_dir() and not base.is_symlink():
            for path in sorted(base.rglob("*")):
                if (path.is_file() and not path.is_symlink() and path.suffix.lower() in CODE_SUFFIXES
                        and not (SKIP_PARTS | {"build", ".gradle", ".dart_tool"}) & set(path.relative_to(project).parts) and len(files) < MAX_FILES):
                    files.append(path)
    for extra in KIND_ROOT.get(kind, KIND_ROOT["python"]):
        if (project / extra).is_file():
            files.append(project / extra)
    entry = project / KIND_ENTRY.get(kind, "src/main.py")
    files.sort(key=lambda p: (not str(p).startswith(str(entry)), p.parent.name in {"tests", "test"}, p.name))
    parts, left, summarized = [], budget_chars, []
    for path in files:
        relative = path.relative_to(project).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        except OSError:
            continue
        block = f"### {relative} ({len(text.splitlines())} خط)\n```\n{text.rstrip()}\n```"
        if len(block) <= left and len(text) <= 9000:
            parts.append(block)
            left -= len(block)
            continue
        sigs = ""
        if path.suffix == ".py":
            try:
                sigs = "; ".join(_signatures(ast.parse(text)))[:600]
            except SyntaxError:
                sigs = "(خطای نحوی)"
        summarized.append(f"- {relative} ({len(text)} نویسه): {sigs}")
    if summarized:
        parts.append("فایل‌هایی که کامل جا نشدند (برای دیدن محتوا read_file بزن):\n" + "\n".join(summarized))
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- other stacks
def analyze_generic(project, kind):
    """Cheap structural check for Flutter / Gradle / Node projects (the real verification is the toolchain's own analyze/test)."""
    project = Path(project)
    problems, warnings = [], []
    files = []
    for folder in KIND_DIRS.get(kind, ()):
        base = project / folder
        if base.is_dir() and not base.is_symlink():
            files += [p for p in base.rglob("*") if p.is_file() and not p.is_symlink() and p.suffix.lower() in CODE_SUFFIXES
                      and not (SKIP_PARTS | {"build", ".gradle", ".dart_tool"}) & set(p.relative_to(project).parts)]
    report = {"kind": kind, "files": {"code": [p.relative_to(project).as_posix() for p in files[:MAX_FILES]]}, "stubs": []}
    if kind == "flutter":
        if not (project / "pubspec.yaml").is_file():
            problems.append("pubspec.yaml نیست؛ toolchain_run با task=create را اجرا کن.")
        main = project / "lib" / "main.dart"
        if not main.is_file():
            problems.append("lib/main.dart نیست.")
        else:
            text = main.read_text(encoding="utf-8", errors="replace")
            if "void main" not in text:
                problems.append("lib/main.dart تابع main ندارد.")
            if "runApp" not in text:
                problems.append("lib/main.dart هیچ‌جا runApp صدا نمی‌زند.")
        dart = [p for p in files if p.suffix == ".dart" and p.relative_to(project).parts[:1] == ("lib",)]
        imported = ""
        for path in dart:
            imported += path.read_text(encoding="utf-8", errors="replace")
        orphans = [p.name for p in dart if p.name != "main.dart" and p.stem not in imported]
        if orphans:
            problems.append("این فایل‌های dart در هیچ فایل دیگری import نشده‌اند: " + ", ".join(orphans))
    elif kind == "gradle":
        if not any((project / n).is_file() for n in ("build.gradle", "build.gradle.kts")):
            problems.append("build.gradle(.kts) نیست؛ toolchain_run با task=init را اجرا کن.")
        if not any((project / n).is_file() for n in ("settings.gradle", "settings.gradle.kts")):
            warnings.append("settings.gradle(.kts) نیست.")
        if not any(p.suffix in {".java", ".kt"} for p in files):
            warnings.append("هنوز فایل منبع .java/.kt ندارد.")
    elif kind == "node":
        package = project / "package.json"
        if not package.is_file():
            problems.append("package.json نیست؛ toolchain_run با task=init را اجرا کن.")
        else:
            try:
                data = json.loads(package.read_text(encoding="utf-8"))
            except ValueError as exc:
                data = {}
                problems.append(f"package.json معتبر نیست: {exc}")
            main = data.get("main") or "index.js"
            if data and not (project / main).is_file() and not (data.get("scripts") or {}).get("start"):
                problems.append(f"فایل شروع «{main}» نیست و اسکریپت start هم تعریف نشده.")
            if data and not (data.get("scripts") or {}).get("test"):
                warnings.append("اسکریپت test در package.json نیست.")
            elif data and "no test specified" in str((data.get("scripts") or {}).get("test", "")):
                warnings.append("اسکریپت test هنوز پیش‌فرض npm است (هیچ تستی اجرا نمی‌کند).")
    report.update(problems=problems, warnings=warnings, healthy=not problems, gui=None, tests={}, entry=[])
    return report


# ----------------------------------------------------------------------------- screenshot harness
HARNESS = r'''
# LocalChat screenshot harness (generated, fixed content; not written by the model)
import json, os, runpy, sys, traceback
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
entry, outdir, spec_text = sys.argv[1], sys.argv[2], sys.argv[3]
spec = json.loads(spec_text)
report = {"frames": 0, "shots": [], "window": None, "finished": False, "exited_early": False, "error": "", "events": []}


def emit():
    print("LCRESULT:" + json.dumps(report, ensure_ascii=False))
    sys.stdout.flush()


try:
    import pygame
except Exception as exc:  # pygame missing
    report["error"] = "pygame در این venv نصب نیست: %s" % exc
    emit()
    sys.exit(0)

os.makedirs(outdir, exist_ok=True)


class _Done(BaseException):
    pass


actions = [("shot", "start")]
for index, (x, y) in enumerate(spec.get("clicks", []), 1):
    actions.append(("click", x, y))
    actions.append(("shot", "after_click_%d" % index))
for index, name in enumerate(spec.get("keys", []), 1):
    actions.append(("key", name))
    actions.append(("shot", "after_key_%d" % index))
if len(actions) == 1:
    actions.append(("shot", "later"))
state = {"i": 0, "next": int(spec.get("warmup", 8)), "up": None, "pos": (0, 0), "down": False, "prev": None}
settle = int(spec.get("settle", 4))


def _small(surface):
    return pygame.transform.smoothscale(surface, (240, 180)) if surface.get_width() > 240 else surface


def _stats(surface):
    small = pygame.transform.scale(surface, (40, 40))
    counts, dark, light = {}, 0, 0
    for yy in range(40):
        for xx in range(40):
            r, g, b = small.get_at((xx, yy))[:3]
            key = (r // 32 * 32, g // 32 * 32, b // 32 * 32)
            counts[key] = counts.get(key, 0) + 1
            lum = 0.3 * r + 0.59 * g + 0.11 * b
            dark += lum < 70
            light += lum > 190
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    return {"top_colors": ["#%02x%02x%02x:%d%%" % (c[0], c[1], c[2], round(100 * n / 1600)) for c, n in top],
            "distinct_colors": len(counts), "dark_pct": round(100 * dark / 1600), "light_pct": round(100 * light / 1600)}


def _shot(name):
    surface = pygame.display.get_surface()
    if surface is None:
        report["events"].append("%s: پنجره‌ای برای عکس گرفتن وجود نداشت" % name)
        return
    path = os.path.join(outdir, "%s.png" % name)
    pygame.image.save(surface, path)
    info = {"name": name, "file": os.path.basename(path), "size": list(surface.get_size())}
    info.update(_stats(surface))
    small = _small(surface)
    data = pygame.image.tostring(small, "RGB")
    if state["prev"] is not None and len(state["prev"]) == len(data):
        changed = sum(1 for k in range(0, len(data), 3) if data[k:k + 3] != state["prev"][k:k + 3])
        info["changed_vs_previous_pct"] = round(100.0 * changed / (len(data) // 3), 2)
    state["prev"] = data
    report["shots"].append(info)


def _post(kind, **kw):
    pygame.event.post(pygame.event.Event(kind, **kw))


def _step():
    report["frames"] += 1
    if report["frames"] < state["next"]:
        return
    if state["up"] is not None:
        x, y = state["up"]
        state["down"] = False
        _post(pygame.MOUSEBUTTONUP, pos=(x, y), button=1)
        state["up"] = None
        state["next"] = report["frames"] + settle
        return
    if state["i"] >= len(actions):
        raise _Done()
    action = actions[state["i"]]
    state["i"] += 1
    if action[0] == "shot":
        _shot(action[1])
        state["next"] = report["frames"] + 1
    elif action[0] == "click":
        x, y = action[1], action[2]
        state["pos"] = (x, y)
        state["down"] = True
        _post(pygame.MOUSEMOTION, pos=(x, y), rel=(0, 0), buttons=(0, 0, 0))
        _post(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1)
        state["up"] = (x, y)
        state["next"] = report["frames"] + 1
        report["events"].append("کلیک در (%d, %d)" % (x, y))
    elif action[0] == "key":
        key = getattr(pygame, "K_" + action[1].lower(), None)
        if key is not None:
            _post(pygame.KEYDOWN, key=key, mod=0, unicode="")
            _post(pygame.KEYUP, key=key, mod=0)
            report["events"].append("کلید " + action[1])
        state["next"] = report["frames"] + settle
    if state["i"] >= len(actions) and state["up"] is None and action[0] == "shot":
        raise _Done()


def _wrap(name):
    real = getattr(pygame.display, name)

    def wrapper(*args, **kwargs):
        result = real(*args, **kwargs)
        _step()
        return result
    setattr(pygame.display, name, wrapper)


for _name in ("flip", "update"):
    _wrap(_name)
_real_set_mode = pygame.display.set_mode


def _set_mode(*args, **kwargs):
    surface = _real_set_mode(*args, **kwargs)
    report["window"] = list(surface.get_size())
    return surface


pygame.display.set_mode = _set_mode
pygame.mouse.get_pos = lambda: state["pos"]
_real_pressed = pygame.mouse.get_pressed
pygame.mouse.get_pressed = lambda *a, **k: (1, 0, 0) if state["down"] else (0, 0, 0)

try:
    runpy.run_path(entry, run_name="__main__")
    report["exited_early"] = state["i"] < len(actions)
    report["finished"] = True
except _Done:
    report["finished"] = True
except SystemExit as exc:
    report["exited_early"] = state["i"] < len(actions)
    report["events"].append("برنامه خودش خارج شد (SystemExit %s) پس از %d فریم" % (exc.code, report["frames"]))
except BaseException:
    report["error"] = traceback.format_exc()[-1800:]
emit()
'''
