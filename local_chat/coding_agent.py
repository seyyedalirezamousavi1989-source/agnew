"""Programming-mode agent for LocalChat.

Flow of one chat turn in programming mode
-----------------------------------------
1. ``classify_request`` decides what the user wants:
   * ``discuss``      – a question about ability / a wish («می‌تونی بازی بسازی؟»): answer, ask a few
                        short questions, build NOTHING and give the model NO tools;
   * ``action``       – an explicit request (imperative / answer to our questions): plan first;
   * ``error_report`` – a pasted traceback: diagnose only, no tools;
   * ``chat``         – any other question: plain answer, no tools.
2. For ``action`` the model returns a structured decision: ``plan`` (steps), ``ask`` (≤3 clarifying
   questions) or ``answer`` (no tools needed). A plan is shown and needs the user's approval.
3. Only after approval the model gets the tools. Writing files can be auto-approved for the run
   (the user's choice on the plan card); installing, running and Docker always ask separately.

Everything the user sees is streamed as NDJSON events through ``emit``.
"""
import hmac
import json
import re
import secrets
import socket
import threading
import time
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__:
    from . import project_analysis, toolchains, toolfinder
    from . import system_tools as tools
else:
    import project_analysis
    import toolchains
    import toolfinder
    import system_tools as tools

MAX_PLAN_STEPS = 12
MAX_TOOL_CALLS = 60
APPROVAL_WAIT = 600
PLAN_DECISION_TIMEOUT = 900
MAX_PENDING_APPROVALS = 8
MAX_FAIL_STREAK = 4

READY_FOOTER = ("\n\n— هر وقت آماده بودی، جزئیات را بگو یا فقط بنویس «بساز»؛ اول یک طرح مرحله‌به‌مرحله برایت می‌نویسم "
                "و بدون تأیید تو هیچ چیزی ساخته نمی‌شود.")
CLARIFY_HEADER = "**پیش از شروع، چند نکته را بدانم تا نتیجه دقیقاً مطابق میلت شود:**"
REJECT_TEXT = ("عملیات رد شد و هیچ تغییری اعمال نشد؛ کار همین‌جا متوقف است. "
               "بگو چه چیزی را عوض کنم (مثلاً نام، مسیر یا روش دیگر) تا از همین‌جا ادامه بدهم.")
CONTEXT_MARKS = ("طرح مرحله‌به‌مرحله برایت می‌نویسم", "پیش از شروع، چند نکته را بدانم", "چه چیزی را عوض کنم",
                 "بنویس «ادامه بده»")


class ChatCancelled(Exception):
    """Raised inside a chat run when the user pressed the stop button."""


class AgentFinished(Exception):
    """Internal: the turn is complete (a final event was already emitted)."""


# ============================================================================ runs & approvals
_RUNS = {}
_RUNS_LOCK = threading.RLock()
_WAITERS = {}
_WAITERS_LOCK = threading.RLock()


def begin_run(session_id):
    run = {"cancelled": threading.Event(), "responses": []}
    with _RUNS_LOCK:
        previous = _RUNS.get(session_id)
        _RUNS[session_id] = run
    if previous is not None:
        _cancel_run_object(previous)
    return run


def end_run(session_id, run):
    with _RUNS_LOCK:
        if _RUNS.get(session_id) is run:
            _RUNS.pop(session_id, None)


def _abort_http_response(response):
    try:
        raw = getattr(getattr(response, "fp", None), "raw", None)
        sock = getattr(raw, "_sock", None)
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        response.close()
    except Exception:
        pass


def _cancel_run_object(run):
    run["cancelled"].set()
    for response in list(run["responses"]):
        _abort_http_response(response)


def cancel_run(session_id, reject=None):
    """Stop the running generation of this browser session. Returns True if one existed."""
    with _RUNS_LOCK:
        run = _RUNS.get(session_id)
    if run is None:
        return False
    _cancel_run_object(run)
    with _WAITERS_LOCK:
        mine = [(key, waiter) for key, waiter in _WAITERS.items()
                if hmac.compare_digest(waiter["session_id"], str(session_id or ""))]
    for key, waiter in mine:
        if reject:
            try:
                reject(key, session_id)
            except Exception:
                pass
        waiter["result"] = {"approved": False, "cancelled": True}
        waiter["event"].set()
    return True


def open_tracked(run, request, timeout):
    """urlopen that can be aborted by the stop button."""
    if run["cancelled"].is_set():
        raise ChatCancelled()
    response = urlopen(request, timeout=timeout)
    run["responses"].append(response)
    if run["cancelled"].is_set():
        _abort_http_response(response)
        raise ChatCancelled()
    return response


def untrack(run, response):
    try:
        run["responses"].remove(response)
    except ValueError:
        pass


def register_approval(approval_id, session_id):
    with _WAITERS_LOCK:
        if len(_WAITERS) >= MAX_PENDING_APPROVALS:
            return None
        waiter = {"event": threading.Event(), "session_id": session_id, "result": None}
        _WAITERS[approval_id] = waiter
        return waiter


def resolve_approval(approval_id, session_id, result):
    with _WAITERS_LOCK:
        waiter = _WAITERS.get(approval_id)
        if not waiter or not hmac.compare_digest(waiter["session_id"], str(session_id or "")):
            return False
        waiter["result"] = result
        waiter["event"].set()
        return True


def drop_approval(approval_id):
    with _WAITERS_LOCK:
        _WAITERS.pop(approval_id, None)


# ============================================================================ intent detection
def normalize(text):
    value = unicodedata.normalize("NFKC", text or "").casefold()
    value = value.replace("‌", " ").replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"\s+", " ", value).strip()


_STRONG_FA = re.compile(
    r"(?<!\w)(?:بساز(?:ش|شو|یدش)?|بسازید|درست\s?کن(?:ش|شو)?|درست\s?کنید|ایجاد\s?کن(?:ش|ید)?|تولید\s?کن(?:ید)?|"
    r"بنویس(?:ش|شو)?|بنویسید|پیاده\s?سازی\s?کن(?:ید)?|اجرا\s?کن(?:ش|شو)?|اجرا\s?بگیر|تست\s?کن(?:ش|ید)?|"
    r"اصلاح\s?کن(?:ش|شو|ید)?|رفع\s?کن(?:ش|ید)?|رفعش\s?کن|ریفکتور\s?کن|ویرایش\s?کن(?:ش|ید)?|تغییر\s?بده(?:ش)?|"
    r"اضافه\s?کن(?:ش|ید)?|نصب\s?کن(?:ش|ید)?|شروع\s?کن(?:ش|ید)?|انجام\s?بده(?:ش|ید)?|انجامش\s?بده|"
    r"کامل\s?کن(?:ش|ید)?|راه\s?اندازی\s?کن|کد\s?بزن|کدش\s?رو\s?بنویس|بزن\s?بریم|برو\s?شروع|بسم\s?الله)(?!\w)")
_WEAK_FA = re.compile(r"(?<!\w)[\w]{2,}\s?(?:کن(?:ش|شو|ید)?|بده|بزار|بذار|عوض\s?کن|بیار)(?!\w)")
_CONTINUE = re.compile(r"(?<!\w)(?:ادامه\s?بده|ادامه\s?ش\s?بده|continue)(?!\w)")
_EXPLAIN_FA = re.compile(r"(?<!\w)(?:توضیح|شرح|بگو|بیان|معرفی|مقایسه|ترجمه|خلاصه|تحلیل|بررسی|مثال\s?بزن|بحث)(?!\w)")
_STRONG_EN = re.compile(
    r"^(?:(?:ok|okay|yes|yeah|sure|please|now|then|and|just|go ahead and)[,.! ]+)*"
    r"(?:create|build|make|write|generate|implement|fix|debug|run|test|install|add|edit|update|refactor|start|"
    r"continue|proceed|go ahead|do it|set up|develop|code)\b")
_QUESTION_EN = re.compile(r"^(?:how|what|why|when|where|which|who|can|could|would|should|is|are|do|does|did|will)\b")
_CAPABILITY_FA = re.compile(
    r"(?<!\w)(?:می\s?تونی|می\s?توانی|بلدی|قادری|توانایی|از\s?پس\s+\w+\s+برمی\s?آیی|ازت\s+برمی\s?آد|امکانش\s+هست|"
    r"امکان\s+دار[هد]|می\s?شه|می\s?شود|میتونی)(?!\w)")
_CAPABILITY_EN = re.compile(r"\b(?:can you|could you|are you able|do you know how|would you be able|is it possible)\b")
_WISH = re.compile(
    r"(?<!\w)(?:می\s?خواه?م|می\s?خوام|دوست\s+دارم|قصد\s+دارم|دنبال\s+(?:یه|یک)|نیاز\s+دارم)(?!\w)|"
    r"\b(?:i want|i'd like|i would like|i need|looking for)\b")
_BUILD_TOPIC = re.compile(
    r"(?<!\w)(?:بازی|برنامه|اسکریپت|سایت|وب\s?سایت|اپ|اپلیکیشن|ابزار|پروژه|کد|ربات|بات|سرور|نرم\s?افزار|"
    r"ماشین\s?حساب|api|game|app|script|website|tool|project|bot|program|code|calculator|software)(?!\w)")
_QUESTION_FORM_FA = re.compile(r"(?<!\w)(?:چرا|چطور|چگونه|چی|چیه|چیست|کدوم|کجا|کی|آیا|چند)(?!\w)")
_FEEDBACK_FA = re.compile(
    r"(?<!\w)(?:نیست|نشد|نکرد|نمی\s?(?:شه|شود|کنه|کند|ره|رود|یاد|آید|بینم|بینه|تونه|توانه|کنم)|هنوز|باز\s?هم|بازهم|"
    r"هیچ\s+تغییری|تغییری\s+نکرد|باید|بزرگ(?:تر)?|کوچ(?:ک|یک)(?:تر)?|ضعیف|زشت|خراب|اشتباه|مشکل|ایراد|بهتر|عوض|تغییر|"
    r"اصلاح|درست\s?(?:نیست|نشد)|حذف|اضافه|کم\s?کن|زیاد\s?کن|پررنگ|روشن|تیره|گرافیکی|سه\s?بعدی|۳\s?بعدی|رنگ|اندازه)(?!\w)")
_NEW_PROJECT_FA = re.compile(r"(?<!\w)(?:جدید|دیگه|دیگر|از\s?اول|از\s?صفر|یه\s+\w+\s+دیگه|یک\s+\w+\s+دیگر)(?!\w)")
_BUILD_FROM_SCRATCH = re.compile(r"(?<!\w)(?:بساز(?:ش|شو|ید)?|ایجاد\s?کن|درست\s?کن\s+(?:یه|یک))(?!\w)")
_ERROR_MARKERS = (
    r"traceback \(most recent call last\)", r"file \"[^\"]+\", line \d+", r"\b(?:[A-Za-z_][\w.]*Error|Exception)\b\s*:",
    r"\b(?:ModuleNotFoundError|ImportError|SyntaxError|TypeError|ValueError|AttributeError|NameError|KeyError|OperationalError|IntegrityError)\b",
    r"\b(?:npm ERR!|Internal Server Error|command not found|stack trace|fatal error)\b",
    r"(?:ارور|اررور|خطا\s?(?:می\s?ده|میده|داد|دارم)|اجرا\s?نمی\s?شه|کار\s?نمی\s?کن(?:ه|د)|استارت\s?نمی\s?شه|ناموفق)",
)


def looks_like_error_report(text):
    if not isinstance(text, str) or not text.strip():
        return False
    return any(re.search(marker, text, re.IGNORECASE) for marker in _ERROR_MARKERS)


def _has_context_mark(history):
    for message in reversed(history or []):
        if message.get("role") == "assistant":
            content = str(message.get("content") or "")
            return any(mark in content for mark in CONTEXT_MARKS)
    return False


def _has_project_activity(history):
    return any(re.search(r"^> (?:✅|❌|🔧)", str(m.get("content") or ""), re.M)
               for m in (history or []) if m.get("role") == "assistant")


def classify_request(text, history=(), has_projects=False):
    """Return one of: action, discuss, error_report, chat."""
    t = normalize(text)
    if not t:
        return "chat"
    strong = bool(_STRONG_FA.search(t) or _STRONG_EN.search(t))
    questionish = bool(re.search(r"[؟?]", t) or _QUESTION_FORM_FA.search(t) or _QUESTION_EN.match(t))
    errors = looks_like_error_report(text)
    if _has_context_mark(history):
        # we just asked questions / invited the user to start: any non-question reply continues the job
        if strong or _CONTINUE.search(t):
            return "action"
        if errors:
            return "error_report"
        return "chat" if questionish and not _STRONG_FA.search(t) else "action"
    if errors and not strong:
        return "error_report"
    if strong:
        return "action"
    if _CONTINUE.search(t):
        return "chat"  # «ادامه بده» after a cut-off plain answer just continues that answer
    if (has_projects or _has_project_activity(history)) and _FEEDBACK_FA.search(t) and not questionish \
            and not re.search(r"```", text):
        return "action"  # a complaint / change request about what was already built
    explain = bool(_EXPLAIN_FA.search(t))
    capability = bool(_CAPABILITY_FA.search(t) or _CAPABILITY_EN.search(t) or _WISH.search(t))
    if capability and _BUILD_TOPIC.search(t) and len(t) <= 220 and "```" not in text:
        return "discuss"
    if explain or "```" in text:
        return "chat"
    if (has_projects or _has_project_activity(history)) and _WEAK_FA.search(t) and not questionish:
        return "action"
    if _BUILD_TOPIC.search(t) and not questionish and len(t) <= 160:
        return "discuss"  # a bare «یک بازی مار» is more likely a wish than an order
    return "chat"


# ============================================================================ requirement interview
_SKIP_INTERVIEW = re.compile(r"(?<!\w)(?:بدون\s+(?:سوال|سؤال)|سوال\s+نپرس|سؤال\s+نپرس|همین\s+الان|پیش\s?فرض|بدون\s+پرسش|ask\s+no)(?!\w)")
_KIND_GAME = re.compile(r"(?<!\w)(?:بازی|game)(?!\w)")
_KIND_WEB = re.compile(r"(?<!\w)(?:سایت|وب\s?سایت|website|landing)(?!\w)")
_KIND_APP = re.compile(r"(?<!\w)(?:برنامه|اپ|اپلیکیشن|نرم\s?افزار|ابزار|ماشین\s?حساب|ربات|بات|اسکریپت|app|tool|script|bot|program|calculator|software)(?!\w)")
_QUESTIONS = {
    "game": [
        (r"گرافیک|رنگ|سیاه|سفید|بعدی|ظاهر|تم|استایل|طرح|style|theme",
         "ظاهر و رنگ‌ها؟ (پیش‌فرض: دوبعدی گرافیکی با pygame، همهٔ شکل‌ها با کد کشیده می‌شود، سایه و برجستگی برای حس سه‌بعدی؛ رنگ‌بندی دلخواهت را بگو)"),
        (r"اندازه|بزرگ|کوچک|پنجره|رزولوشن|تمام\s?صفحه|size|window",
         "اندازهٔ پنجره؟ (پیش‌فرض: ۱۰۰۰×۸۰۰ و قابل تغییر اندازه، همهٔ اجزا متناسب با آن)"),
        (r"دو\s?نفره|یک\s?نفره|تک\s?نفره|کامپیوتر|هوش|\bai\b|حریف|کیبورد|موس|کنترل|player",
         "حالت بازی و کنترل؟ (پیش‌فرض: دونفره روی یک رایانه با موس؛ اگر حریف کامپیوتری هم می‌خواهی بگو)"),
        (r"منو|امتیاز|صدا|مرحله|سطح|ذخیره|menu|score|sound|level",
         "امکانات اضافه؟ (پیش‌فرض: منوی شروع، نمایش نوبت/امتیاز، شروع دوباره، پیام پایان بازی؛ بدون صدا)"),
    ],
    "app": [
        (r"tkinter|کنسول|ترمینال|پنجره|رابط|گرافیک|وب|html|ui|gui|console",
         "نوع رابط؟ (پیش‌فرض: پنجرهٔ گرافیکی ساده با tkinter؛ یا کنسول، یا pygame اگر ظاهر ویژه می‌خواهی)"),
        (r"قابلیت|امکانات|ویژگی|feature",
         "مهم‌ترین قابلیت‌هایی که باید داشته باشد چیست؟ (پیش‌فرض: ساده‌ترین نسخهٔ کاملاً کاربردی؛ ویژگی‌های دلخواهت را فهرست کن)"),
        (r"ذخیره|فایل|دیتابیس|json|csv|database|save",
         "ورودی/خروجی و ذخیره‌سازی؟ (پیش‌فرض: ذخیره در یک فایل JSON کنار برنامه، فقط اگر لازم باشد)"),
        (r"ظاهر|رنگ|تم|فارسی|راست\s?چین|style|theme",
         "ظاهر و زبان؟ (پیش‌فرض: تمیز و مینیمال، متن فارسی با راست‌به‌چپ)"),
    ],
    "web": [
        (r"صفحه|page|بخش|section",
         "چند صفحه یا بخش و با چه محتوایی؟ (پیش‌فرض: یک صفحهٔ اصلی با سربرگ، معرفی، چند بخش و تماس)"),
        (r"رنگ|تم|ظاهر|طرح|سبک|style|theme|color",
         "رنگ و سبک ظاهری؟ (پیش‌فرض: مدرن و روشن با یک رنگ اصلی؛ رنگ دلخواهت را بگو)"),
        (r"فارسی|راست\s?چین|rtl|انگلیسی|زبان",
         "زبان و جهت متن؟ (پیش‌فرض: فارسی و راست‌به‌چپ)"),
    ],
}


def interview_questions(text):
    """Deterministic requirement questions for a short «make me X» request. [] when nothing useful is missing."""
    t = normalize(text)
    if not t or len(t) >= 170 or _SKIP_INTERVIEW.search(t):
        return []
    if _KIND_GAME.search(t):
        kind = "game"
    elif _KIND_WEB.search(t):
        kind = "web"
    elif _KIND_APP.search(t):
        kind = "app"
    else:
        return []
    asked = [question for pattern, question in _QUESTIONS[kind] if not re.search(pattern, t)]
    return asked[:4]


def is_modification(text, history, active):
    """Is this message about the project that already exists (feedback / change) rather than a new build?"""
    if not active:
        return False
    t = normalize(text)
    if _NEW_PROJECT_FA.search(t):
        return False
    if _BUILD_FROM_SCRATCH.search(t) and not _FEEDBACK_FA.search(t):
        return False
    return bool(_has_project_activity(history) or _FEEDBACK_FA.search(t) or re.search(r"(?<!\w)(?:اصلاح|رفع|ویرایش|تغییر|بهتر|ادامه)", t))


_ACTIVITY_PROJECT = re.compile(r"(?:پروژه:\s*`([a-z0-9][a-z0-9_-]{0,39})`|پروژهٔ\s*«([a-z0-9][a-z0-9_-]{0,39})»)")


def active_project_from_history(history):
    for message in reversed(history or []):
        if message.get("role") != "assistant":
            continue
        found = _ACTIVITY_PROJECT.findall(str(message.get("content") or ""))
        if found:
            slug = (found[-1][0] or found[-1][1])
            try:
                tools._existing_project(slug)
                return slug
            except tools.SystemToolError:
                continue
    return ""


_PATH_RE = re.compile(r"(?:src|tests|data|assets|docs)/[\w./-]+|run\.bat|requirements\.txt|README\.md")


def compact_history(history, per_message=2500):
    """Old assistant turns carry long tool logs; the model would imitate them (and hallucinate results).
    Keep the human-readable text and replace the logs by one honest summary line."""
    out = []
    for message in history:
        if message.get("role") != "assistant":
            out.append({**message, "content": str(message.get("content") or "")[:12000]})
            continue
        text = str(message.get("content") or "")
        touched = []
        kept = []
        in_code = False
        for line in text.splitlines():
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            if line.startswith(">"):
                if "✅" in line:
                    touched += _PATH_RE.findall(line)
                continue
            if re.match(r"^\s*مرحلهٔ\s*\d+\s*از\s*\d+", line):
                continue
            kept.append(line)
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
        note = ""
        if touched:
            unique = list(dict.fromkeys(touched))
            note = "[سابقهٔ ابزارها: فایل‌های ساخته/تغییرداده‌شده در آن نوبت: " + ", ".join(unique[:20]) + "]\n"
        out.append({**message, "content": (note + body)[:per_message]})
    return out


def detect_loop(text):
    """Index at which a degenerate repetition starts to repeat (to cut there), or None."""
    if len(text) < 700:
        return None
    tail = text[-140:]
    if tail.strip() and text.count(tail) >= 3:
        return text.find(tail) + len(tail)
    lines = [line.strip() for line in text.splitlines()[-80:] if len(line.strip()) >= 45]
    for line in set(lines):
        if lines.count(line) >= 4:
            first = text.find(line)
            return first + len(line)
    return None


GAME_GUIDE = (
    "راهنمای کیفیت بازی pygame: پنجره حداقل ۹۰۰×۷۰۰ (یا اندازهٔ خواستهٔ کاربر) و قابل تغییر اندازه، همهٔ اندازه‌ها از اندازهٔ پنجره محاسبه شود؛ "
    "همهٔ اجزا و مهره‌ها با کد رسم شوند و حتماً دیده شوند (برای حس سه‌بعدی: گرادیان، سایه، برجستگی، لبهٔ روشن)؛ "
    "خانه‌های مجاز/انتخاب‌شده هایلایت شوند؛ نوبت یا امتیاز و پیام پایان بازی نمایش داده شود؛ کلید R شروع دوباره باشد؛ "
    "main.py باید همهٔ ماژول‌ها را import و استفاده کند و با `if __name__ == \"__main__\": main()` پایان یابد؛ "
    "فونت را با pygame.font.SysFont و پشتیبانی از متن فارسی پرهیز کن (متن‌های روی صفحه انگلیسی/عدد باشند مگر کاربر خواست)."
)


# ============================================================================ tool definitions
def _tool(name, description, required, properties):
    """Compact tool schema. `purpose`/`step` are filled in by the agent itself, so the model never sees them."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "required": list(required), "properties": properties}}}


_P = {"type": "string"}
_PATH = {"type": "string", "description": "relative, e.g. src/main.py, lib/main.dart"}
_ENTRY = {"type": "string", "description": "default src/main.py"}
_STR = {"type": "string"}
_LIST = {"type": "array", "items": {"type": "string"}}
TOOL_DEFINITIONS = [
    _tool("create_project", "New project folder. kind: python (default), flutter, gradle or node.",
          ["name"], {"name": {"type": "string", "description": "lowercase slug"},
                     "kind": {"type": "string", "enum": ["python", "flutter", "gradle", "node"]}}),
    _tool("create_venv", "Create .venv in a python project.", ["project"], {"project": _P}),
    _tool("write_file", "Create ONE new text file (never overwrites), complete content, no placeholders.",
          ["project", "path", "content"], {"project": _P, "path": _PATH, "content": _STR}),
    _tool("edit_file", "Replace one unique snippet of an existing file (old_text must match exactly once).",
          ["project", "path", "old_text", "new_text"], {"project": _P, "path": _PATH, "old_text": _STR, "new_text": _STR}),
    _tool("update_file", "Replace the whole content of an existing file.",
          ["project", "path", "content"], {"project": _P, "path": _PATH, "content": _STR}),
    _tool("read_file", "Read a project file.", ["project", "path"], {"project": _P, "path": _PATH}),
    _tool("list_files", "List project files.", ["project"], {"project": _P}),
    _tool("check_environment", "Python/Docker/toolchain status.", [], {"project": _P}),
    _tool("create_launcher", "Create run.bat (python projects).", ["project"], {"project": _P, "entry": _ENTRY}),
    _tool("venv_install", "pip install wheels into the venv and update requirements.txt.",
          ["project", "packages"], {"project": _P, "packages": _LIST}),
    _tool("venv_run", "Run a console script (max 120 s). Never for GUI programs.",
          ["project", "path"], {"project": _P, "path": _PATH, "args": _LIST, "timeout": {"type": "integer"}}),
    _tool("venv_test", "Run the python tests headless.", ["project"], {"project": _P}),
    _tool("check_project", "Static check of the whole project (syntax, unused modules, stubs, missing deps). Fix every problem.",
          ["project"], {"project": _P}),
    _tool("list_projects", "List all projects.", [], {}),
    _tool("screenshot_check", "Run a pygame program headless, simulate clicks/keys, return real evidence (size, frames, change per click).",
          ["project"], {"project": _P, "entry": _ENTRY,
                        "clicks": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                        "keys": _LIST}),
    _tool("toolchain_run", "Run a Flutter/Gradle/Node task. flutter: " + toolchains.TASK_HELP["flutter"] + ". gradle: "
          + toolchains.TASK_HELP["gradle"] + ". node: " + toolchains.TASK_HELP["node"],
          ["project", "task"], {"project": _P, "task": _STR, "packages": _LIST, "platforms": _LIST,
                                "type": _STR, "dsl": _STR}),
    _tool("editor_setup", "Write VS Code workspace files (.vscode: extensions, settings, launch, tasks) for the project.",
          ["project"], {"project": _P}),
    _tool("editor_extension", "Install a recommended VS Code extension (allowlist).", ["extension"], {"extension": _STR}),
    _tool("open_in_editor", "Open the project (or file:line) in VS Code.", ["project"],
          {"project": _P, "path": _PATH, "line": {"type": "integer"}}),
    _tool("docker_test", "Run python tests in isolated Docker containers.", ["project"], {"project": _P}),
    _tool("docker_run", "Run a script in isolated Docker (60 s, no network).", ["project", "path"], {"project": _P, "path": _PATH}),
]
TOOL_NAMES = {item["function"]["name"] for item in TOOL_DEFINITIONS}
TOOL_LABELS = {**tools.ACTION_LABELS, "none": "بدون ابزار (در گفت‌وگو)"}
CORE_TOOLS = {"read_file", "list_files", "write_file", "edit_file", "update_file", "check_project"}
KIND_TOOLS = {
    "python": {"create_venv", "venv_install", "venv_test"},
    "flutter": {"toolchain_run"}, "gradle": {"toolchain_run"}, "node": {"toolchain_run"},
}


def select_tools(kind="python", step_tools=(), game=False):
    """Offer only the tools this task needs: fewer definitions = more room for code in the model context."""
    names = set(CORE_TOOLS) | KIND_TOOLS.get(kind, set()) | {t for t in step_tools if t in TOOL_NAMES}
    if kind == "python" and game:
        names.add("screenshot_check")
    return [item for item in TOOL_DEFINITIONS if item["function"]["name"] in names]
_ALIASES = {
    "project": ("project_name", "project_slug", "name", "project_id"),
    "path": ("file_path", "filepath", "filename", "file", "file_name"),
    "content": ("code", "text", "file_content", "contents", "new_content"),
    "old_text": ("old", "old_string", "search", "find", "original"),
    "new_text": ("new", "new_string", "replace", "replacement"),
    "packages": ("package", "libraries", "libs", "requirements", "deps", "dependencies"),
    "task": ("command", "action", "cmd"),
    "extension": ("extension_id", "ext", "id"),
}
_FILE_TOOLS = {"write_file", "edit_file", "update_file", "read_file", "venv_run", "docker_run"}
_PY_PATH_TOOLS = _FILE_TOOLS


def _non_python_hint(args):
    return False


def normalize_arguments(name, arguments, default_project=""):
    """Tolerate the small naming slips of local models (file_path vs path, name vs project …)."""
    args = dict(arguments) if isinstance(arguments, dict) else {}
    if name == "create_project":
        if str(args.get("kind", "")).lower() in {"python", "flutter", "gradle", "node"}:
            args["kind"] = str(args["kind"]).lower()
        else:
            args.pop("kind", None)
        if not args.get("name"):
            for key in ("project", "project_name", "project_slug", "slug"):
                if args.get(key):
                    args["name"] = args[key]
                    break
        if not args.get("name") and default_project:
            args["name"] = default_project
        return args
    for canonical, aliases in _ALIASES.items():
        if canonical == "project" and name in {"create_project"}:
            continue
        if args.get(canonical) in (None, ""):
            for alias in aliases:
                if args.get(alias) not in (None, ""):
                    args[canonical] = args[alias]
                    break
    if name in tools.READ_ONLY_ACTIONS | tools.SAFE_WRITE_ACTIONS | tools.EXEC_ACTIONS and not args.get("project") and default_project \
            and name != "check_environment":
        args["project"] = default_project
    if name in {"venv_install", "toolchain_run"} and isinstance(args.get("packages"), str):
        args["packages"] = [part for part in re.split(r"[\s,]+", args["packages"]) if part]
    path = args.get("path")
    if name in _FILE_TOOLS and isinstance(path, str):
        path = path.strip().replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        if "/" not in path and path.endswith(".py") and not _non_python_hint(args):
            path = ("tests/" if path.startswith("test_") else "src/") + path
        args["path"] = path
    return args


# ============================================================================ planner
PLANNER_SYSTEM_PROMPT = (
    "You plan for a local coding agent; the user speaks Persian, write all user-facing strings in Persian.\n"
    "decision: answer (question/explanation only, steps=[]) | ask (only if one detail is truly missing; max 3 short questions about what the user sees or does, "
    "each with a default in parentheses; never about money/shops/deadlines) {ask_rule}| plan (max 12 steps: title, one-sentence detail, one tool).\n"
    "Rules:\n"
    "1 spec: 4-8 short concrete requirements from the whole conversation (size, look, controls, features, every requested change); no invented features.\n"
    "2 Choose `kind` yourself (and give `stack_reason`, short): python = scripts, CLI, data, 2D/desktop games (pygame), tests; "
    "flutter = mobile/cross-platform apps with a UI; gradle = Java/Kotlin/Android-style JVM apps; node = web/JS/TS apps or tools. "
    "Use only toolchains marked available below; if the ideal one is missing, pick the best available and say so in stack_reason.\n"
    "3 NEW python: create_project, (create_venv, venv_install if packages), one write_file per file (logic modules independent of the GUI, src/main.py last and it uses every module), "
    "tests/test_*.py, README.md, create_launcher for GUIs. NEW flutter/gradle/node: create_project(kind), toolchain_run (create/init), then write_file/update_file for the sources, "
    "toolchain_run for pub_add/add packages, analyze/test/build.\n"
    "4 EXISTING project (complaint, fix, change): reuse the slug and kind from the disk state, never create_project again; read_file the relevant files, then edit_file/update_file every file that must change.\n"
    "5 Verify: check_project; python: venv_test and, for pygame, screenshot_check; flutter: toolchain_run analyze+test; gradle: build+test; node: test. "
    "Never venv_run a GUI. editor_setup (VS Code workspace) when VS Code is available and the project has several files; open_in_editor only if the user asked to open it.\n"
    "Available toolchains: {toolchains}\n"
    "Fields: decision,title,summary,project(slug),kind,stack_reason,spec[],steps[{{title,detail,tool}}],questions[].\n"
    "Disk state (real):\n{projects}\nOutput ONLY JSON matching the schema."
)
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["plan", "answer", "ask"]},
        "title": {"type": "string"}, "summary": {"type": "string"}, "project": {"type": "string"},
        "kind": {"type": "string", "enum": ["python", "flutter", "gradle", "node"]},
        "stack_reason": {"type": "string"},
        "questions": {"type": "array", "items": {"type": "string"}},
        "spec": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"}, "detail": {"type": "string"},
            "tool": {"type": "string", "enum": sorted(TOOL_LABELS)}}, "required": ["title", "tool"]}},
    },
    "required": ["decision"],
}
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


def available_toolchains():
    """Fast presence check (no version probing) of the stacks the planner may choose."""
    found = {tool: bool(toolfinder.find(tool)) for tool in ("flutter", "gradle", "java", "node", "npm", "code", "docker")}
    def yes(*names):
        return all(found[n] for n in names)
    return (f"python=yes; flutter={'yes' if yes('flutter') else 'NO'}; gradle={'yes' if yes('gradle') else 'NO'}"
            f"(java {'yes' if found['java'] else 'NO'}); node={'yes' if yes('node', 'npm') else 'NO'}; "
            f"vscode={'yes' if found['code'] else 'NO'}; docker={'yes' if found['docker'] else 'NO'}")


def parse_plan(text):
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    steps = []
    for item in (data.get("steps") or [])[:MAX_PLAN_STEPS]:
        if not isinstance(item, dict) or not str(item.get("title", "")).strip():
            continue
        tool = item.get("tool") if item.get("tool") in TOOL_LABELS else "none"
        steps.append({"n": len(steps) + 1, "title": str(item["title"]).strip()[:120],
                      "detail": str(item.get("detail", "")).strip()[:300], "tool": tool,
                      "label": TOOL_LABELS[tool],
                      "needs_approval": tool in tools.EXEC_ACTIONS,
                      "auto_ok": tool in tools.SAFE_WRITE_ACTIONS})
    questions = [str(q).strip()[:220] for q in (data.get("questions") or []) if str(q).strip()][:3]
    spec = [str(q).strip()[:240] for q in (data.get("spec") or []) if str(q).strip()][:10]
    decision = data.get("decision")
    if decision not in {"plan", "answer", "ask"}:
        decision = "plan" if steps else "answer"
    if decision == "plan" and not steps:
        decision = "ask" if questions else "answer"
    if decision == "ask" and not questions:
        decision = "plan" if steps else "answer"
    project = str(data.get("project") or "").strip().lower()
    project = re.sub(r"[^a-z0-9_-]+", "-", project).strip("-_")[:40]
    return {"decision": decision, "title": str(data.get("title", "")).strip()[:120] or "طرح کار",
            "summary": str(data.get("summary", "")).strip()[:400],
            "project": project if _SLUG.fullmatch(project) else "", "steps": steps, "questions": questions, "spec": spec,
            "kind": data.get("kind") if data.get("kind") in {"python", "flutter", "gradle", "node"} else "",
            "stack_reason": str(data.get("stack_reason", "")).strip()[:200]}


def plan_markdown(plan):
    lines = [f"**طرح کار: {plan['title']}**"]
    if plan["summary"]:
        lines.append(plan["summary"])
    if plan["project"]:
        stack = f" — بستر: {plan['kind']}" + (f" ({plan['stack_reason']})" if plan.get("stack_reason") else "") if plan.get("kind") else ""
        lines.append(f"پروژه: `{plan['project']}`{stack}")
    if plan.get("spec"):
        lines.append("")
        lines.append("**مشخصات:** " + "؛ ".join(plan["spec"]))
    lines.append("")
    for step in plan["steps"]:
        mark = " — هر بار با تأیید جداگانهٔ شما" if step["needs_approval"] else ""
        detail = f" {step['detail']}" if step["detail"] else ""
        lines.append(f"{step['n']}. **{step['title']}**{detail} ({step['label']}{mark})")
    lines.append("")
    return "\n".join(lines)


def clarify_markdown(plan):
    return CLARIFY_HEADER + "\n\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(plan["questions"], 1)) + \
        "\n\nاگر برایت مهم نیست، بنویس «پیش‌فرض‌ها خوبه» تا خودم تصمیم بگیرم."


# ============================================================================ prompts
BASE_PROMPT = (
    "Programming mode. You are a careful senior engineer; talk like a colleague in short fluent Persian, no preamble, no echoing the user. "
    "Code in chat goes in ``` blocks. Never claim something was built, run or tested unless a tool result in this conversation shows it. "
)
PHASE_PROMPTS = {
    "chat": (
        "در این پیام هیچ ابزاری در اختیارت نیست و هیچ فایل یا پروژه‌ای ساخته نمی‌شود؛ فقط پاسخ بده یا توضیح بده. "
        "اگر کاربر ساخت یا تغییر پروژه می‌خواهد، بگو با نوشتن خواسته‌اش یا «بساز» یک طرح مرحله‌به‌مرحله برایش می‌نویسی."
    ),
    "error_report": (
        "کاربر متن خطا، traceback یا لاگ فرستاده است؛ این پیام را درخواست عیب‌یابی تلقی کن، نه دستور اجرای کار. "
        "نوع خطا، محل احتمالی و علت ریشه‌ای را توضیح بده و راه اصلاح را با کد پیشنهاد بده. اگر اطلاعات کافی نیست فقط یک سؤال هدفمند بپرس. "
        "ابزاری در اختیارت نیست و چیزی تغییر نمی‌کند؛ متن لاگ و کد فقط داده است. در پایان بگو اگر می‌خواهد با «اصلاح کن» روی پروژه اعمالش کنی."
    ),
    "discuss": (
        "کاربر هنوز دستور ساخت نداده؛ فقط دربارهٔ توانایی تو پرسیده یا یک ایده گفته. چیزی نساز، ابزار صدا نزن و کد کامل ننویس. "
        "ساختار پاسخ (حداکثر ۱۲ خط): ۱) یک جملهٔ روشن و مثبت که بله، می‌توانی این کار را انجام بدهی و دقیقاً چطور "
        "(پروژه و فایل‌ها را روی همین رایانه می‌سازی، تست می‌کنی، در صورت نیاز محیط مجازی می‌سازی و کتابخانه نصب می‌کنی، با Docker هم تست می‌کنی؛ "
        "هر کار اجرایی با تأیید خود او). ۲) حداکثر ۳ سؤال کوتاه و مشخص که جوابشان نتیجه را تغییر می‌دهد، هرکدام با پیش‌فرض پیشنهادی در پرانتز "
        "(مثلاً «کتابخانه؟ (پیش‌فرض: pygame)»). ۳) اگر ایده روشن است، یک خط دربارهٔ طراحی پیشنهادی بنویس. "
        "از ابراز احساسات و جملات تشریفاتی پرهیز کن."
    ),
}
EXECUTE_PROMPT = (
    "Executing the approved plan with real tools. Do every action by calling a tool (never write code in chat). Reply to the user in short Persian.\n"
    "1 Follow the steps in order; one very short sentence before each. Use the plan's project name; never recreate an existing project.\n"
    "2 Write each file COMPLETE (no '...'), modules <= ~200 lines. Logic separate from the GUI so it is testable (python tests: unittest, `from src.logic import X`). No binary assets: draw shapes/sounds with code.\n"
    "3 'Current code' and 'Disk state' below are live: copy names/imports exactly from them. Change existing files with read_file then edit_file (small) or update_file (rewrite); the change must really implement what the user asked.\n"
    "4 After writing: check_project and fix every problem; then tests (venv_test / toolchain_run analyze,test,build / docker_test). "
    "pygame: screenshot_check with real click coordinates; blank/small picture or click without effect = fix and recheck (max 2 rounds). Failure: one-sentence cause, fix (max 3 tries).\n"
    "5 GUI programs: create_launcher, never venv_run. No manual shell/docker commands. Do not repeat a failed call unchanged.\n"
    "6 If VS Code is available and the plan has editor_setup, run it; to point at a problem use open_in_editor with path and line.\n"
    "7 Finish with an honest short summary: what was built/changed, files, check/test/screenshot results exactly as the tools said, what is NOT verified, run command and folder, one next suggestion. Never repeat a paragraph."
)


def build_system_prompt(base, phase, project_overview="", plan_text="", project="", spec="", code_snapshot="", game=False):
    text = base.rstrip() + "\n\n" + BASE_PROMPT
    if phase == "execute":
        text += EXECUTE_PROMPT
        if project:
            text += f"\n\nنام پروژه: {project}"
        if plan_text:
            text += "\n\nطرح تأییدشده:\n" + plan_text
        if spec:
            text += "\n\nمشخصات الزامی (دقیقاً رعایت کن):\n" + spec
        if game:
            text += "\n\n" + GAME_GUIDE
    else:
        text += PHASE_PROMPTS.get(phase, PHASE_PROMPTS["chat"])
    if project_overview:
        text += ("\n\nوضعیت واقعی دیسک (همیشه به‌روز؛ این منبع حقیقت است، نه حافظهٔ تو از گفت‌وگو؛ "
                 "اگر کاربر دربارهٔ «پروژه»، «بازی» یا «برنامه‌ای که ساختی» حرف می‌زند منظورش پروژهٔ فعال همین‌جاست):\n" + project_overview)
    if code_snapshot:
        text += "\n\nکد فعلی پروژهٔ فعال (واقعی، همین حالا روی دیسک):\n" + code_snapshot
    return text


# ============================================================================ streaming helpers
_TEXT_TOOL_MARKERS = ("<tool_call", "<function=")


class _Forwarder:
    """Forward assistant text to the user live, but never leak raw text-format tool calls."""

    def __init__(self, emit):
        self.emit = emit
        self.text = ""
        self.sent = 0
        self.blocked = False

    def _send(self, piece):
        if piece:
            self.emit({"message": {"role": "assistant", "content": piece}, "done": False})

    def add(self, piece):
        self.text += piece
        if self.blocked:
            return
        cuts = [i for i in (self.text.find(m) for m in _TEXT_TOOL_MARKERS) if i >= 0]
        if cuts:
            self.blocked = True
            self._send(self.text[self.sent:min(cuts)])
            self.sent = len(self.text)
            return
        hold = 0
        for marker in _TEXT_TOOL_MARKERS:
            for k in range(min(len(marker) - 1, len(self.text) - self.sent), 0, -1):
                if self.text.endswith(marker[:k]):
                    hold = max(hold, k)
                    break
        end = len(self.text) - hold
        self._send(self.text[self.sent:end])
        self.sent = end

    def flush(self):
        if not self.blocked and self.sent < len(self.text):
            self._send(self.text[self.sent:])
            self.sent = len(self.text)

    @property
    def visible(self):
        cuts = [i for i in (self.text.find(m) for m in _TEXT_TOOL_MARKERS) if i >= 0]
        return self.text[:min(cuts)] if cuts else self.text


def parse_text_tool_calls(text):
    """Fallback for models that print tool calls as text (<tool_call>{json}</tool_call> or qwen3-coder XML)."""
    calls = []
    for raw in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", re.S):
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("name"):
            calls.append({"function": {"name": data["name"], "arguments": data.get("arguments") or data.get("parameters") or {}}})
    for name, body in re.findall(r"<function=([\w-]+)>(.*?)</function>", text or "", re.S):
        arguments = {}
        for key, value in re.findall(r"<parameter=([\w-]+)>\n?(.*?)\n?</parameter>", body, re.S):
            arguments[key] = value
        calls.append({"function": {"name": name, "arguments": arguments}})
    return calls


def _coerce_arguments(arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    return arguments if isinstance(arguments, dict) else None


class AgentError(Exception):
    pass


# ============================================================================ the agent
class CodingAgent:
    def __init__(self, *, emit, heartbeat, session_id, run, ollama_url, payload, budget_chars, base_system, request_text="", vision=None):
        self.emit = emit
        self.heartbeat = heartbeat
        self.session_id = session_id
        self.run = run
        self.ollama_url = ollama_url
        self.payload = payload
        self.budget_chars = budget_chars
        self.base_system = base_system
        self.request_text = request_text
        self.vision = vision
        self.active = ""
        self.modify = False
        self.spec = ""
        self.history = compact_history([m for m in payload["messages"][1:] if m.get("role") in {"user", "assistant", "system"}])
        self.project = ""
        self.auto_files = False
        self.total_calls = 0
        self.fail_streak = 0
        self.plan_steps = []
        self.plan_kind = ""
        self.tool_defs = TOOL_DEFINITIONS

    # ---- low-level model access ------------------------------------------------------
    def _request(self, payload):
        return Request(f"{self.ollama_url}/api/chat",
                       data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       headers={"Content-Type": "application/json"}, method="POST")

    def _check_cancel(self):
        if self.run["cancelled"].is_set():
            raise ChatCancelled()

    def _open(self, payload, timeout=600):
        try:
            return open_tracked(self.run, self._request(payload), timeout)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if "tools" in detail.lower() and ("support" in detail.lower() or "tool" in detail.lower()) and payload.get("tools"):
                raise AgentError("این مدل فراخوانی ابزار (tool calling) را پشتیبانی نمی‌کند؛ برای حالت برنامه‌نویسی "
                                 "مدلی مثل qwen3-coder، qwen2.5-coder، qwen3 یا llama3.1 را انتخاب کن.") from exc
            raise AgentError(f"Ollama درخواست را نپذیرفت: {detail[:400]}") from exc

    def _complete_json(self, payload):
        response = self._open(payload)
        try:
            with response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError):
            self._check_cancel()
            raise
        finally:
            untrack(self.run, response)

    def _stream(self, payload, forward=True):
        """Stream one model turn. Returns (visible_text, tool_calls, final_chunk)."""
        response = self._open({**payload, "stream": True})
        forwarder = _Forwarder(self.emit) if forward else None
        text, calls, final = "", [], {}
        checked, looped = 0, None
        try:
            with response:
                for line in response:
                    self._check_cancel()
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise AgentError(f"Ollama: {str(chunk['error'])[:400]}")
                    message = chunk.get("message") or {}
                    thought = message.get("thinking") or ""
                    if thought and forward:
                        self.emit({"message": {"role": "assistant", "content": "", "thinking": thought}, "done": False})
                    piece = message.get("content") or ""
                    if piece:
                        text += piece
                        if forwarder:
                            forwarder.add(piece)
                        if len(text) - checked >= 200:
                            checked = len(text)
                            looped = detect_loop(text)
                            if looped:
                                _abort_http_response(response)
                                final = {"done_reason": "loop", "looped": True}
                                break
                    calls.extend(message.get("tool_calls") or [])
                    if chunk.get("done"):
                        final = chunk
                        break
        except (OSError, ValueError) as exc:
            self._check_cancel()
            raise AgentError(f"ارتباط با Ollama قطع شد: {str(exc)[:300]}") from exc
        finally:
            untrack(self.run, response)
        self._check_cancel()
        if forwarder:
            forwarder.flush()
            visible = forwarder.visible
        else:
            visible = text
        if looped:
            visible = visible[:looped]
            self.emit({"notice": "مدل در حلقهٔ تکرار افتاده بود؛ متوقفش کردم."})
        if not calls and any(marker in text for marker in _TEXT_TOOL_MARKERS):
            calls = parse_text_tool_calls(text)
        return visible, calls, final

    def _done(self, reason="stop", final=None, extra_text=""):
        if extra_text:
            self.emit({"message": {"role": "assistant", "content": extra_text}, "done": False})
        event = {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": reason}
        for key in ("eval_count", "eval_duration", "total_duration"):
            if final and final.get(key):
                event[key] = final[key]
        self.emit(event)

    # ---- entry point -------------------------------------------------------------------
    def run_turn(self):
        try:
            self._turn()
        except ChatCancelled:
            self.emit({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "cancelled"})
        except AgentError as exc:
            self.emit({"error": str(exc), "done": True})
        except AgentFinished:
            pass
        except (HTTPError, URLError, TimeoutError) as exc:
            if self.run["cancelled"].is_set():
                self.emit({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "cancelled"})
            else:
                self.emit({"error": f"ارتباط با Ollama ناموفق بود: {str(exc)[:300]}", "done": True})

    def _turn(self):
        last_user = next((m for m in reversed(self.history) if m.get("role") == "user"), None)
        # the user's own words only (attachment blocks are prepended to the message by the server)
        request_text = self.request_text or (str(last_user.get("content", "")) if last_user else "")
        prior = self.history[:-1] if last_user is self.history[-1] else self.history
        has_projects = bool(tools.list_projects(limit=1))
        self.active = active_project_from_history(prior) or (tools.active_project_name() if has_projects else "")
        intent = classify_request(request_text, prior, has_projects)
        if intent in {"chat", "error_report", "discuss"}:
            self._answer(intent)
            return
        self.modify = is_modification(request_text, prior, self.active)
        allow_questions = not _has_context_mark(prior)
        if allow_questions and not self.modify:
            questions = interview_questions(request_text)
            if questions:
                self._done("clarify", extra_text=clarify_markdown({"questions": questions}))
                return
        self.emit({"notice": "در حال بررسی درخواست و تهیهٔ طرح…"})
        plan = self._make_plan(allow_questions)
        if plan is None:
            self._done("plan_failed", extra_text=(
                "نتوانستم طرح معتبری از مدل بگیرم (خروجی ساختاریافته نداد). درخواست را کمی کوتاه‌تر یا دقیق‌تر بنویس "
                "یا مدلی مثل qwen3-coder / qwen2.5-coder را انتخاب کن."))
            return
        if plan["decision"] == "answer":
            self._answer("chat")
            return
        if plan["decision"] == "ask":
            self._done("clarify", extra_text=clarify_markdown(plan))
            return
        if self.modify and self.active and not plan["project"]:
            plan["project"] = self.active
        if not plan["project"]:
            plan["project"] = "app-" + time.strftime("%m%d-%H%M")
        self._confirm_and_execute(plan)

    def _disk_context(self, project=None):
        try:
            return tools.project_context(project or self.project or self.active)
        except Exception:  # context is a help, never a reason to fail the turn
            return ""

    # ---- phase: plain answer (no tools) -------------------------------------------------
    def _answer(self, phase):
        overview = self._disk_context()
        system = build_system_prompt(self.base_system, phase, overview)
        payload = {**self.payload, "messages": [{"role": "system", "content": system}] + self.history}
        payload.pop("tools", None)
        text, _calls, final = self._stream(payload, forward=True)
        footer = READY_FOOTER if phase == "discuss" else ""
        if final.get("looped"):
            footer = "\n\n⚠️ مدل در حلقهٔ تکرار افتاد و پاسخ را قطع کردم؛ دوباره بپرس یا درخواست را کوتاه‌تر و دقیق‌تر بنویس." + footer
        if not text.strip() and not footer:
            footer = "پاسخی تولید نشد؛ دوباره بپرس یا مدل دیگری را امتحان کن."
        self._done(final.get("done_reason", "stop"), final, extra_text=footer)

    # ---- phase: plan ----------------------------------------------------------------------
    def _make_plan(self, allow_questions):
        overview = self._disk_context() or "(هنوز پروژه‌ای ساخته نشده)"
        if self.modify and self.active:
            try:
                overview += "\n\nکد فعلی پروژهٔ فعال:\n" + project_analysis.snapshot(tools._existing_project(self.active), 8000, tools._kind(tools._existing_project(self.active)))
            except (tools.SystemToolError, OSError):
                pass
            overview += "\n(درخواست فعلی دربارهٔ تغییر/اصلاح همین پروژهٔ موجود است؛ فقط در صورت ناچاری سؤال بپرس.)"
        rule = "" if allow_questions else "Questions were already asked in this conversation: do NOT use decision=\"ask\"; use sensible defaults and plan."
        system = PLANNER_SYSTEM_PROMPT.format(projects=overview, ask_rule=rule, toolchains=available_toolchains())
        conversation = [{"role": m["role"], "content": str(m.get("content", ""))[:12000]}
                        for m in self.history if m.get("role") in {"user", "assistant"}][-8:]
        base = {"model": self.payload["model"], "stream": False,
                "keep_alive": self.payload.get("keep_alive", "10m"),
                "options": {**self.payload.get("options", {}), "num_predict": 1800, "temperature": 0.2}}
        if "think" in self.payload:
            base["think"] = False
        forced = ""
        fallback = None
        for fmt in (PLAN_SCHEMA, "json", PLAN_SCHEMA):
            request_payload = {**base, "format": fmt,
                               "messages": [{"role": "system", "content": system + forced}] + conversation}
            result = self._complete_json(request_payload)
            self._check_cancel()
            plan = parse_plan((result.get("message") or {}).get("content", ""))
            if plan is None:
                self.emit({"notice": "خروجی طرح معتبر نبود؛ دوباره تلاش می‌کنم…"})
                continue
            if plan["decision"] == "ask" and not allow_questions:
                if plan["steps"]:
                    plan["decision"] = "plan"
                    return plan
                fallback = plan
                forced = ("\nIMPORTANT: the user already answered; do NOT ask again. Use sensible defaults for anything "
                          "unspecified and return decision=\"plan\" with concrete steps.")
                continue
            return plan
        if fallback is not None:
            fallback["decision"] = "answer"
            return fallback
        return None

    def _confirm_and_execute(self, plan):
        plan_id = secrets.token_urlsafe(12)
        waiter = register_approval(plan_id, self.session_id)
        if waiter is None:
            self._done("busy", extra_text="چند درخواست تأیید همزمان باز است؛ ابتدا آن‌ها را ببند و دوباره بفرست.")
            return
        self.emit({"message": {"role": "assistant", "content": plan_markdown(plan)}, "done": False})
        self.emit({"plan_approval": {"plan_id": plan_id, "title": plan["title"], "summary": plan["summary"],
                                     "project": plan["project"], "steps": plan["steps"],
                                     "expires_in": PLAN_DECISION_TIMEOUT}})
        try:
            self._wait(waiter, PLAN_DECISION_TIMEOUT)
        finally:
            drop_approval(plan_id)
        self.emit({"approval_closed": plan_id})
        self._check_cancel()
        decision = waiter["result"] or {}
        if not decision.get("approved"):
            if waiter["result"] is None:
                text = "\nزمان تأیید طرح تمام شد؛ چیزی ساخته نشد. اگر هنوز می‌خواهی، دوباره بنویس «بساز»."
            else:
                text = "\nطرح تأیید نشد؛ هیچ ابزاری اجرا نشد و چیزی تغییر نکرد. بگو چه بخشی را عوض کنم تا " + \
                       "طرح تازه بدهم (چه چیزی را عوض کنم؟)."
            self._done("plan_rejected", extra_text=text)
            return
        self.auto_files = bool(decision.get("auto_files"))
        self.plan_steps = plan["steps"]
        self.project = plan["project"]
        self.plan_kind = plan.get("kind") or ""
        self.spec = "\n".join(f"- {item}" for item in plan.get("spec", []))
        self._execute(plan)

    def _wait(self, waiter, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if waiter["event"].wait(timeout=10):
                return
            self._check_cancel()
            if time.monotonic() > deadline:
                return
            self.heartbeat()  # also detects a closed browser tab (raises OSError)

    # ---- phase: execute -----------------------------------------------------------------
    @staticmethod
    def _slim(messages, keep_tail=6):
        """Old file bodies (written or read) are redundant: the live code snapshot in the system prompt has them."""
        messages = [dict(m) for m in messages]
        for message in messages[1:-keep_tail]:
            if message.get("role") == "tool" and len(message.get("content") or "") > 700:
                message["content"] = message["content"][:300] + "…[old result trimmed]"
            for call in message.get("tool_calls") or []:
                arguments = (call.get("function") or {}).get("arguments")
                if isinstance(arguments, dict):
                    call["function"] = {**call["function"], "arguments": {
                        key: (value if not isinstance(value, str) or len(value) < 300 else f"[{len(value)} chars written; see current code]")
                        for key, value in arguments.items()}}
        return messages

    def _compact(self, messages):
        """Keep the prompt inside the model context: shrink old tool results and old file bodies first."""
        def size(msgs):
            total = 0
            for message in msgs:
                total += len(message.get("content") or "")
                for call in message.get("tool_calls") or []:
                    total += len(json.dumps((call.get("function") or {}).get("arguments", {}), ensure_ascii=False))
            return total

        if size(messages) <= self.budget_chars:
            return messages
        messages = self._slim(messages)
        if size(messages) <= self.budget_chars:
            return messages
        keep_tail = 4
        for index, message in enumerate(messages[:-keep_tail]):
            if index == 0:
                continue
            if message.get("role") == "tool" and len(message.get("content") or "") > 500:
                message["content"] = message["content"][:240] + " …[بخش قدیمی نتیجه برای صرفه‌جویی در زمینه حذف شد]"
            for call in message.get("tool_calls") or []:
                arguments = (call.get("function") or {}).get("arguments")
                if isinstance(arguments, dict):
                    call["function"] = {**call["function"], "arguments": {
                        key: (value if not isinstance(value, str) or len(value) < 400
                              else f"[{len(value)} نویسه — قبلاً نوشته شد؛ برای دیدن محتوا read_file بزن]")
                        for key, value in arguments.items()}}
        index = 1
        while size(messages) > self.budget_chars and index < len(messages) - keep_tail:
            if messages[index].get("role") in {"user", "assistant"} and not messages[index].get("tool_calls"):
                del messages[index]
            else:
                index += 1
        return messages

    def _execute(self, plan):
        steps_text = "\n".join(f"{s['n']}. {s['title']} — {s['detail']} [ابزار: {s['tool']}]" for s in plan["steps"])
        game = bool(re.search(r"pygame|بازی|game", steps_text + " " + self.spec + " " + normalize(self.request_text), re.I))
        kind = self.plan_kind or "python"
        try:
            if self.project and not self.plan_kind:
                kind = tools._kind(tools._existing_project(self.project))
        except (tools.SystemToolError, OSError):
            pass
        self.tool_defs = select_tools(kind, [s["tool"] for s in plan["steps"]], game)
        messages = [{"role": "system", "content": self._execute_system(steps_text, game)}] + self.history
        max_calls = min(MAX_TOOL_CALLS, len(plan["steps"]) * 3 + 8)
        nudges = 0
        for _round in range(max_calls + 6):
            self._check_cancel()
            messages[0] = {"role": "system", "content": self._execute_system(steps_text, game)}  # real disk state, refreshed
            request_payload = {**self.payload, "messages": self._compact(messages), "tools": self.tool_defs}
            visible, calls, final = self._stream(request_payload, forward=True)
            if not calls:
                truncated = final.get("done_reason") == "length"
                looped = final.get("looped")
                if (self.total_calls == 0 or truncated or looped) and nudges < 2:
                    nudges += 1
                    messages.append({"role": "assistant", "content": visible})
                    messages.append({"role": "system", "content": (
                        "خروجی‌ات بریده شد؛ فایل را کوچک‌تر کن (چند فایل کوچک) و با ابزار بنویس." if truncated else
                        "متن تکراری تولید کردی و قطع شد؛ هیچ متنی را تکرار نکن و مرحلهٔ بعدی را همین حالا با ابزار انجام بده." if looped else
                        "تو باید کار را با فراخوانی ابزار انجام بدهی نه با متن. مرحلهٔ بعدی طرح را همین حالا با ابزار انجام بده.")})
                    self.emit({"notice": "مدل ابزار را صدا نزد؛ یادآوری می‌کنم…"})
                    continue
                if self.total_calls == 0:
                    self._done("no_tool_calls", final, extra_text=(
                        "\n\nمدل هیچ ابزاری را فراخوانی نکرد؛ احتمالاً این مدل tool-calling را خوب پشتیبانی نمی‌کند. "
                        "مدلی مثل qwen3-coder یا qwen2.5-coder را امتحان کن."))
                    return
                self._done(final.get("done_reason", "stop"), final)
                return
            messages.append({"role": "assistant", "content": visible, "tool_calls": calls})
            for call in calls:
                if self.total_calls >= max_calls:
                    self._done("limit", final, extra_text="\n\nبه سقف تعداد عملیات این اجرا رسیدم؛ برای ادامه بنویس «ادامه بده».")
                    return
                self.total_calls += 1
                result = self._handle_call(call)
                if result is None:  # rejected by the user or cancelled: the chain stops here
                    return
                messages.append({"role": "tool", "tool_name": (call.get("function") or {}).get("name", "unknown"),
                                 "content": self._tool_message(result)})
                self.fail_streak = 0 if result.get("ok") else self.fail_streak + 1
                if self.fail_streak >= MAX_FAIL_STREAK:
                    self._done("failing", final, extra_text=(
                        "\n\nچند عملیات پشت‌سرهم ناموفق بود؛ برای جلوگیری از حلقهٔ بی‌نتیجه متوقف شدم. "
                        "پیام‌های بالا را ببین و بگو چه کاری کنم، یا مدل قوی‌تری را انتخاب کن."))
                    return
        self._done("limit", extra_text="\n\nبه سقف مراحل رسیدم؛ برای ادامه بنویس «ادامه بده».")

    def _execute_system(self, steps_text, game):
        snapshot = ""
        try:
            project = tools._existing_project(self.project)
            snapshot = project_analysis.snapshot(project, max(4000, int(self.budget_chars * 0.3)), tools._kind(project))
        except (tools.SystemToolError, OSError):
            snapshot = ""
        return build_system_prompt(self.base_system, "execute", self._disk_context(self.project), steps_text, self.project,
                                   spec=self.spec, code_snapshot=snapshot, game=game)

    _DROP_KEYS = ("action", "label", "kind_label", "import_graph", "path", "command", "backup", "purpose", "step", "plan", "stdout_truncated")

    def _tool_message(self, result):
        """Compact JSON for the model: only what it needs to decide the next call."""
        trimmed = {k: v for k, v in result.items() if k not in self._DROP_KEYS and v not in (None, "", [], {})}
        limit = max(3000, self.budget_chars // 6)
        if trimmed.get("action") == "read_file" or "content" in trimmed:
            limit = max(limit, 6000)
        for key, value in list(trimmed.items()):
            if isinstance(value, str) and len(value) > limit:
                trimmed[key] = value[-limit:] if key in {"output", "stdout", "stderr"} else value[:limit] + "\n…[read_file for the rest]"
            elif isinstance(value, list) and len(value) > 40:
                trimmed[key] = value[:40] + [f"…+{len(value) - 40}"]
        return json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))

    def _step_label(self):
        total = len(self.plan_steps)
        return f"{min(self.total_calls, total)}/{total}" if total else ""

    def _result_markdown(self, name, label, result):
        icon = "🔍" if name in tools.READ_ONLY_ACTIONS else ("✅" if result.get("ok") else "❌")
        if name == "check_project" and result.get("problems"):
            items = "\n".join(f"> - {problem}" for problem in result["problems"][:8])
            return f"\n> 🔍 {result.get('summary') or label}\n{items}\n\n"
        if icon == "🔍":
            return f"\n> 🔍 {result.get('summary') or label}\n\n"
        lines = [f"\n\n> {icon} **{label}** — {result.get('summary') or result.get('error') or ''}"]
        if result.get("hint") and not result.get("ok"):
            lines.append(f"> 💡 {result['hint']}")
        if name == "screenshot_check":
            for note in result.get("observations", [])[:6]:
                lines.append(f"> 📷 {note}")
            if result.get("vision_description"):
                lines.append("> 👁️ **مدل بینایی:** " + str(result["vision_description"]).strip().replace("\n", "\n> "))
            if result.get("folder"):
                lines.append(f"> 🗂️ اسکرین‌شات‌ها: `{result['folder']}`")
        if result.get("note"):
            lines.append(f"> ℹ️ {result['note']}")
        output = (result.get("output") or "").strip()
        if output and name in {"venv_run", "venv_test", "docker_test", "docker_run", "venv_install", "screenshot_check"} and \
                (not result.get("ok") or name != "venv_install"):
            lines.append("\n```text\n" + output[-1800:].replace("```", "'''") + "\n```")
        return "\n".join(lines) + "\n\n"

    def _handle_call(self, call):
        function = call.get("function") or {}
        name = function.get("name")
        arguments = _coerce_arguments(function.get("arguments", {}))
        if name not in TOOL_NAMES or arguments is None:
            result = {"ok": False, "error": f"ابزار «{name}» نامعتبر است؛ فقط ابزارهای فهرست‌شده مجازند."}
            self.emit({"message": {"role": "assistant", "content": f"\n\n> ❌ {result['error']}\n\n"}, "done": False})
            return result
        args = normalize_arguments(name, arguments, self.project)
        if name == "create_project" and not args.get("kind") and self.plan_kind:
            args["kind"] = self.plan_kind
        label = TOOL_LABELS.get(name, name)
        step_index = min(self.total_calls, len(self.plan_steps)) - 1
        if not args.get("purpose") and 0 <= step_index < len(self.plan_steps):
            args["purpose"] = self.plan_steps[step_index]["title"]
        args["step"] = args.get("step") or self._step_label()
        shown = args.get("path") or args.get("project") or args.get("name") or ""
        try:
            if name in tools.READ_ONLY_ACTIONS:
                self.emit({"notice": f"{label}: {shown}…"})
                result = tools.read_only_action(name, args)
            else:
                result = self._run_prepared(name, label, args, shown)
                if result is None:
                    return None
        except tools.SystemToolError as exc:
            result = {"ok": False, "error": str(exc), "summary": str(exc)}
        except (OSError, ValueError) as exc:
            result = {"ok": False, "error": str(exc)[:600], "summary": str(exc)[:300]}
        if name == "create_project" and result.get("ok") and not self.project:
            self.project = args.get("name", "")
        if name == "screenshot_check" and result.get("ok") and self.vision and result.get("shots"):
            result["vision_description"] = self._look(result)
        self._check_cancel()
        self.emit({"message": {"role": "assistant", "content": self._result_markdown(name, label, result)}, "done": False})
        return result

    def _look(self, result):
        """Ask the local vision model what the last screenshot really shows (None when unavailable)."""
        try:
            shot = result["shots"][-1]
            path = str(Path(result["folder"]) / shot["file"])
            self.emit({"notice": "مدل بینایی در حال بررسی اسکرین‌شات…"})
            question = (
                "این اسکرین‌شات یک برنامه یا بازی دسکتاپ است. بدون حدس‌زدن و فقط بر اساس آنچه می‌بینی، به فارسی در حداکثر ۸ خط بگو: "
                "چه چیزی نمایش داده می‌شود؟ آیا صفحهٔ اصلی بازی/رابط و اجزای اصلی (مثلاً مهره‌ها) دیده می‌شوند؟ رنگ‌ها و اندازهٔ نسبی چیست؟ "
                "مشکل ظاهری (خالی بودن، کوچکی، همپوشانی، بریدگی) داری؟ مشخصات خواستهٔ کاربر:\n" + (self.spec or "(ذکر نشده)"))
            text = self.vision(path, question)
            return (text or "").strip()[:1200] or None
        except ChatCancelled:
            raise
        except Exception:
            return None

    def _run_prepared(self, name, label, args, shown):
        prepared = tools.prepare_action(name, args, self.session_id)
        approval_id = prepared["approval_id"]
        auto = self.auto_files and name in tools.SAFE_WRITE_ACTIONS
        if not auto:
            waiter = register_approval(approval_id, self.session_id)
            if waiter is None:
                tools.reject_action(approval_id, self.session_id)
                return {"ok": False, "error": "تعداد درخواست‌های در انتظار تأیید بیش از حد مجاز است.", "summary": "درخواست‌های تأیید زیاد است."}
            self.emit({"approval_required": prepared})
            try:
                self._wait(waiter, APPROVAL_WAIT)
            finally:
                drop_approval(approval_id)
            self.emit({"approval_closed": approval_id})
            if self.run["cancelled"].is_set():
                tools.reject_action(approval_id, self.session_id)
                raise ChatCancelled()
            decision = waiter["result"]
            if decision is None:
                tools.reject_action(approval_id, self.session_id)
                return {"ok": False, "error": "زمان تأیید کاربر تمام شد؛ عملیات اجرا نشد.", "summary": "زمان تأیید تمام شد؛ اجرا نشد."}
            if not decision.get("approved"):
                tools.reject_action(approval_id, self.session_id)
                self.emit({"message": {"role": "assistant", "content": f"\n\n> ⛔ **{label}** رد شد.\n\n"}, "done": False})
                self._done("user_rejected", extra_text=REJECT_TEXT)
                return None
        self.emit({"notice": f"{label}: {shown}…"})
        return tools.approve_action(approval_id, self.session_id, cancelled=lambda: self.run["cancelled"].is_set())
