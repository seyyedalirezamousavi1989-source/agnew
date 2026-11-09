"""Explicit, local-only long-term memory for LocalChat.

Design notes (v2):

* Only what the user explicitly asks to remember is stored (``parse_memory_command``).  The command may be
  single-line or multiline and the trigger phrase may stand anywhere on its own line / at a line edge.
* A multiline block (a list of models, hardware, rules ...) is split into one entry per line, so each entry
  can be retrieved, updated and forgotten independently; the introducing line («مدل‌های من:») is kept as a
  prefix of every item so that an entry never loses its meaning.
* Every entry gets a category (preference / hardware / model / project / instruction / general).
* Duplicates are detected (exact and near-duplicate); «label: value» facts about a single-valued thing
  (GPU, RAM, OS ...) update the old value instead of piling up.
* Temporary, trivial or secret-looking entries are rejected with a reason.
* Writes are real transactions and are read back before the caller is told «saved».
* Retrieval is relevance based (``relevant`` / ``prompt_context(query)``), not «the newest N rows».
* The schema is versioned (``PRAGMA user_version``); an old database is backed up before it is migrated and
  every existing memory is preserved.
"""
import logging
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

try:
    from .dbutil import Database
except ImportError:                     # pragma: no cover - flat import (launcher / tests)
    from dbutil import Database

log = logging.getLogger("localchat.memory")

SCHEMA_VERSION = 2
MAX_COMMAND_LENGTH = 20000     # a whole inventory / long list is fine; a pasted book is not
MAX_COMMAND_LINES = 400
MAX_ENTRY_LENGTH = 2000        # add(): single entry limit (unchanged from v1)
SPLIT_ENTRY_LENGTH = 800       # a longer line is split into sentences of at most this size
MAX_ENTRIES_PER_COMMAND = 300
KEEP_BACKUPS = 3

CATEGORIES = ("instruction", "preference", "hardware", "model", "project", "general")
CATEGORY_LABELS = {
    "instruction": "دستورهای دائمی",
    "preference": "ترجیحات کاربر",
    "hardware": "سخت‌افزار",
    "model": "مدل‌های محلی",
    "project": "پروژه‌ها",
    "general": "سایر",
}

_ZW = "‌"
_SP = r"[\s‌]*"
_OBJECT = (r"(?:این(?:[‌]?ها|ا|و)?"
           r"(?:\s*(?:اطلاعات(?:\s*بالا)?|موارد|مورد|لیست|فهرست|مشخصات|نکات|مدل[‌]?ها|دستورات|چیزها))?"
           r"(?:\s*(?:رو|را))?|اینارو)")
_LEAD = r"(?:(?:لطفاً|لطفا)\s*)?"

# (pattern, needs_object_before) – earliest match on a line wins
_TRIGGERS = (
    re.compile(_LEAD + r"(?:" + _OBJECT + r"\s*)?(?:یادت\s*(?:باشه|باشد|بمونه|نره|بمانه)"
               r"|به" + _SP + r"خاطر\s*(?:بسپار|بسپر|داشته" + _SP + r"باش)"
               r"|(?:در|تو|توی|به)\s*حافظه(?:" + _ZW + r"?ات)?\s*(?:نگه" + _SP + r"دار|ذخیره\s*کن|اضافه\s*کن))", re.I),
    # «این اطلاعات رو ذخیره کن» – the object is REQUIRED so «این فایل را ذخیره کن» stays a normal request
    re.compile(_LEAD + _OBJECT + r"\s+(?:ذخیره|یادداشت|ثبت)\s*کن", re.I),
    # bare «ذخیره کن: ...» only with a colon
    re.compile(_LEAD + r"(?:ذخیره|یادداشت|ثبت)\s*کن(?=\s*[:：])", re.I),
)
_EDGE_PUNCT = ".!؟?،,;:؛"
FORGET_ALL = re.compile(r"^\s*(?:همه(?:ی|ٔ)?\s*)?(?:حافظه|یادداشت(?:ها)?)\s*(?:رو|را)?\s*(?:پاک|حذف)\s*کن\s*[.!؟]?\s*$", re.I)
# «فراموش کن ...» always; «حذف/پاک کن» only with a colon («حذف کن خط ۳ را» is a normal request)
FORGET_ONE = re.compile(
    r"^\s*(?:این(?:و|\s*(?:رو|را))?\s*)?(?:فراموش\s*کن\s*[:：،\-]?|(?:حذف|پاک)\s*کن\s*[:：])\s*(.+)$",
    re.I | re.S)
LIST_MEMORY = re.compile(r"(?:چه\s*چیز(?:هایی)?\s*(?:رو|را)?\s*(?:یادت|به\s*خاطرت)\s*هست|حافظه(?:ات)?\s*(?:رو|را)?\s*(?:نشون|نمایش)\s*بده|چه\s*چیز(?:هایی)?\s*ذخیره\s*کردی)", re.I)


def _first_trigger(line):
    best = None
    for pattern in _TRIGGERS:
        match = pattern.search(line)
        if match and (best is None or match.start() < best.start()):
            best = match
    return best


def extract_save_content(text):
    """The content to remember if ``text`` is a save command, else None.

    A line counts as carrying the trigger when the trigger stands at the start of the line, at its end, or
    right after sentence punctuation.  A trigger buried in the middle of an ordinary sentence («فقط یادت باشه
    x رو بنویس») is NOT a command.  The trigger line is replaced by whatever text surrounds the trigger.
    """
    lines = text.split("\n")
    parts, found = [], False
    for line in lines:
        match = _first_trigger(line)
        if match:
            prefix = re.sub(r"^\s*(?:لطفاً|لطفا)\s*", "", line[:match.start()]).strip()
            suffix = line[match.end():].strip(" \t:：،,-–")
            at_edge = (not prefix) or (not suffix) or prefix[-1] in _EDGE_PUNCT
            if at_edge:
                found = True
                prefix = prefix.rstrip(" \t،,")
                if prefix:
                    parts.append(prefix)
                if suffix:
                    parts.append(re.sub(r"^(?:که|اینکه)\s+", "", suffix))
                continue
        parts.append(line)
    if not found:
        return None
    content = "\n".join(parts).strip()
    content = re.sub(r"^(?:که|اینکه)\s+", "", content)
    return content or None


def parse_memory_command(text):
    if not isinstance(text, str):
        return None
    value = text.strip()
    # pasted code / huge documents are never memory commands
    if not value or "```" in value or len(value) > MAX_COMMAND_LENGTH or value.count("\n") > MAX_COMMAND_LINES:
        return None
    if FORGET_ALL.fullmatch(value):
        return ("forget_all", "")
    match = FORGET_ONE.match(value)
    if match:
        return ("forget", match.group(1).strip())
    if LIST_MEMORY.search(value) and len(value) < 200:
        return ("list", "")
    content = extract_save_content(value)
    if content:
        return ("save", content)
    return None


# ---------------------------------------------------------------- text helpers
_BULLET = re.compile(r"^\s*(?:[-*•●▪‣·–—]+|[0-9۰-۹]+\s*[.)\-–]|[a-zA-Z]\))\s+")
_HEADER_FILLER = re.compile(
    r"(?:\s*(?:اینه|این[‌]?هاست|این[‌]?ها|اینا|هستن|هستند|به\s*شرح\s*زیر(?:\s*است)?|زیره|زیر\s*است|زیر))+\s*$")


_LABELLED = re.compile(r"^[^:：\d]{2,30}[:：]\s+\S")


def _clean_line(line):
    return " ".join(line.replace("\t", " ").split())


def _is_bullet(line):
    return bool(_BULLET.match(line))


def _split_long(line):
    if len(line) <= SPLIT_ENTRY_LENGTH:
        return [line]
    sentences = re.split(r"(?<=[.!؟?؛])\s+", line)
    chunks, current = [], ""
    for sentence in sentences:
        while len(sentence) > SPLIT_ENTRY_LENGTH:          # an unbroken monster sentence
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:SPLIT_ENTRY_LENGTH])
            sentence = sentence[SPLIT_ENTRY_LENGTH:]
        if current and len(current) + 1 + len(sentence) > SPLIT_ENTRY_LENGTH:
            chunks.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        chunks.append(current)
    return chunks


def split_entries(text):
    """Turn a (possibly multiline) command body into single-line entries, keeping the list header."""
    lines = [_clean_line(line) for line in str(text).splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []
    header = None
    if len(lines) > 1:
        first = lines[0]
        rest = lines[1:]
        if first.endswith((":", "：")):
            header = _HEADER_FILLER.sub("", first.rstrip(":： ")).strip() or first.rstrip(":： ").strip()
            lines = rest
        elif not _is_bullet(first) and sum(_is_bullet(line) for line in rest) * 2 >= len(rest) and len(first) <= 120:
            header = _HEADER_FILLER.sub("", first.rstrip(" .:：")).strip()
            lines = rest
    entries = []
    for line in lines:
        item = _BULLET.sub("", line).strip()
        if not item:
            continue
        self_describing = bool(_LABELLED.match(item))      # «رم: 32GB» already says what it is
        for piece in _split_long(item):
            entries.append(f"{header}: {piece}" if header and not self_describing else piece)
    return entries[:MAX_ENTRIES_PER_COMMAND]


_FILLER = re.compile(r"^(?:سلام|درود|ممنون|مرسی|تشکر|ok|okay|باشه|اوکی|آره|نه|بله|خب|hi|hello|thanks)[\s.!؟?]*$", re.I)
_TEMPORARY = re.compile(
    r"(?:همین\s*الان|\bالان\b|امروز|امشب|فردا|دیروز|این\s*لحظه|چند\s*دقیقه|چند\s*ساعت|\bnow\b|\btoday\b|\btonight\b|\btomorrow\b)", re.I)
_PERSISTENT = re.compile(r"(?:همیشه|معمولاً|معمولا|هرگز|هیچ\s*وقت|هیچ‌وقت|از\s*این\s*به\s*بعد|دائماً|دائما|\balways\b|\bnever\b)", re.I)
_SECRET = re.compile(
    r"(?:sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
    r"|(?:password|passwd|api[\s_\-]?key|secret|token|رمز(?:\s*عبور)?|پسورد)\s*(?:[:=]|است|هست)\s*\S{6,})", re.I)


def validate_entry(text):
    """(ok, reason) – reject empty, trivial, temporary, question-only and secret-looking entries."""
    value = str(text or "").strip()
    if len(value) < 2 or not re.search(r"\w", value):
        return False, "متن خیلی کوتاه یا نامعتبر است"
    if len(value) > MAX_ENTRY_LENGTH:
        return False, "متن بیش از حد طولانی است"
    if _FILLER.match(value):
        return False, "این یک جملهٔ عادی است و واقعیتی برای نگه‌داشتن ندارد"
    if _SECRET.search(value):
        return False, "شبیه رمز یا کلید محرمانه است و ذخیره نمی‌شود"
    if value.rstrip().endswith(("؟", "?")):
        return False, "این یک پرسش است، نه اطلاعاتی برای ذخیره"
    if _TEMPORARY.search(value) and not _PERSISTENT.search(value):
        return False, "به نظر اطلاعاتی موقت است (امروز/الان/فردا)"
    return True, ""


_CATEGORY_RULES = (
    ("instruction", re.compile(
        r"(?:همیشه|هرگز|هیچ\s*وقت|هیچ‌وقت|از\s*این\s*به\s*بعد|حتماً|حتما|دستور|قانون|قاعده|باید\s+(?:همیشه|پاسخ|جواب)"
        r"|\balways\b|\bnever\b|\brule\b|instruction)", re.I)),
    ("model", re.compile(
        r"(?:ollama|اولاما|\bمدل|\bllm\b|\b[\w.\-]+:(?:\d+(?:\.\d+)?[bm]\b|latest|q\d)|gemma|llama|qwen|mistral|mixtral|phi\d?|"
        r"deepseek|codellama|starcoder|nomic|embed|\bgguf\b|whisper)", re.I)),
    ("hardware", re.compile(
        r"(?:\bgpu\b|vram|\bram\b|\bcpu\b|\brtx\b|\bgtx\b|\bssd\b|\bnvme\b|کارت\s*گرافیک|گرافیک|پردازنده|رم\b|حافظه\s*رم|هارد|"
        r"لپ[‌]?تاپ|مادربرد|سخت[‌]?افزار|\bintel\b|\bamd\b|\bnvidia\b|ویندوز|لینوکس|سیستم[‌]?عامل|\bwindows\b|\blinux\b|\bmacos\b)", re.I)),
    ("project", re.compile(
        r"(?:پروژه|project|\brepo\b|مخزن|گیت[‌]?هاب|github|localchat|django|flutter|اپلیکیشن|برنامه[‌]?ام|سایت[‌]?م)", re.I)),
    ("preference", re.compile(
        r"(?:ترجیح|دوست\s*دارم|دوست\s*ندارم|خوشم|می[‌ ]?خوام|میخوام|می[‌ ]?خواهم|علاقه|لحن|کوتاه|مختصر|خلاصه\s*باشه|"
        r"فارسی\s*(?:صحبت|جواب|پاسخ)|\bprefer|\blike\b)", re.I)),
)


def classify(text):
    value = str(text or "")
    head = value.split(":", 1)[0] if ":" in value[:60] else ""
    for scope in (head, value):         # the introducing label is the strongest hint
        if not scope:
            continue
        for category, pattern in _CATEGORY_RULES:
            if pattern.search(scope):
                return category
    return "general"


_SINGLETON = re.compile(
    r"^(کارت\s*گرافیک|گرافیک|gpu|vram|رم|ram|پردازنده|cpu|سیستم[‌]?عامل|os|نام|اسم|زبان)\s*(?:من|ام)?\s*[:：]\s*(.+)$", re.I)


def singleton_label(text):
    match = _SINGLETON.match(str(text).strip())
    if not match:
        return None
    return " ".join(match.group(1).casefold().replace(_ZW, " ").split())


def _fold(value):
    value = str(value).casefold()
    for source, target in (("ي", "ی"), ("ك", "ک"), ("ۀ", "ه"), ("ـ", ""), ("ة", "ه")):
        value = value.replace(source, target)
    return value


_STOP = frozenset("""و در به از که این آن را رو با برای تا یا هم من تو ما شما او چی چه چرا کجا کی چطور چگونه است هست بود باید می
the a an of to in is are and or for with on my me what which how do you i can""".split())
_SUFFIXES = ("هایی", "های", "هام", "ها", "ام", "ات")


def _tokens(text):
    value = _fold(text).replace(_ZW, " ")
    found = set()
    for raw in re.findall(r"[\w.:+\-]+", value):
        raw = raw.strip(".:-_+")
        if len(raw) < 2:
            continue
        pieces = {raw}
        if re.search(r"[:.\-_]", raw):
            pieces.update(piece for piece in re.split(r"[:.\-_]", raw) if len(piece) >= 2)
        for piece in pieces:
            for suffix in _SUFFIXES:
                if piece.endswith(suffix) and len(piece) - len(suffix) >= 3:
                    piece = piece[:-len(suffix)]
                    break
            if piece not in _STOP:
                found.add(piece)
    return found


def _overlap(left, right):
    """Number of query tokens matched by item tokens (exact or a 3+ char prefix of each other)."""
    score = 0
    for token in left:
        if token in right:
            score += 2
        elif len(token) >= 3 and any((len(other) >= 3 and (other.startswith(token) or token.startswith(other))) for other in right):
            score += 1
    return score


_QUERY_HINTS = {
    "model": re.compile(r"(?:مدل|ollama|اولاما|llm|gemma|llama|qwen|mistral|deepseek|phi)", re.I),
    "hardware": re.compile(r"(?:گرافیک|gpu|vram|رم\b|ram\b|cpu|پردازنده|سخت[‌]?افزار|سیستم|ویندوز|لپ[‌]?تاپ|هارد|ssd)", re.I),
    "project": re.compile(r"(?:پروژه|project|repo|مخزن|کد|برنامه|localchat)", re.I),
    "preference": re.compile(r"(?:ترجیح|دوست|لحن|سبک|پاسخ|جواب)", re.I),
}


class MemoryStoreError(RuntimeError):
    """The write could not be committed / verified; the caller must NOT tell the user it was saved."""


class PersistentMemory:
    def __init__(self, database_path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.path)
        self.last_backup = None
        self._migrate()

    # ------------------------------------------------------------ connection (shared plumbing: dbutil.Database)
    def _tx(self):
        return self.db.tx()

    def _read(self):
        return self.db.read()

    # ------------------------------------------------------------ schema / migration
    def _migrate(self):
        with self._read() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            existing = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'").fetchone()
        if version >= SCHEMA_VERSION and existing:
            return
        if existing and version < SCHEMA_VERSION:
            self.last_backup = self.db.backup("v1", KEEP_BACKUPS)     # never touch old data without a copy
        with self._tx() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, content TEXT NOT NULL, normalized TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
            )""")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(memories)")}
            if "category" not in columns:
                db.execute("ALTER TABLE memories ADD COLUMN category TEXT NOT NULL DEFAULT 'general'")
            if "label" not in columns:
                db.execute("ALTER TABLE memories ADD COLUMN label TEXT")
            if version < SCHEMA_VERSION:
                for row in db.execute("SELECT id, content FROM memories").fetchall():
                    db.execute("UPDATE memories SET category=?, label=? WHERE id=?",
                               (classify(row["content"]), singleton_label(row["content"]), row["id"]))
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category, updated_at DESC)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_label ON memories(category, label)")
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @staticmethod
    def _normalize(value):
        return " ".join(_fold(value).replace(_ZW, " ").split())

    # ------------------------------------------------------------ writing
    def _upsert(self, db, content, now):
        """Insert/update one validated, single-line entry inside an open transaction."""
        normalized = self._normalize(content)
        category = classify(content)
        label = singleton_label(content)
        row = db.execute("SELECT id FROM memories WHERE normalized=?", (normalized,)).fetchone()
        if row:
            db.execute("UPDATE memories SET content=?, updated_at=? WHERE id=?", (content, now, row["id"]))
            return row["id"], "duplicate", category
        if label:                                          # «کارت گرافیک: ...» replaces the old value
            row = db.execute("SELECT id FROM memories WHERE category=? AND label=?", (category, label)).fetchone()
            if row:
                db.execute("UPDATE memories SET content=?, normalized=?, updated_at=? WHERE id=?",
                           (content, normalized, now, row["id"]))
                return row["id"], "updated", category
        tokens = _tokens(content)
        if len(tokens) >= 3:                               # near-duplicate in the same category = update
            for other in db.execute("SELECT id, content FROM memories WHERE category=?", (category,)).fetchall():
                other_tokens = _tokens(other["content"])
                union = tokens | other_tokens
                if union and len(tokens & other_tokens) / len(union) >= 0.85:
                    db.execute("UPDATE memories SET content=?, normalized=?, updated_at=? WHERE id=?",
                               (content, normalized, now, other["id"]))
                    return other["id"], "updated", category
        memory_id = str(uuid.uuid4())
        db.execute("INSERT INTO memories (id, content, normalized, created_at, updated_at, category, label) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?)", (memory_id, content, normalized, now, now, category, label))
        return memory_id, "saved", category

    def _verify(self, ids):
        if not ids:
            return
        with self._read() as db:
            marks = ",".join("?" for _ in ids)
            found = db.execute(f"SELECT COUNT(*) FROM memories WHERE id IN ({marks})", list(ids)).fetchone()[0]
        if found != len(set(ids)):
            raise MemoryStoreError("ذخیره‌سازی حافظه تأیید نشد.")

    def save_text(self, text):
        """Save a command body (single or multiline).  Returns a report; raises MemoryStoreError on failure.

        ``report = {"saved": [...], "updated": [...], "duplicates": [...], "rejected": [(text, reason)], "total": n}``
        The caller may confirm «saved» only for the items listed here – they were committed AND read back.
        """
        entries = split_entries(text)
        report = {"saved": [], "updated": [], "duplicates": [], "rejected": [], "total": len(entries)}
        accepted = []
        for entry in entries:
            ok, reason = validate_entry(entry)
            if ok:
                accepted.append(entry)
            else:
                report["rejected"].append((entry, reason))
        if not accepted:
            return report
        now = int(time.time())
        ids = []
        try:
            with self._tx() as db:
                for entry in accepted:
                    memory_id, outcome, category = self._upsert(db, entry, now)
                    ids.append(memory_id)
                    report[{"saved": "saved", "updated": "updated", "duplicate": "duplicates"}[outcome]].append(
                        {"id": memory_id, "content": entry, "category": category})
        except sqlite3.Error as exc:
            log.exception("memory write failed")
            raise MemoryStoreError(f"نوشتن در پایگاه حافظه ناموفق بود: {exc}") from exc
        self._verify(ids)
        return report

    def add(self, content):
        """Store ONE entry (kept from v1 for the API / tests)."""
        content = " ".join(str(content).split()).strip()
        if not content or len(content) > MAX_ENTRY_LENGTH:
            raise ValueError("متن حافظه باید بین ۱ تا ۲۰۰۰ نویسه باشد.")
        ok, reason = validate_entry(content)
        if not ok:
            raise ValueError(reason + ".")
        now = int(time.time())
        try:
            with self._tx() as db:
                memory_id, outcome, category = self._upsert(db, content, now)
        except sqlite3.Error as exc:
            log.exception("memory write failed")
            raise MemoryStoreError(f"نوشتن در پایگاه حافظه ناموفق بود: {exc}") from exc
        self._verify([memory_id])
        return {"id": memory_id, "content": content, "updated_at": now, "category": category,
                "duplicate": outcome == "duplicate", "updated": outcome == "updated"}

    # ------------------------------------------------------------ reading
    def list(self):
        with self._read() as db:
            rows = db.execute("SELECT id, content, category, created_at, updated_at FROM memories "
                              "ORDER BY updated_at DESC, created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def delete_id(self, memory_id):
        with self._tx() as db:
            return db.execute("DELETE FROM memories WHERE id=?", (memory_id,)).rowcount

    def forget(self, query=""):
        with self._tx() as db:
            if not query:
                count = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                db.execute("DELETE FROM memories")
                return count
            needle = self._normalize(query)
            rows = db.execute("SELECT id, normalized FROM memories").fetchall()
            # a stored fact is also removed when the user quotes it inside a longer sentence,
            # but only for reasonably long facts so that short ones are never matched by accident
            ids = [row["id"] for row in rows
                   if needle in row["normalized"]
                   or (len(row["normalized"]) >= 8 and row["normalized"] in needle)]
            if ids:
                db.executemany("DELETE FROM memories WHERE id=?", [(item,) for item in ids])
            return len(ids)

    def relevant(self, query="", max_items=30):
        """The memories worth showing the model for ``query``.

        Always: persistent instructions and preferences (they shape every answer).  Then the entries that share
        words with the query (a category the query clearly asks about gets a boost).  When nothing matches,
        the most recently used entries fill the rest.  Order inside the result: instructions first.
        """
        items = self.list()
        if not items:
            return []
        query_tokens = _tokens(query) if query else set()
        hinted = {name for name, pattern in _QUERY_HINTS.items() if query and pattern.search(query)}
        scored = []
        for index, item in enumerate(items):
            score = _overlap(query_tokens, _tokens(item["content"])) if query_tokens else 0
            if item["category"] in hinted:
                score += 3
            scored.append((score, -index, item))
        chosen, seen = [], set()

        def take(candidates, limit):
            for item in candidates:
                if len(chosen) >= max_items or limit <= 0:
                    return
                if item["id"] not in seen:
                    seen.add(item["id"])
                    chosen.append(item)
                    limit -= 1

        take((item for item in items if item["category"] == "instruction"), 8)
        take((item for item in items if item["category"] == "preference"), 5)
        matched = [entry[2] for entry in sorted(scored, key=lambda entry: (entry[0], entry[1]), reverse=True) if entry[0] > 0]
        take(matched, max_items)
        if not matched:                                     # nothing relates to the query: show the newest
            take(items, max_items)
        order = {name: index for index, name in enumerate(CATEGORIES)}
        return sorted(chosen, key=lambda item: order.get(item["category"], len(order)))

    def prompt_context(self, query="", max_items=30, max_chars=4500):
        groups, total = {}, 0
        for item in self.relevant(query, max_items=max_items):
            line = "- " + item["content"]
            if total + len(line) > max_chars:
                continue                                    # a shorter, later entry may still fit
            groups.setdefault(item["category"], []).append(line)
            total += len(line)
        out = []
        for category in CATEGORIES:
            if groups.get(category):
                out.append(f"[{CATEGORY_LABELS[category]}]")
                out.extend(groups[category])
        return "\n".join(out)[:max_chars] if out else ""


# ---------------------------------------------------------------- user-facing wording
def describe_save_report(report):
    """The chat reply for ``save_text``.  Claims «saved» only for what was committed and verified."""
    saved, updated, dups = report["saved"], report["updated"], report["duplicates"]
    rejected = report["rejected"]
    if len(saved) + len(updated) + len(dups) + len(rejected) == 1:
        if saved:
            return "این مورد را در حافظهٔ محلی ذخیره کردم و در گفت‌وگوهای بعدی هم از آن استفاده می‌کنم."
        if updated:
            return "این مورد با نسخهٔ قبلی‌اش یکی بود؛ مقدار قدیمی را با مقدار جدید به‌روزرسانی کردم."
        if dups:
            return "این مورد از قبل در حافظه بود؛ متن آن را به‌روزرسانی کردم."
    lines = []
    if saved:
        lines.append(f"{len(saved)} مورد جدید را در حافظهٔ محلی ذخیره کردم.")
    if updated:
        lines.append(f"{len(updated)} مورد قبلی را به‌روزرسانی کردم.")
    if dups:
        lines.append(f"{len(dups)} مورد از قبل در حافظه بود.")
    if rejected:
        lines.append(f"{len(rejected)} مورد را ذخیره نکردم:")
        lines.extend(f"• «{text[:80]}» — {reason}" for text, reason in rejected[:8])
        if len(rejected) > 8:
            lines.append(f"… و {len(rejected) - 8} مورد دیگر")
    if not (saved or updated or dups):
        lines.insert(0, "چیزی ذخیره نشد.")
    return "\n".join(lines)


def format_listing(items):
    if not items:
        return "هنوز چیزی در حافظهٔ بلندمدت ذخیره نشده است."
    groups = {}
    for item in items:
        groups.setdefault(item.get("category") or "general", []).append(item["content"])
    lines = ["مواردی که در حافظهٔ محلی ذخیره شده‌اند:"]
    for category in CATEGORIES:
        if groups.get(category):
            lines.append(f"\n{CATEGORY_LABELS[category]}:")
            lines.extend(f"• {content}" for content in groups[category])
    return "\n".join(lines)
