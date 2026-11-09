"""Explicit, local-only long-term memory for LocalChat."""
import re
import sqlite3
import time
import uuid
from pathlib import Path

_SAVE_TRIGGER = (
    r"(?:یادت\s*(?:باشه|بمونه|نره)|به\s*خاطر\s*(?:بسپار|داشته\s*باش))"
)
SAVE_PATTERNS = (
    # «یادت باشه ...» / «ذخیره کن: ...»  (the bare «ذخیره کن» needs a colon so that
    # ordinary requests like «این فایل را ذخیره کن» are never treated as memory commands)
    re.compile(
        r"^\s*(?:لطفاً\s*)?(?:این(?:و|\s*(?:رو|را))?\s*)?(?:"
        + _SAVE_TRIGGER + r"\s*[:：،\-]?\s*"
        r"|(?:در|تو)\s*حافظه(?:‌ات)?\s*نگه\s*دار\s*[:：،\-]?\s*"
        r"|(?:ذخیره|یادداشت)\s*کن\s*[:：]\s*)(.+)$", re.I | re.S),
    # «... ، اینو یادت باشه»
    re.compile(
        r"^\s*(.+?)\s*[,،]?\s*(?:این(?:و|\s*(?:رو|را))?\s*)?" + _SAVE_TRIGGER + r"\s*[.!؟]?\s*$",
        re.I | re.S),
)
FORGET_ALL = re.compile(r"^\s*(?:همه(?:ی|ٔ)?\s*)?(?:حافظه|یادداشت(?:ها)?)\s*(?:رو|را)?\s*(?:پاک|حذف)\s*کن\s*[.!؟]?\s*$", re.I)
# «فراموش کن ...» always; «حذف/پاک کن» only with a colon («حذف کن خط ۳ را» is a normal request)
FORGET_ONE = re.compile(
    r"^\s*(?:این(?:و|\s*(?:رو|را))?\s*)?(?:فراموش\s*کن\s*[:：،\-]?|(?:حذف|پاک)\s*کن\s*[:：])\s*(.+)$",
    re.I | re.S)
LIST_MEMORY = re.compile(r"(?:چه\s*چیز(?:هایی)?\s*(?:رو|را)?\s*(?:یادت|به\s*خاطرت)\s*هست|حافظه(?:ات)?\s*(?:رو|را)?\s*(?:نشون|نمایش)\s*بده|چه\s*چیز(?:هایی)?\s*ذخیره\s*کردی)", re.I)
MAX_COMMAND_LENGTH = 2200

def parse_memory_command(text):
    if not isinstance(text, str):
        return None
    value = text.strip()
    # pasted code / long documents are never memory commands
    if "```" in value or len(value) > MAX_COMMAND_LENGTH or value.count("\n") > 8:
        return None
    if FORGET_ALL.fullmatch(value):
        return ("forget_all", "")
    match = FORGET_ONE.match(value)
    if match:
        return ("forget", match.group(1).strip())
    if LIST_MEMORY.search(value):
        return ("list", "")
    for pattern in SAVE_PATTERNS:
        match = pattern.match(value)
        if match:
            content = match.group(1).strip()
            content = re.sub(r"^(?:که|اینکه)\s+", "", content)
            content = re.sub(r"\s*(?:لطفاً\s*)?(?:این(?:و|\s*(?:رو|را))?\s*)?" + _SAVE_TRIGGER + r"\s*$", "", content).strip(" \t\r\n،,:：-")
            if content and len(content) <= 2000:
                return ("save", content)
    return None

class PersistentMemory:
    def __init__(self, database_path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, content TEXT NOT NULL, normalized TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
            )""")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _normalize(value):
        return " ".join(value.casefold().split())

    def add(self, content):
        content = " ".join(str(content).split()).strip()
        if not content or len(content) > 2000:
            raise ValueError("متن حافظه باید بین ۱ تا ۲۰۰۰ نویسه باشد.")
        now = int(time.time())
        memory_id = str(uuid.uuid4())
        normalized = self._normalize(content)
        with self._connect() as db:
            row = db.execute("SELECT id FROM memories WHERE normalized=?", (normalized,)).fetchone()
            if row:
                db.execute("UPDATE memories SET content=?, updated_at=? WHERE id=?", (content, now, row["id"]))
                return {"id": row["id"], "content": content, "updated_at": now, "duplicate": True}
            db.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?)", (memory_id, content, normalized, now, now))
        return {"id": memory_id, "content": content, "updated_at": now, "duplicate": False}

    def list(self):
        with self._connect() as db:
            rows = db.execute("SELECT id, content, created_at, updated_at FROM memories ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def delete_id(self, memory_id):
        with self._connect() as db:
            cursor = db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            return cursor.rowcount

    def forget(self, query=""):
        with self._connect() as db:
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

    def prompt_context(self, max_items=30, max_chars=4500):
        items = self.list()[:max_items]
        lines = []
        total = 0
        for item in items:
            line = "- " + item["content"]
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)
