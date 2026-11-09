"""Shared SQLite plumbing for every LocalChat database (memory, agent tasks, ...).

One place for the things that make a local database trustworthy:

* explicit transactions (``BEGIN IMMEDIATE`` .. ``COMMIT`` / ``ROLLBACK``) – a batch is all-or-nothing;
* connections are always closed (an open handle keeps a file locked on Windows);
* ``synchronous=FULL`` + a busy timeout – a power cut after «saved» does not lose the row;
* versioned schema (``PRAGMA user_version``) with an automatic, pruned backup before any migration;
* ``integrity_report`` – read-only health check used by the audit and by the tests.
"""
import logging
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("localchat.db")


class Database:
    def __init__(self, path, wal=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.wal = wal
        self.last_backup = None

    def connect(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=15000")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        if self.wal:
            db.execute("PRAGMA journal_mode=WAL")
        return db

    @contextmanager
    def tx(self):
        """One real transaction on a connection that is always closed."""
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            else:
                db.execute("COMMIT")
        finally:
            db.close()

    @contextmanager
    def read(self):
        db = self.connect()
        try:
            yield db
        finally:
            db.close()

    def version(self):
        with self.read() as db:
            return db.execute("PRAGMA user_version").fetchone()[0]

    def has_table(self, name):
        with self.read() as db:
            return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

    def backup(self, tag, keep=3):
        """Copy the database file next to itself before a risky change; older backups are pruned."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        target = self.path.with_name(f"{self.path.name}.bak-{tag}-{int(time.time())}")
        shutil.copy2(self.path, target)
        backups = sorted(self.path.parent.glob(self.path.name + ".bak-*"), key=lambda item: item.stat().st_mtime)
        for old in backups[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass
        self.last_backup = target
        return target

    def integrity_report(self):
        """Read-only health summary: integrity_check, foreign-key violations, tables with row counts and indexes."""
        with self.read() as db:
            report = {
                "path": str(self.path),
                "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
                "user_version": db.execute("PRAGMA user_version").fetchone()[0],
                "integrity": db.execute("PRAGMA integrity_check").fetchone()[0],
                "foreign_key_violations": len(db.execute("PRAGMA foreign_key_check").fetchall()),
                "tables": {},
            }
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
                name = row["name"]
                report["tables"][name] = {
                    "rows": db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0],
                    "indexes": [i["name"] for i in db.execute(f'PRAGMA index_list("{name}")').fetchall()],
                }
        report["ok"] = report["integrity"] == "ok" and report["foreign_key_violations"] == 0
        return report
