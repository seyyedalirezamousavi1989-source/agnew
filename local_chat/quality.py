"""Quality checks for the output of a long task (translation / rewriting / analysis chunks).

``check_chunk`` compares one model answer with its source chunk and reports issues; ``severity == "error"``
means the chunk must be redone (the task manager retries it), ``"warn"`` is shown to the user but kept.
``check_assembled`` looks across chunks (missing, duplicated, out-of-order output).
"""
import re

try:
    from .docwriter import ARABIC, FOOT_DEF, FOOT_REF
except ImportError:                     # pragma: no cover
    from docwriter import ARABIC, FOOT_DEF, FOOT_REF

LATIN = re.compile(r"[A-Za-z]")
_REFUSAL = re.compile(r"(?:as an ai|i cannot|i can't (?:help|assist|translate)|متأسفم|نمی[‌ ]?توانم|sorry, but)", re.I)
_PREAMBLE = re.compile(r"^\s*(?:sure|here(?:'s| is)|certainly|of course|البته|بله|حتماً|حتما|ترجمه(?:ی|‌ی)? (?:متن|شما)|در ادامه)[^\n]{0,80}[:：]\s*\n", re.I)
_SENTENCE_END = re.compile(r"[.!?؟…:؛»\")\]}]\s*$")


class Issue:
    def __init__(self, code, message, severity="error"):
        self.code, self.message, self.severity = code, message, severity

    def __repr__(self):
        return f"Issue({self.code}, {self.severity})"

    def to_dict(self):
        return {"code": self.code, "message": self.message, "severity": self.severity}


def _counts(text):
    lines = text.split("\n")
    return {
        "headings": sum(1 for ln in lines if re.match(r"^#{1,6}\s+\S", ln)),
        "fences": sum(1 for ln in lines if ln.strip().startswith(("```", "~~~"))),
        "table_rows": sum(1 for ln in lines if ln.strip().startswith("|") and not re.match(r"^\s*\|?\s*:?-{2,}", ln.strip())),
        "footnote_refs": len(FOOT_REF.findall("\n".join(ln for ln in lines if not FOOT_DEF.match(ln)))),
        "footnote_defs": sum(1 for ln in lines if FOOT_DEF.match(ln)),
    }


def strip_preamble(text):
    """Remove «Here is the translation:» style chatter and surrounding code fences the model sometimes adds."""
    text = _PREAMBLE.sub("", text, count=1).strip()
    match = re.fullmatch(r"```(?:markdown|md|text)?\n(.*)\n```", text, re.S)
    return (match.group(1) if match else text).strip()


def _shingles(text, size=8):
    words = re.findall(r"\w+", text.lower())
    return [" ".join(words[i:i + size]) for i in range(max(0, len(words) - size + 1))]


def check_chunk(source, output, task="translate", target="fa", glossary=None, ratio=(0.35, 3.0)):
    """Issues for one answer.  ``task``: translate | rewrite | summarize | other."""
    issues = []
    out = (output or "").strip()
    src = (source or "").strip()
    if not out:
        return [Issue("empty", "خروجی خالی است.")]
    if _REFUSAL.search(out[:200]) and not _REFUSAL.search(src[:2000]):
        issues.append(Issue("refusal", "مدل به‌جای انجام کار، عذرخواهی/امتناع کرده است."))
    if task in ("translate", "rewrite"):
        low, high = ratio
        measured = len(out) / max(1, len(src))
        if measured < low:
            issues.append(Issue("too_short", f"خروجی فقط {measured:.0%} طول منبع است؛ بخشی جا افتاده."))
        elif measured > high:
            issues.append(Issue("too_long", f"خروجی {measured:.1f} برابر منبع است؛ احتمالاً توضیح اضافه یا تکرار.", "warn"))
        if _SENTENCE_END.search(src) and not _SENTENCE_END.search(out) and not out.rstrip().endswith(("|", "-")):
            issues.append(Issue("truncated", "خروجی ناتمام به نظر می‌رسد (جمله‌ی آخر کامل نیست)."))
    if task == "translate":
        src_latin, src_arabic = len(LATIN.findall(src)), len(ARABIC.findall(src))
        out_latin, out_arabic = len(LATIN.findall(out)), len(ARABIC.findall(out))
        total_out = max(1, out_latin + out_arabic)
        if target == "fa" and src_latin > src_arabic and out_arabic / total_out < 0.35 and len(out) > 200:
            issues.append(Issue("untranslated", "بخش زیادی از خروجی هنوز ترجمه نشده است."))
        if target == "en" and src_arabic > src_latin and out_latin / total_out < 0.35 and len(out) > 200:
            issues.append(Issue("untranslated", "Most of the output is still in the source language."))
    # a model stuck in a loop repeats phrases far more often than the source itself does
    def peak(text):
        found = {}
        for shingle in _shingles(text):
            found[shingle] = found.get(shingle, 0) + 1
        return max(found.values()) if found else 0

    out_peak, src_peak = peak(out), peak(src)
    if out_peak >= 4 and out_peak > 2 * src_peak + 2:
        issues.append(Issue("loop", "بخشی از خروجی بارها تکرار شده (گیر کردن مدل)."))
    if task in ("translate", "rewrite"):
        a, b = _counts(src), _counts(out)
        for key, label in (("headings", "تیتر"), ("fences", "بلوک کد"), ("table_rows", "ردیف جدول"),
                           ("footnote_refs", "ارجاع پاورقی"), ("footnote_defs", "متن پاورقی")):
            if a[key] != b[key]:
                issues.append(Issue("structure", f"تعداد {label} در خروجی ({b[key]}) با منبع ({a[key]}) برابر نیست."))
        if b["fences"] % 2:
            issues.append(Issue("structure", "بلوک کد بسته نشده است."))
    for term, wanted in (glossary or {}).items():
        if re.search(re.escape(term), src, re.I) and wanted not in out:
            issues.append(Issue("glossary", f"«{term}» باید «{wanted}» ترجمه می‌شد.", "warn"))
    return issues


def has_errors(issues):
    return any(i.severity == "error" for i in issues)


def check_assembled(rows, min_overlap=0.8):
    """Cross-chunk checks over ``chunk_rows`` (ordered): missing / failed output, duplicates, order markers."""
    problems = []
    expected = 0
    for row in rows:
        if row["idx"] != expected:
            problems.append(Issue("order", f"شمارهٔ بخش‌ها پیوسته نیست (انتظار {expected}، یافت {row['idx']})."))
            expected = row["idx"]
        expected += 1
        if row["status"] != "done" or not (row.get("output") or "").strip():
            problems.append(Issue("missing", f"بخش {row['idx'] + 1} خروجی ندارد ({row['status']})."))
    done = [r for r in rows if r["status"] == "done" and (r.get("output") or "").strip()]
    sets = [set(_shingles(r["output"], 6)) for r in done]
    for i in range(len(done)):
        for j in range(i + 1, min(len(done), i + 6)):                    # duplicates are produced by neighbours
            a, b = sets[i], sets[j]
            if len(a) > 12 and len(b) > 12 and len(a & b) / min(len(a), len(b)) >= min_overlap:
                problems.append(Issue("duplicate", f"بخش‌های {done[i]['idx'] + 1} و {done[j]['idx'] + 1} تقریباً یکسان‌اند."))
    return problems
