"""Task processors and result assembly on top of ``tasks.py`` + ``docengine.py`` + ``quality.py``.

A processor is built from a plain ``generate(messages) -> str`` callable (the server passes one that talks to
Ollama; tests pass a fake), so nothing here depends on a running model.

    translate: heading path + glossary + running summary → one chunk → quality gate → (retry with feedback)
    summarize: one short summary per chunk (map); ``reduce_summaries`` merges them hierarchically
    qa:        keyword retrieval over chunks – only the best few go to the model, never the whole document
"""
import re

try:
    from . import docengine, docwriter, quality, tasks
    from .context_manager import estimate_tokens
    from .memory import _tokens, _overlap
except ImportError:                     # pragma: no cover
    import docengine, docwriter, quality, tasks
    from context_manager import estimate_tokens
    from memory import _tokens, _overlap

TERMS_MARK = "<<TERMS>>"
LANGUAGE_NAMES = {"fa": "فارسی", "en": "انگلیسی", "ar": "عربی", "de": "آلمانی", "fr": "فرانسوی", "es": "اسپانیایی",
                  "tr": "ترکی", "ru": "روسی", "zh": "چینی", "ja": "ژاپنی"}


class QualityError(Exception):
    def __init__(self, issues):
        self.issues = issues
        super().__init__("; ".join(f"{i.code}: {i.message}" for i in issues if i.severity == "error")[:600])


def parse_terms(text):
    """Split ``<<TERMS>>`` footer (``term = translation`` lines) off a model answer."""
    if TERMS_MARK not in text:
        return text.strip(), {}
    body, footer = text.split(TERMS_MARK, 1)
    terms = {}
    for line in footer.strip().splitlines():
        match = re.match(r"^\s*[-*•]?\s*(.+?)\s*(?:=|→|->|:)\s*(.+?)\s*$", line)
        if match and len(match.group(1)) < 60 and len(match.group(2)) < 60:
            terms[match.group(1)] = match.group(2)
    return body.strip(), terms


def relevant_glossary(glossary, text, limit=40):
    low = text.lower()
    hits = {t: v for t, v in glossary.items() if t.lower() in low}
    return dict(list(hits.items())[:limit])


def carry_summary(markdown, limit=420):
    """Cheap, deterministic hand-over for the next chunk: the last headings and the tail of the translated text."""
    headings = [ln.lstrip("# ").strip() for ln in markdown.split("\n") if re.match(r"^#{1,6}\s", ln)][-2:]
    plain = re.sub(r"\s+", " ", re.sub(r"[#`*|>]|\[\^\w+\]:?.*", " ", markdown)).strip()
    tail = plain[-limit:]
    cut = max(tail.find(". "), tail.find("؟ "), tail.find("! "))
    tail = tail[cut + 2:] if 0 <= cut < len(tail) - 40 else tail
    return ((" / ".join(headings) + " — ") if headings else "") + tail


def translate_messages(chunk, ctx, target="fa", style="", feedback=""):
    language = LANGUAGE_NAMES.get(target, target)
    glossary = relevant_glossary(ctx.glossary(), chunk.text)
    previous = ctx.previous_summary()
    system = [f"تو مترجم حرفه‌ای هستی. متن را به {language} روان و دقیق ترجمه کن.",
              "قواعد: ساختار Markdown را عیناً نگه دار (تیترها با #، جدول‌ها، بلوک‌های کد، فهرست‌ها و ارجاع‌های [^n]). "
              "کد و نام‌های فنی را ترجمه نکن. چیزی کم یا زیاد نکن، توضیح یا مقدمه ننویس و فقط ترجمه را بده."]
    if style:
        system.append("سبک: " + style)
    if glossary:
        system.append("واژه‌نامهٔ اجباری:\n" + "\n".join(f"- {k} = {v}" for k, v in glossary.items()))
    system.append(f"اگر اصطلاح یا نام خاص مهمی برایش معادل ثابت انتخاب کردی، بعد از ترجمه یک خط {TERMS_MARK} و زیرش خطوط «اصطلاح = معادل» بنویس (حداکثر ۱۰ مورد).")
    user = []
    if chunk.heading_path:
        user.append("[موقعیت در سند: " + " › ".join(chunk.heading_path) + "]")
    if previous:
        user.append("[خلاصهٔ بخش قبلی برای هماهنگی لحن و اصطلاحات — ترجمه‌اش نکن: " + previous + "]")
    if feedback:
        user.append("[نکتهٔ اصلاحی از تلاش قبلی: " + feedback + "]")
    user.append("متن برای ترجمه:\n\n" + chunk.text)
    return [{"role": "system", "content": "\n".join(system)}, {"role": "user", "content": "\n\n".join(user)}]


def make_translate_processor(generate, target="fa", style=""):
    state = {}

    def processor(chunk, ctx):
        ctx.check()
        key = (ctx.task["id"], chunk.index)
        feedback = state.pop(key, "")
        raw = generate(translate_messages(chunk, ctx, target, style, feedback))
        ctx.check()
        text, terms = parse_terms(quality.strip_preamble(raw))
        issues = quality.check_chunk(chunk.text, text, "translate", target, ctx.glossary())
        if quality.has_errors(issues):
            state[key] = " ".join(i.message for i in issues if i.severity == "error")[:300]
            raise QualityError(issues)
        if terms:
            ctx.remember_terms(terms)
        return tasks.ChunkResult(text, summary=carry_summary(text),
                                 meta={"issues": [i.to_dict() for i in issues], "terms": terms})
    return processor


def make_summarize_processor(generate, language="fa", words=120):
    def processor(chunk, ctx):
        ctx.check()
        where = (" › ".join(chunk.heading_path) + "\n") if chunk.heading_path else ""
        text = generate([
            {"role": "system", "content": f"متن را در حدود {words} کلمه به زبان {LANGUAGE_NAMES.get(language, language)} خلاصه کن. "
                                          "نکات، نام‌ها، عددها و تصمیم‌ها را نگه دار و چیزی از خودت اضافه نکن."},
            {"role": "user", "content": where + chunk.text}])
        text = quality.strip_preamble(text)
        if not text.strip():
            raise QualityError([quality.Issue("empty", "خلاصه خالی است.")])
        return tasks.ChunkResult(text, summary=text[-300:])
    return processor


def reduce_summaries(generate, summaries, max_tokens=1800, language="fa"):
    """Merge many chunk summaries into one: summarize groups that fit the window, repeat until one remains."""
    level = [s.strip() for s in summaries if s and s.strip()]
    while len(level) > 1:
        groups, current, used = [], [], 0
        for item in level:
            cost = estimate_tokens(item)
            if current and used + cost > max_tokens:
                groups.append(current)
                current, used = [], 0
            current.append(item)
            used += cost
        groups.append(current)
        if len(groups) == len(level):                    # nothing could be merged (items bigger than the window)
            break
        level = [generate([
            {"role": "system", "content": f"این خلاصه‌های پشت‌سرهم را به یک خلاصهٔ منسجم به زبان {LANGUAGE_NAMES.get(language, language)} تبدیل کن."},
            {"role": "user", "content": "\n\n".join(group)}]) if len(group) > 1 else group[0] for group in groups]
    return level[0] if level else ""


def retrieve(chunks, question, top_k=4, max_tokens=3000):
    """Best chunks for a question by word overlap (+ heading words); only these are ever sent to the model."""
    query = _tokens(question)
    scored = []
    for chunk in chunks:
        haystack = _tokens(" ".join(chunk.heading_path) + " " + chunk.text)
        heads = _tokens(" ".join(chunk.heading_path))
        scored.append((_overlap(query, haystack) + 2 * _overlap(query, heads), -chunk.index, chunk))
    picked, used = [], 0
    for score, _, chunk in sorted(scored, key=lambda s: (s[0], s[1]), reverse=True):
        if score <= 0 or len(picked) >= top_k or used + chunk.tokens > max_tokens:
            continue
        picked.append(chunk)
        used += chunk.tokens
    return sorted(picked, key=lambda c: c.index)


def answer_question(generate, doc, question, num_ctx=8192, num_predict=1024):
    size = docengine.plan_chunk_tokens(num_ctx, num_predict, task="qa")
    chunks = docengine.chunk_document(doc, min(size // 3, 1200))
    best = retrieve(chunks, question, max_tokens=max(600, size))
    if not best:
        return "در سند چیزی مرتبط با این پرسش پیدا نشد.", []
    context = "\n\n---\n\n".join(("[" + " › ".join(c.heading_path) + "]\n" if c.heading_path else "") + c.text for c in best)
    answer = generate([
        {"role": "system", "content": "فقط بر اساس بخش‌های داده‌شدهٔ سند پاسخ بده. اگر پاسخ در آن‌ها نیست بگو در سند نیامده."},
        {"role": "user", "content": f"بخش‌های سند:\n\n{context}\n\nپرسش: {question}"}])
    return answer, [c.index for c in best]


def project_overview(doc):
    """Codebase inspection without loading code: file list + sizes + per-file outline (functions/classes)."""
    files, current = [], None
    for block in doc.blocks:
        if block.kind == "heading" and block.text.startswith("File: "):
            current = {"path": block.text[6:], "tokens": 0, "symbols": []}
            files.append(current)
        elif current and block.kind == "code":
            current["tokens"] += estimate_tokens(block.text)
            current["symbols"] += re.findall(r"^\s*(?:def|class|function|interface|struct|fn|func)\s+(\w+)", block.text, re.M)[:25]
    return files


# ----------------------------------------------------------------------------- assembly
def assemble(manager, task_id, fmt="md", title="", allow_partial=False):
    """Final file for a task: ordered chunk outputs → one document → TXT / MD / DOCX / PDF.

    Refuses to build from an incomplete task (a book with a missing chapter) unless ``allow_partial`` is set,
    in which case missing chunks are replaced by an explicit marker so the gap can never go unnoticed.
    """
    rows = manager.chunk_rows(task_id)
    problems = quality.check_assembled(rows)
    blocking = [p for p in problems if p.code in ("missing", "order", "duplicate")]
    if blocking and not allow_partial:
        raise tasks.TaskError("خروجی کامل نیست: " + "؛ ".join(p.message for p in blocking[:4]))
    parts = []
    for row in rows:
        if row["status"] == "done" and (row.get("output") or "").strip():
            parts.append(row["output"].strip())
        else:
            parts.append(f"> ⚠️ [بخش {row['idx'] + 1} ترجمه/پردازش نشده است]")
    task = manager.get(task_id)
    markdown = "\n\n".join(parts)
    return docwriter.render(markdown, fmt, title or task["title"]), {"problems": [p.to_dict() for p in problems],
                                                                        "chunks": len(rows)}
