"""Document engine: structured extraction, token-aware chunking and integrity checks for large files.

Pure and model-free.  The long-running task layer (``tasks.py``) feeds the chunks to a model; this module only
guarantees that what goes in is complete, ordered and measurable.

Pipeline
    bytes/path --load_document--> Document(blocks, footnotes)      PDF / DOCX / TXT / MD / ZIP (+ code projects)
    Document --chunk_document(max_tokens)--> [Chunk]               boundaries at headings / paragraph ends
    [Chunk] --verify_chunks--> report                              no gaps, no duplicates, no reordering

A block is the smallest unit that is never reordered: heading, paragraph, table, code.  A block bigger than the
chunk size is split into numbered parts (sentences / table rows with the header repeated / code lines), so the
sequence ``(block id, part)`` is a total order that ``verify_chunks`` can check.  The model-facing text is
Markdown (``#`` headings, pipe tables, fenced code, ``[^n]`` footnote markers), which the writer can parse back.
"""
import hashlib
import io
import re
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

try:                                    # package import (server) or flat import (tests / launcher)
    from . import files as _files
    from .context_manager import estimate_tokens
except ImportError:                     # pragma: no cover
    import files as _files
    from context_manager import estimate_tokens

MAX_DOC_BYTES = 256 * 1024 * 1024
MAX_PDF_PAGES = 3000
MAX_ZIP_ENTRIES = 20000
MAX_ZIP_DECLARED = 1024 * 1024 * 1024
MAX_ZIP_FILE_BYTES = 4 * 1024 * 1024
MAX_ZIP_DEPTH = 1

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
TEXT_FORMATS = {".txt", ".text", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv"}


class DocumentError(Exception):
    pass


# ----------------------------------------------------------------------------- model
class Block:
    """heading | paragraph | table | code"""

    __slots__ = ("id", "kind", "text", "level", "rows", "lang", "source", "page")

    def __init__(self, id, kind, text="", level=0, rows=None, lang="", source="", page=0):
        self.id, self.kind, self.text, self.level = id, kind, text, level
        self.rows, self.lang, self.source, self.page = rows, lang, source, page

    def render(self):
        if self.kind == "heading":
            return "#" * max(1, min(6, self.level)) + " " + self.text
        if self.kind == "table":
            return render_table(self.rows or [])
        if self.kind == "code":
            return f"```{self.lang}\n{self.text}\n```"
        return self.text

    def to_dict(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}


class Document:
    def __init__(self, name, kind, blocks, footnotes=None, notes=None):
        self.name, self.kind = name, kind
        self.blocks = blocks
        self.footnotes = footnotes or {}      # id -> text
        self.notes = notes or []              # extraction warnings for the user

    def outline(self):
        return [(b.level, b.text, b.id) for b in self.blocks if b.kind == "heading"]

    def markdown(self):
        return "\n\n".join(b.render() for b in self.blocks)

    def stats(self):
        kinds = {}
        for block in self.blocks:
            kinds[block.kind] = kinds.get(block.kind, 0) + 1
        return {"blocks": len(self.blocks), "kinds": kinds, "footnotes": len(self.footnotes),
                "tokens": sum(estimate_tokens(b.render()) for b in self.blocks)}


def render_table(rows):
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [[(c or "").replace("|", "\\|").replace("\n", " ").strip() for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(norm[0]) + " |", "|" + "|".join(" --- " for _ in range(width)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in norm[1:]]
    return "\n".join(lines)


def _new(blocks, kind, text="", **kw):
    block = Block(len(blocks), kind, text, **kw)
    blocks.append(block)
    return block


# ----------------------------------------------------------------------------- heading heuristics
_H1 = re.compile(r"^(?:فصل|باب|بخش\s+(?:اول|دوم|سوم|چهارم|پنجم)|مقدمه|پیشگفتار|سخن\s*آغازین|نتیجه[\u200c ]?گیری|پیوست|فهرست|"
                 r"chapter|part|prologue|epilogue|preface|introduction|conclusion|appendix|contents)\b", re.I)
_NUMBERED = re.compile(r"^(?P<num>[0-9۰-۹]+(?:\.[0-9۰-۹]+){0,3})[.)]?\s+(?P<title>\S.*)$")
_SENTENCE_END = re.compile(r"[.!؟?؛:،,]$")


def _heading_level(line, isolated):
    """Level of a plain-text line that looks like a heading, else 0.  Only isolated short lines qualify."""
    text = line.strip()
    if not isolated or not text or len(text) > 90 or _SENTENCE_END.search(text):
        return 0
    if _H1.match(text):
        return 1
    match = _NUMBERED.match(text)
    if match and len(text.split()) <= 12:
        return min(4, max(2, match.group("num").count(".") + 1))
    return 0


def _paragraphs_from_text(text, blocks, source="", page=0, reflow=False):
    """Blank-line separated paragraphs, with isolated heading-looking lines promoted to headings."""
    chunks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    for raw in chunks:
        lines = [ln.rstrip() for ln in raw.split("\n") if ln.strip()]
        if not lines:
            continue
        if len(lines) == 1:
            level = _heading_level(lines[0], True)
            if level:
                _new(blocks, "heading", lines[0].strip(), level=level, source=source, page=page)
                continue
        body = " ".join(ln.strip() for ln in lines) if reflow else "\n".join(lines)
        _new(blocks, "paragraph", body.strip(), source=source, page=page)


# ----------------------------------------------------------------------------- markdown / txt
_FENCE = re.compile(r"^\s*(```|~~~)\s*([\w+#.-]*)\s*$")
_ATX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _split_row(line):
    cells = re.split(r"(?<!\\)\|", line.strip())
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return [c.replace("\\|", "|").strip() for c in cells]


def parse_markdown(text, source=""):
    """Markdown → blocks (also used to read a model's translated chunk back into structure)."""
    blocks = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i, para = 0, []

    def flush():
        if para:
            _new(blocks, "paragraph", "\n".join(para).strip(), source=source)
            para.clear()

    while i < len(lines):
        line = lines[i]
        fence = _FENCE.match(line)
        if fence:
            flush()
            marker, lang, code = fence.group(1), fence.group(2), []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(marker):
                code.append(lines[i])
                i += 1
            _new(blocks, "code", "\n".join(code), lang=lang, source=source)
            i += 1
            continue
        atx = _ATX.match(line)
        if atx:
            flush()
            _new(blocks, "heading", atx.group(2).strip(), level=len(atx.group(1)), source=source)
            i += 1
            continue
        if "|" in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]) and "-" in lines[i + 1]:
            flush()
            rows = [_split_row(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            _new(blocks, "table", rows=rows, source=source)
            continue
        if i + 1 < len(lines) and line.strip() and re.fullmatch(r"=+|-+", lines[i + 1].strip() or "x") and not para:
            _new(blocks, "heading", line.strip(), level=1 if lines[i + 1].strip()[0] == "=" else 2, source=source)
            i += 2
            continue
        if not line.strip():
            flush()
        else:
            para.append(line.rstrip())
        i += 1
    flush()
    return blocks


def load_text(data, name):
    text = _files.decode_text(data)
    if text is None:
        raise DocumentError("فایل متنی نیست.")
    ext = PurePosixPath(name.lower()).suffix
    if ext in (".md", ".markdown"):
        return Document(name, "markdown", parse_markdown(text, name))
    blocks = []
    _paragraphs_from_text(text, blocks, name)
    return Document(name, "text", blocks)


# ----------------------------------------------------------------------------- docx
def _para_text(paragraph):
    parts = []
    for node in paragraph.iter():
        tag = node.tag
        if tag == W + "t" and node.text:
            parts.append(node.text)
        elif tag == W + "tab":
            parts.append("\t")
        elif tag in (W + "br", W + "cr"):
            parts.append("\n")
        elif tag == W + "footnoteReference":
            parts.append(f"[^{node.get(W + 'id')}]")
    return "".join(parts)


def _heading_from_style(paragraph):
    props = paragraph.find(f"{W}pPr")
    if props is None:
        return 0
    style = props.find(f"{W}pStyle")
    if style is not None:
        value = (style.get(W + "val") or "").lower().replace(" ", "")
        match = re.fullmatch(r"(?:heading|عنوان)(\d)", value)
        if match:
            return int(match.group(1))
        if value == "title":
            return 1
    outline = props.find(f"{W}outlineLvl")
    if outline is not None:
        try:
            return int(outline.get(W + "val")) + 1
        except (TypeError, ValueError):
            return 0
    return 0


def load_docx(data, name):
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        root = ET.fromstring(_files._read_member(archive, "word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        raise DocumentError("فایل Word خراب یا نامعتبر است.")
    footnotes = {}
    if "word/footnotes.xml" in archive.namelist():
        try:
            for note in ET.fromstring(_files._read_member(archive, "word/footnotes.xml")).iter(W + "footnote"):
                note_id = note.get(W + "id")
                if note_id in (None, "-1", "0") or note.get(W + "type") in ("separator", "continuationSeparator"):
                    continue
                text = " ".join(_para_text(p).strip() for p in note.iter(W + "p")).strip()
                if text:
                    footnotes[note_id] = text
        except ET.ParseError:
            pass
    blocks, body = [], root.find(W + "body")
    for child in (body if body is not None else []):
        if child.tag == W + "p":
            text = _para_text(child).strip()
            if not text:
                continue
            level = _heading_from_style(child)
            if level:
                _new(blocks, "heading", text, level=level, source=name)
            elif child.find(f"{W}pPr/{W}numPr") is not None:
                _new(blocks, "paragraph", "- " + text, source=name)
            else:
                _new(blocks, "paragraph", text, source=name)
        elif child.tag == W + "tbl":
            rows = []
            for row in child.iter(W + "tr"):
                rows.append([" ".join(_para_text(p).strip() for p in cell.iter(W + "p")).strip()
                             for cell in row.findall(W + "tc")])
            if rows:
                _new(blocks, "table", rows=rows, source=name)
    return Document(name, "docx", blocks, footnotes)


# ----------------------------------------------------------------------------- pdf
def load_pdf(data, name):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise DocumentError("خواندن PDF نیاز به بستهٔ pypdf دارد.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        pages = []
        for number, page in enumerate(reader.pages[:MAX_PDF_PAGES], 1):
            try:
                pages.append((page.extract_text() or "").strip())
            except Exception:
                pages.append("")
    except Exception as exc:
        raise DocumentError(f"PDF خوانده نشد: {type(exc).__name__}")
    notes = []
    if len(reader.pages) > MAX_PDF_PAGES:
        notes.append(f"فقط {MAX_PDF_PAGES} صفحهٔ اول PDF خوانده شد.")
    if pages and not any(pages):
        notes.append("PDF متن قابل‌استخراج ندارد (احتمالاً اسکن‌شده است؛ OCR لازم است).")
    # running headers / footers: a line that opens or closes at least half of the pages is page furniture
    furniture = set()
    if len(pages) >= 6:
        counts = {}
        for text in pages:
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            for line in set(lines[:1] + lines[-1:]):      # only the very first / last line can be page furniture
                key = re.sub(r"[0-9۰-۹]+", "#", line)
                counts[key] = counts.get(key, 0) + 1
        furniture = {key for key, n in counts.items() if n >= len(pages) * 0.5 and len(key) < 100}
    blocks = []
    for number, text in enumerate(pages, 1):
        kept = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                kept.append("")
                continue
            if re.sub(r"[0-9۰-۹]+", "#", stripped) in furniture or re.fullmatch(r"[-–— ]*[0-9۰-۹]+[-–— ]*", stripped):
                continue
            kept.append(stripped)
        _paragraphs_from_text("\n".join(kept), blocks, name, page=number, reflow=True)
    return Document(name, "pdf", _merge_page_breaks(blocks), notes=notes)


def _merge_page_breaks(blocks):
    """A paragraph cut by a page break (next page starts lowercase / no capital, previous has no end mark) is joined."""
    merged = []
    for block in blocks:
        previous = merged[-1] if merged else None
        if (previous and previous.kind == "paragraph" and block.kind == "paragraph" and block.page == previous.page + 1
                and not _SENTENCE_END.search(previous.text.rstrip()) and not re.search(r"[.!؟?]$", previous.text.rstrip())):
            previous.text += " " + block.text
            continue
        merged.append(block)
    for index, block in enumerate(merged):
        block.id = index
    return merged


# ----------------------------------------------------------------------------- zip / projects
_CODE_LANG = {".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx", ".java": "java",
              ".kt": "kotlin", ".dart": "dart", ".c": "c", ".h": "c", ".cpp": "cpp", ".cs": "csharp", ".go": "go",
              ".rs": "rust", ".php": "php", ".rb": "ruby", ".swift": "swift", ".sh": "bash", ".bat": "bat",
              ".ps1": "powershell", ".sql": "sql", ".html": "html", ".css": "css", ".json": "json", ".xml": "xml",
              ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini", ".gradle": "groovy"}


def load_zip(data, name, depth=0):
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        infos = [i for i in archive.infolist() if not i.is_dir()]
    except zipfile.BadZipFile:
        raise DocumentError("فایل ZIP خراب یا نامعتبر است.")
    if len(infos) > MAX_ZIP_ENTRIES:
        raise DocumentError(f"این ZIP بیش از حد فایل دارد ({len(infos)}).")
    if sum(i.file_size for i in infos) > MAX_ZIP_DECLARED:
        raise DocumentError("حجم باز‌شدهٔ این ZIP بیش از حد بزرگ است.")
    blocks, notes, skipped = [], [], 0
    infos.sort(key=lambda i: (len(PurePosixPath(_files.safe_name_path(i.filename)).parts), _files.safe_name_path(i.filename).lower()))
    listing = [f"{_files.safe_name_path(i.filename)} ({_files.human_size(i.file_size)})" for i in infos]
    _new(blocks, "heading", f"Archive: {name}", level=1, source=name)
    _new(blocks, "code", "\n".join(listing[:500]) + (f"\n… +{len(listing) - 500}" if len(listing) > 500 else ""),
         lang="text", source=name)
    for info in infos:
        path = _files.safe_name_path(info.filename)
        parts = PurePosixPath(path).parts
        ext = PurePosixPath(path.lower()).suffix
        if any(part in _files.SKIP_DIRS for part in parts[:-1]) or info.flag_bits & 0x1:
            skipped += 1
            continue
        if info.file_size > MAX_ZIP_FILE_BYTES and ext not in (".pdf", ".docx"):
            notes.append(f"فایل «{path}» بزرگ‌تر از حد مجاز بود و نادیده گرفته شد.")
            continue
        try:
            raw = archive.read(info)
        except (RuntimeError, zipfile.BadZipFile, NotImplementedError, OSError):
            skipped += 1
            continue
        try:
            if ext in (".docx", ".pdf", ".zip"):
                if ext == ".zip" and depth >= MAX_ZIP_DEPTH:
                    continue
                inner = load_document(raw, path, _depth=depth + 1)
                _new(blocks, "heading", f"File: {path}", level=2, source=path)
                offset = len(blocks)
                for b in inner.blocks:
                    b.id = len(blocks)
                    b.level = min(6, b.level + 2) if b.kind == "heading" else b.level
                    blocks.append(b)
                continue
            if _files.looks_like_text_name(path) or ext in _CODE_LANG or ext == ".ipynb":
                text = _files.decode_text(raw)
                if text is None:
                    skipped += 1
                    continue
                if ext == ".ipynb":
                    text = _files.extract_ipynb(text)
                _new(blocks, "heading", f"File: {path}", level=2, source=path)
                if ext in (".md", ".markdown"):
                    for b in parse_markdown(text, path):
                        b.id = len(blocks)
                        b.level = min(6, b.level + 2) if b.kind == "heading" else b.level
                        blocks.append(b)
                elif ext in _CODE_LANG or ext == ".ipynb" or ext in (".txt", ".cfg", ".env"):
                    _new(blocks, "code", text.rstrip("\n"), lang=_CODE_LANG.get(ext, ""), source=path)
                else:
                    _paragraphs_from_text(text, blocks, path)
            else:
                skipped += 1
        except DocumentError as exc:
            notes.append(f"«{path}»: {exc}")
    if skipped:
        notes.append(f"{skipped} فایل غیرمتنی/رمزدار/پوشهٔ ابزار فقط در فهرست آمد.")
    return Document(name, "zip", blocks, notes=notes)


# ----------------------------------------------------------------------------- entry point
def load_document(source, name=None, _depth=0):
    """Load a path or bytes into a structured ``Document``."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        name = name or path.name
        if path.stat().st_size > MAX_DOC_BYTES:
            raise DocumentError("فایل بیش از حد بزرگ است.")
        data = path.read_bytes()
    else:
        data = bytes(source)
        name = name or "document"
        if len(data) > MAX_DOC_BYTES:
            raise DocumentError("فایل بیش از حد بزرگ است.")
    ext = PurePosixPath(name.lower()).suffix
    if data[:4] == b"%PDF" or ext == ".pdf":
        return load_pdf(data, name)
    if data[:2] == b"PK":
        try:
            names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
        except zipfile.BadZipFile:
            raise DocumentError("فایل خراب است.")
        if "word/document.xml" in names or ext == ".docx":
            return load_docx(data, name)
        if ext in (".pptx", ".xlsx"):
            raise DocumentError("این قالب پشتیبانی نمی‌شود؛ PDF، DOCX، TXT، Markdown و ZIP.")
        return load_zip(data, name, _depth)
    if ext in TEXT_FORMATS or _files.looks_like_text_name(name) or ext == "":
        return load_text(data, name)
    if _files.decode_text(data) is not None and ext in _CODE_LANG:
        doc = Document(name, "code", [])
        _new(doc.blocks, "code", _files.decode_text(data), lang=_CODE_LANG[ext], source=name)
        return doc
    raise DocumentError("قالب فایل پشتیبانی نمی‌شود؛ PDF، DOCX، TXT، Markdown و ZIP.")


# ----------------------------------------------------------------------------- chunking
class Segment:
    """One block, or one numbered part of an oversized block."""

    __slots__ = ("block", "part", "parts", "text", "tokens", "kind", "level")

    def __init__(self, block, part, parts, text, kind, level):
        self.block, self.part, self.parts, self.text, self.kind, self.level = block, part, parts, text, kind, level
        self.tokens = estimate_tokens(text)


class Chunk:
    def __init__(self, index, segments, heading_path, footnotes):
        self.index = index
        self.segments = segments
        self.heading_path = heading_path          # headings in force where the chunk starts (context for the model)
        self.footnote_text = footnotes
        body = "\n\n".join(s.text for s in segments)
        self.text = body + (("\n\n" + footnotes) if footnotes else "")
        self.tokens = estimate_tokens(self.text)
        self.digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    @property
    def first(self):
        return (self.segments[0].block, self.segments[0].part)

    @property
    def last(self):
        return (self.segments[-1].block, self.segments[-1].part)

    def to_dict(self):
        return {"index": self.index, "first": list(self.first), "last": list(self.last), "tokens": self.tokens,
                "headings": self.heading_path, "digest": self.digest}


def _split_paragraph(text, limit):
    pieces = re.split(r"(?<=[.!؟?؛])\s+|\n+", text)
    parts, current = [], ""
    for piece in pieces:
        while estimate_tokens(piece) > limit:                     # one gigantic unbroken sentence
            cut = max(40, int(len(piece) * limit / max(1, estimate_tokens(piece))))
            if current:
                parts.append(current)
                current = ""
            parts.append(piece[:cut])
            piece = piece[cut:]
        if current and estimate_tokens(current + " " + piece) > limit:
            parts.append(current)
            current = piece
        else:
            current = (current + " " + piece).strip()
    if current:
        parts.append(current)
    return parts or [text]


def _split_code(block, limit):
    lines, parts, current = block.text.split("\n"), [], []
    for line in lines:
        probe = "\n".join(current + [line])
        if current and estimate_tokens(probe) > limit - 12:
            parts.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current))
    return [f"```{block.lang}\n{part}\n```" for part in parts]


def _split_table(block, limit):
    rows = block.rows or []
    if len(rows) <= 1:
        return [block.render()]
    header, parts, current = rows[0], [], []
    for row in rows[1:]:
        if current and estimate_tokens(render_table([header] + current + [row])) > limit:
            parts.append(render_table([header] + current))
            current = [row]
        else:
            current.append(row)
    if current:
        parts.append(render_table([header] + current))
    return parts


def _segments(block, limit):
    rendered = block.render()
    if estimate_tokens(rendered) <= limit:
        return [Segment(block.id, 0, 1, rendered, block.kind, block.level)]
    if block.kind == "code":
        texts = _split_code(block, limit)
    elif block.kind == "table":
        texts = _split_table(block, limit)
    else:
        texts = _split_paragraph(rendered, limit)
    return [Segment(block.id, n, len(texts), t, block.kind, block.level) for n, t in enumerate(texts)]


_FOOTREF = re.compile(r"\[\^(\w+)\]")


def _title(segment):
    return segment.text.lstrip("#").strip()


def chunk_document(doc, max_tokens, boundary_ratio=0.5):
    """Split ``doc`` into chunks of at most ``max_tokens`` (footnote text included), in document order.

    * a heading never ends a chunk (it travels with what follows);
    * a chunk is closed early at a chapter/section heading once it is ``boundary_ratio`` full;
    * code blocks and tables are kept whole when they fit, else split by lines / rows (header repeated).
    """
    max_tokens = max(200, int(max_tokens))
    segments = []
    for block in doc.blocks:
        segments.extend(_segments(block, max(120, int(max_tokens * 0.8))))

    def footnotes_for(chunk_segments):
        ids = []
        for seg in chunk_segments:
            for found in _FOOTREF.findall(seg.text):
                if found in doc.footnotes and found not in ids:
                    ids.append(found)
        return "\n".join(f"[^{i}]: {doc.footnotes[i]}" for i in ids)

    chunks, current, used, path, headings = [], [], 0, [], []
    start_path = []

    def close():
        nonlocal current, used, start_path
        if not current:
            return
        # a trailing heading belongs to the next chunk
        tail = []
        while len(current) > 1 and current[-1].kind == "heading":
            tail.insert(0, current.pop())
        chunks.append(Chunk(len(chunks), current, list(start_path), footnotes_for(current)))
        tail_titles = {_title(s) for s in tail if s.kind == "heading"}
        current, used = tail, sum(s.tokens + 2 for s in tail)
        start_path = [t for t in path if t not in tail_titles]

    for seg in segments:
        before = list(path)
        if seg.kind == "heading":
            headings = [h for h in headings if h[0] < seg.level] + [(seg.level, _title(seg))]
        cost = seg.tokens + 2
        if current and used + cost > max_tokens * 0.92:
            close()
        elif current and seg.kind == "heading" and seg.level <= 2 and used >= max_tokens * boundary_ratio:
            close()
        if not current:
            start_path = before
        current.append(seg)
        used += cost
        path = [h[1] for h in headings]
    close()
    return chunks


def plan_chunk_tokens(num_ctx, num_predict, overhead_tokens=900, task="translate"):
    """Largest chunk (tokens) that can be processed with ``task`` inside a window of ``num_ctx``.

    translate / edit: the answer is as long as the input, so input ≈ output and both must fit.
    summarize / analyze / qa: the answer is short, so the input may fill most of the window.
    """
    usable = int(num_ctx * 0.75) - int(overhead_tokens)          # keep the 75 % soft limit used everywhere
    if task in ("translate", "edit", "rewrite"):
        size = min(usable // 2, max(200, int(num_predict * 0.8)))
    else:
        size = usable - min(int(num_predict), usable // 3)
    return max(200, size)


# ----------------------------------------------------------------------------- integrity
def verify_chunks(doc, chunks):
    """Check that ``chunks`` cover every block of ``doc`` exactly once and in order.

    Returns ``{"ok": bool, "problems": [...], "blocks": n, "chunks": n}``.
    """
    problems = []
    expected = []
    for block in doc.blocks:
        limit_parts = None
        expected.append(block.id)
    seen, last = {}, None
    order = []
    for chunk in chunks:
        for seg in chunk.segments:
            key = (seg.block, seg.part)
            if key in seen:
                problems.append(f"تکراری: بلوک {seg.block} بخش {seg.part} در چانک {seen[key]} و {chunk.index}")
            seen[key] = chunk.index
            if last is not None and key < last:
                problems.append(f"جابه‌جایی ترتیب: {key} بعد از {last} (چانک {chunk.index})")
            last = max(last, key) if last is not None else key
            order.append(key)
    covered = {}
    for block_id, part in seen:
        covered.setdefault(block_id, []).append(part)
    for block_id in expected:
        parts = sorted(covered.get(block_id, []))
        if not parts:
            problems.append(f"جاافتاده: بلوک {block_id}")
            continue
        declared = next(s.parts for c in chunks for s in c.segments if s.block == block_id)
        if parts != list(range(declared)):
            problems.append(f"بخش‌های ناقص برای بلوک {block_id}: {parts} از {declared}")
    extra = set(covered) - set(expected)
    if extra:
        problems.append(f"بلوک ناشناخته: {sorted(extra)[:5]}")
    for chunk in chunks:
        if [c.index for c in chunks].count(chunk.index) != 1:
            problems.append(f"شمارهٔ چانک تکراری: {chunk.index}")
    if [c.index for c in chunks] != list(range(len(chunks))):
        problems.append("شمارهٔ چانک‌ها پیوسته نیست")
    return {"ok": not problems, "problems": problems, "blocks": len(doc.blocks), "chunks": len(chunks)}
