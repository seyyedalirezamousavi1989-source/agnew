"""Chat attachments for LocalChat: safe extraction + a small on-disk store.

Everything here is standard library (PDF needs the optional ``pypdf``).
Nothing from an uploaded file is ever written to disk as a file, executed or
extracted to a folder: archives are only *read in memory* with hard limits.
"""
import base64
import io
import json
import os
import re
import secrets
import time
import wave
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_STORED_TEXT_CHARS = 400_000
MAX_AUDIO_SECONDS = 600
AUDIO_CHUNK_SECONDS = 28  # Gemma audio models accept at most 30 s per clip
AUDIO_RATE = 16000
ATTACHMENT_TTL_SECONDS = 7 * 24 * 3600
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

# --- zip limits (zip-bomb / huge archive protection) --------------------------
ZIP_MAX_ENTRIES_LISTED = 400
ZIP_MAX_ENTRIES_SCANNED = 5000
ZIP_MAX_DECLARED_TOTAL = 500 * 1024 * 1024
ZIP_MAX_FILE_BYTES = 1_000_000
ZIP_MAX_FILES_READ = 80
ZIP_MAX_TEXT_CHARS = 250_000
ZIP_PER_FILE_CHARS = 40_000
SKIP_DIRS = {
    "__pycache__", ".git", ".svn", ".hg", "node_modules", ".venv", "venv", "env",
    ".idea", ".vscode", "dist", "build", ".mypy_cache", ".pytest_cache",
    ".tox", "site-packages", ".gradle", "target", ".next", ".cache",
}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".py", ".pyw", ".pyi", ".js", ".mjs",
    ".cjs", ".ts", ".tsx", ".jsx", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".env", ".xml", ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".sql", ".csv", ".tsv", ".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1", ".c", ".h",
    ".cpp", ".hpp", ".cc", ".cxx", ".cs", ".java", ".kt", ".kts", ".go", ".rs", ".rb",
    ".php", ".swift", ".lua", ".r", ".m", ".dart", ".vue", ".svelte", ".gradle",
    ".properties", ".tex", ".srt", ".vtt", ".gitignore", ".dockerfile", ".editorconfig",
    ".lock", ".pl", ".scala", ".hs", ".ex", ".exs", ".clj", ".nim", ".jl",
    ".asm", ".s", ".v", ".sv", ".vhd", ".ahk", ".reg", ".inf", ".manifest", ".spec",
    ".iss",
}
TEXT_FILENAMES = {
    "dockerfile", "makefile", "readme", "license", "licence", "requirements", "procfile",
    "gemfile", "rakefile", "cmakelists.txt", "pipfile", ".gitignore", ".env", ".dockerignore",
}
IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


class AttachmentError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


# --- helpers ---------------------------------------------------------------------

def human_size(size):
    size = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def safe_name(name, limit=160):
    value = re.sub(r"[\x00-\x1f\x7f]", "", str(name or "")).replace("\\", "/")
    value = value.rsplit("/", 1)[-1].strip()
    return (value or "file")[:limit]


def _ext(name):
    return PurePosixPath(name.lower()).suffix


def looks_like_text_name(name):
    lowered = name.lower()
    base = PurePosixPath(lowered).name
    return _ext(lowered) in TEXT_EXTENSIONS or base in TEXT_FILENAMES


def decode_text(data):
    """Decode bytes into text. Returns None when the data is clearly binary."""
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    if b"\x00" in data[:8192]:
        return None
    for encoding in ("utf-8-sig",):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    try:  # Windows Persian/Arabic files are often cp1256
        return data.decode("cp1256")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def _clip(text, limit):
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def image_mime(data):
    for magic, mime in IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


# --- office formats (stdlib XML) ---------------------------------------------------

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
S_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_XML_MEMBER_LIMIT = 30 * 1024 * 1024


def _read_member(archive, name, limit=_XML_MEMBER_LIMIT):
    with archive.open(name) as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise AttachmentError("یکی از بخش‌های فایل بیش از حد بزرگ است.", 413)
    return data


def extract_docx(data):
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        root = ET.fromstring(_read_member(archive, "word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        raise AttachmentError("فایل Word خراب یا نامعتبر است.", 415)

    def paragraph_text(paragraph):
        parts = []
        for node in paragraph.iter():
            if node.tag == W_NS + "t" and node.text:
                parts.append(node.text)
            elif node.tag in (W_NS + "tab",):
                parts.append("\t")
            elif node.tag in (W_NS + "br", W_NS + "cr"):
                parts.append("\n")
        return "".join(parts)

    lines = []
    body = root.find(W_NS + "body")
    for child in (body if body is not None else []):
        if child.tag == W_NS + "p":
            text = paragraph_text(child).strip()
            if text:
                style = child.find(f"{W_NS}pPr/{W_NS}pStyle")
                level = ""
                if style is not None:
                    value = style.get(W_NS + "val", "")
                    match = re.fullmatch(r"(?i)heading\s*(\d)", value)
                    if match:
                        level = "#" * int(match.group(1)) + " "
                lines.append(level + text)
        elif child.tag == W_NS + "tbl":
            for row in child.iter(W_NS + "tr"):
                cells = []
                for cell in row.findall(W_NS + "tc"):
                    cells.append(" ".join(
                        paragraph_text(p).strip() for p in cell.iter(W_NS + "p")
                    ).strip())
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_pptx(data):
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise AttachmentError("فایل PowerPoint خراب یا نامعتبر است.", 415)
    slides = sorted(
        (n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
        key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)),
    )
    out = []
    for index, name in enumerate(slides[:200], 1):
        try:
            root = ET.fromstring(_read_member(archive, name))
        except ET.ParseError:
            continue
        paragraphs = []
        for paragraph in root.iter(A_NS + "p"):
            text = "".join(t.text or "" for t in paragraph.iter(A_NS + "t")).strip()
            if text:
                paragraphs.append(text)
        out.append(f"--- اسلاید {index} ---\n" + "\n".join(paragraphs))
    return "\n\n".join(out)


def _column_index(reference):
    letters = re.match(r"[A-Z]+", reference or "")
    if not letters:
        return 0
    number = 0
    for char in letters.group(0):
        number = number * 26 + (ord(char) - 64)
    return number - 1


def extract_xlsx(data):
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise AttachmentError("فایل Excel خراب یا نامعتبر است.", 415)
    shared = []
    if "xl/sharedStrings.xml" in archive.namelist():
        try:
            root = ET.fromstring(_read_member(archive, "xl/sharedStrings.xml"))
            for item in root.iter(S_NS + "si"):
                shared.append("".join(t.text or "" for t in item.iter(S_NS + "t")))
        except ET.ParseError:
            shared = []
    names = []
    if "xl/workbook.xml" in archive.namelist():
        try:
            root = ET.fromstring(_read_member(archive, "xl/workbook.xml"))
            names = [s.get("name", "") for s in root.iter(S_NS + "sheet")]
        except ET.ParseError:
            names = []
    sheets = sorted(
        (n for n in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)),
        key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)),
    )
    out = []
    for index, name in enumerate(sheets[:10]):
        title = names[index] if index < len(names) and names[index] else f"Sheet{index + 1}"
        try:
            root = ET.fromstring(_read_member(archive, name))
        except ET.ParseError:
            continue
        rows = []
        for row in root.iter(S_NS + "row"):
            if len(rows) >= 500:
                rows.append(["…"])
                break
            values = {}
            for cell in row.findall(S_NS + "c"):
                kind = cell.get("t")
                value_node = cell.find(S_NS + "v")
                text = ""
                if kind == "s" and value_node is not None and value_node.text is not None:
                    try:
                        text = shared[int(value_node.text)]
                    except (ValueError, IndexError):
                        text = ""
                elif kind == "inlineStr":
                    text = "".join(t.text or "" for t in cell.iter(S_NS + "t"))
                elif value_node is not None and value_node.text is not None:
                    text = value_node.text
                if text != "":
                    values[min(_column_index(cell.get("r")), 60)] = text
            if values:
                width = max(values) + 1
                rows.append([values.get(i, "") for i in range(width)])
        out.append(f"--- شیت «{title}» ---\n" + "\n".join("\t".join(r) for r in rows))
    return "\n\n".join(out)


def extract_pdf(data):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise AttachmentError(
            "خواندن PDF نیاز به بستهٔ pypdf دارد (pip install pypdf). "
            "نسخهٔ نصبی برنامه آن را همراه خودش دارد.", 501)
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise AttachmentError("این PDF رمزدار است.", 415)
        pages = []
        for number, page in enumerate(reader.pages[:300], 1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:
                text = ""
            if text:
                pages.append(f"--- صفحه {number} ---\n{text}")
        return "\n\n".join(pages)
    except AttachmentError:
        raise
    except Exception:
        raise AttachmentError("PDF خوانده نشد؛ فایل خراب یا نامعتبر است.", 415)


def extract_ipynb(text):
    try:
        notebook = json.loads(text)
        cells = notebook.get("cells", [])
    except (ValueError, AttributeError):
        return text
    out = []
    for index, cell in enumerate(cells, 1):
        if not isinstance(cell, dict):
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else str(source)
        kind = cell.get("cell_type", "code")
        out.append(f"# --- سلول {index} ({kind}) ---\n{source}")
        for output in cell.get("outputs", []) if kind == "code" else []:
            chunk = output.get("text") if isinstance(output, dict) else None
            if chunk:
                chunk = "".join(chunk) if isinstance(chunk, list) else str(chunk)
                out.append("# خروجی:\n" + chunk[:1500])
    return "\n\n".join(out)


# --- zip ---------------------------------------------------------------------------

def extract_zip(data, name):
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        infos = archive.infolist()
    except zipfile.BadZipFile:
        raise AttachmentError("فایل ZIP خراب یا نامعتبر است.", 415)
    if len(infos) > ZIP_MAX_ENTRIES_SCANNED:
        raise AttachmentError(
            f"این ZIP بیش از حد فایل دارد ({len(infos)}). حداکثر {ZIP_MAX_ENTRIES_SCANNED} فایل.", 413)
    files = [i for i in infos if not i.is_dir()]
    declared_total = sum(i.file_size for i in files)
    notes = []
    if declared_total > ZIP_MAX_DECLARED_TOTAL:
        raise AttachmentError(
            "حجم باز‌شدهٔ این ZIP بیش از حد بزرگ است (" + human_size(declared_total) + ").", 413)

    def display(info):
        return safe_name_path(info.filename)

    def skipped(path):
        return any(part in SKIP_DIRS for part in PurePosixPath(path).parts[:-1])

    readable = []
    listing = []
    for info in files:
        path = display(info)
        listing.append((path, info.file_size))
        if skipped(path) or info.flag_bits & 0x1:
            continue
        if looks_like_text_name(path) or _ext(path) in {".ipynb"}:
            readable.append((path, info))
    readable.sort(key=lambda item: (len(PurePosixPath(item[0]).parts), item[0].lower()))

    lines = [f"آرشیو ZIP «{name}»: {len(files)} فایل، حجم بازشده {human_size(declared_total)}", "", "ساختار:"]
    for path, size in listing[:ZIP_MAX_ENTRIES_LISTED]:
        lines.append(f"  {path} ({human_size(size)})")
    if len(listing) > ZIP_MAX_ENTRIES_LISTED:
        lines.append(f"  … و {len(listing) - ZIP_MAX_ENTRIES_LISTED} فایل دیگر")
    encrypted = sum(1 for i in files if i.flag_bits & 0x1)
    if encrypted:
        notes.append(f"{encrypted} فایل رمزدار خوانده نشد.")

    lines += ["", "محتوای فایل‌های متنی:"]
    total_chars = 0
    read_count = 0
    unread = 0
    for path, info in readable:
        if read_count >= ZIP_MAX_FILES_READ or total_chars >= ZIP_MAX_TEXT_CHARS:
            unread += 1
            continue
        try:
            with archive.open(info) as handle:
                raw = handle.read(ZIP_MAX_FILE_BYTES + 1)
        except (RuntimeError, zipfile.BadZipFile, NotImplementedError, OSError):
            unread += 1
            continue
        too_big = len(raw) > ZIP_MAX_FILE_BYTES
        text = decode_text(raw[:ZIP_MAX_FILE_BYTES])
        if text is None:
            continue
        if path.lower().endswith(".ipynb"):
            text = extract_ipynb(text)
        text, clipped = _clip(text, ZIP_PER_FILE_CHARS)
        total_chars += len(text)
        read_count += 1
        marker = " [بخشی از فایل نمایش داده شد]" if (clipped or too_big) else ""
        lines.append(f"\n=== {path}{marker} ===\n{text}")
    if unread:
        notes.append(f"{unread} فایل متنی دیگر به‌خاطر محدودیت حجم نمایش داده نشد.")
    binary_count = len(files) - len(readable)
    if binary_count > 0:
        notes.append(f"{binary_count} فایل غیرمتنی/نادیده‌گرفته‌شده (تصویر، باینری، پوشهٔ ابزارها) فقط در فهرست آمده‌اند.")
    return "\n".join(lines), notes


def safe_name_path(path):
    value = re.sub(r"[\x00-\x1f\x7f]", "", str(path)).replace("\\", "/")
    parts = [p for p in value.split("/") if p not in ("", ".", "..")]
    return "/".join(parts)[:200] or "file"


# --- audio -------------------------------------------------------------------------

def split_wav(data):
    """Validate a 16 kHz mono 16-bit WAV and cut it into <=28 s WAV chunks."""
    try:
        reader = wave.open(io.BytesIO(data), "rb")
    except (wave.Error, EOFError):
        raise AttachmentError("فایل صوتی WAV معتبر نیست.", 415)
    with reader:
        if (reader.getnchannels(), reader.getframerate(), reader.getsampwidth()) != (1, AUDIO_RATE, 2):
            raise AttachmentError(
                "صدا باید 16kHz، تک‌کاناله و 16 بیت باشد؛ رابط برنامه خودش این تبدیل را انجام می‌دهد.", 415)
        total_frames = reader.getnframes()
        seconds = total_frames / AUDIO_RATE
        truncated = seconds > MAX_AUDIO_SECONDS
        frames_left = min(total_frames, MAX_AUDIO_SECONDS * AUDIO_RATE)
        chunk_frames = AUDIO_CHUNK_SECONDS * AUDIO_RATE
        chunks = []
        while frames_left > 0:
            take = min(chunk_frames, frames_left)
            pcm = reader.readframes(take)
            if not pcm:
                break
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(AUDIO_RATE)
                out.writeframes(pcm)
            chunks.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
            frames_left -= take
    if not chunks:
        raise AttachmentError("فایل صوتی خالی است.", 415)
    return chunks, min(seconds, MAX_AUDIO_SECONDS), truncated


# --- main entry --------------------------------------------------------------------

def process_upload(name, data):
    """Turn raw uploaded bytes into an attachment record (not yet stored)."""
    name = safe_name(name)
    if not data:
        raise AttachmentError("فایل خالی است.", 400)
    if len(data) > MAX_UPLOAD_BYTES:
        raise AttachmentError("حجم فایل بیش از حد مجاز است.", 413)
    ext = _ext(name)
    record = {
        "name": name, "size": len(data), "kind": "text", "text": "", "truncated": False,
        "notes": [], "images": [], "audio_chunks": [], "audio_seconds": 0,
        "transcript": None, "transcript_model": None, "created": int(time.time()),
    }

    mime = image_mime(data)
    if mime:
        if len(data) > MAX_IMAGE_BYTES:
            raise AttachmentError("حجم تصویر بیش از ۱۲ مگابایت است.", 413)
        record["kind"] = "image"
        record["images"] = [base64.b64encode(data).decode("ascii")]
        return record

    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        chunks, seconds, truncated = split_wav(data)
        record.update(kind="audio", audio_chunks=chunks, audio_seconds=round(seconds, 1))
        if truncated:
            record["notes"].append(f"فقط {MAX_AUDIO_SECONDS // 60} دقیقهٔ اول صدا پردازش می‌شود.")
        return record

    if data[:5] == b"%PDF-":
        record["kind"] = "document"
        text = extract_pdf(data)
        if not text.strip():
            record["notes"].append("متنی در این PDF پیدا نشد (احتمالاً اسکن/تصویری است).")
    elif data[:4] == b"PK\x03\x04" or data[:4] == b"PK\x05\x06":
        if ext in (".docx", ".dotx", ".docm"):
            record["kind"] = "document"
            text = extract_docx(data)
        elif ext in (".xlsx", ".xlsm"):
            record["kind"] = "document"
            text = extract_xlsx(data)
        elif ext in (".pptx", ".ppsx"):
            record["kind"] = "document"
            text = extract_pptx(data)
        else:
            record["kind"] = "archive"
            text, notes = extract_zip(data, name)
            record["notes"].extend(notes)
    else:
        text = decode_text(data)
        if text is None or (not looks_like_text_name(name) and _control_ratio(text) > 0.02):
            raise AttachmentError(
                "این نوع فایل پشتیبانی نمی‌شود. فایل‌های مجاز: عکس، متن/کد (پایتون، جاوااسکریپت، …)، "
                "ZIP، PDF، Word/Excel/PowerPoint و صدا.", 415)
        if ext == ".ipynb":
            text = extract_ipynb(text)
    text, clipped = _clip(text, MAX_STORED_TEXT_CHARS)
    record["text"] = text
    record["truncated"] = clipped
    if clipped:
        record["notes"].append("متن فایل خیلی بلند بود و فقط بخش اول ذخیره شد.")
    return record


def _control_ratio(text):
    sample = text[:20000]
    if not sample:
        return 0
    bad = sum(1 for c in sample if ord(c) < 32 and c not in "\n\r\t\f")
    return bad / len(sample)


def public_view(record):
    return {
        "id": record["id"],
        "name": record["name"],
        "kind": record["kind"],
        "size": record["size"],
        "chars": len(record.get("text", "")),
        "image_count": len(record.get("images", [])),
        "audio_seconds": record.get("audio_seconds", 0),
        "truncated": bool(record.get("truncated")),
        "notes": list(record.get("notes", [])),
    }


# --- store -------------------------------------------------------------------------

class AttachmentStore:
    def __init__(self, directory_getter):
        self._directory_getter = directory_getter

    @property
    def directory(self):
        path = Path(self._directory_getter())
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _path(self, attachment_id):
        if not isinstance(attachment_id, str) or not ID_PATTERN.fullmatch(attachment_id):
            return None
        return self.directory / f"{attachment_id}.json"

    def put(self, record):
        attachment_id = secrets.token_hex(16)
        record = dict(record, id=attachment_id)
        self._write(record)
        self.cleanup()
        return record

    def _write(self, record):
        path = self._path(record["id"])
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)

    def get(self, attachment_id):
        path = self._path(attachment_id)
        if path is None or not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def update(self, record):
        if self._path(record.get("id")) is not None:
            self._write(record)

    def delete(self, attachment_id):
        path = self._path(attachment_id)
        if path is None:
            return False
        try:
            path.unlink()
            return True
        except OSError:
            return False

    def cleanup(self, max_age=ATTACHMENT_TTL_SECONDS):
        cutoff = time.time() - max_age
        try:
            for item in self.directory.glob("*.json"):
                try:
                    if item.stat().st_mtime < cutoff:
                        item.unlink()
                except OSError:
                    pass
        except OSError:
            pass


# --- prompt rendering --------------------------------------------------------------

KIND_LABELS = {
    "image": "تصویر", "text": "فایل متنی/کد", "archive": "آرشیو ZIP",
    "document": "سند", "audio": "صدا",
}


def render_for_prompt(record, limit):
    """Text block describing one attachment, at most ``limit`` characters of content.

    Returns (block, was_truncated).
    """
    label = KIND_LABELS.get(record.get("kind"), "فایل")
    header = f"«{record['name']}» ({label}، {human_size(record['size'])})"
    kind = record.get("kind")
    if kind == "image":
        return f"[پیوست {header} — تصویر مستقیماً همراه همین پیام برای مدل ارسال شده است.]", False
    if kind == "audio":
        transcript = record.get("transcript")
        if transcript is None:
            return f"[پیوست {header} — رونویسی صدا در دسترس نیست.]", False
        text, clipped = _clip(transcript, max(limit, 200))
        model = record.get("transcript_model") or "مدل صوتی"
        return (f"[پیوست {header} — متن رونویسی‌شدهٔ صدا توسط {model}]\n{text}"
                + ("\n[… ادامهٔ متن به‌خاطر محدودیت حجم حذف شد]" if clipped else "")), clipped
    text = record.get("text", "")
    if not text.strip():
        note = " ".join(record.get("notes", [])) or "محتوای متنی ندارد."
        return f"[پیوست {header} — {note}]", False
    clipped_text, clipped = _clip(text, max(limit, 200))
    marker = ""
    if clipped or record.get("truncated"):
        marker = "\n[… ادامهٔ فایل به‌خاطر محدودیت حجم/زمینهٔ مدل حذف شد]"
    return f"[پیوست {header}]\n{clipped_text}{marker}", bool(clipped or record.get("truncated"))
