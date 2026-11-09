"""Text preparation for natural Persian speech synthesis.

This module is intentionally dependency-free (standard library only) so that both
the main server and the isolated TTS worker can import it.

The XTTS model is trained on clean spoken Persian. Raw chat answers contain
things it cannot pronounce (digits, Latin words, emoji, markdown, symbols),
which is what produces unintelligible words. ``normalize_for_speech`` rewrites
the answer into plain spoken Persian, and ``split_for_speech`` cuts it into
short sentence-sized chunks, which XTTS reads far more naturally than one long
paragraph (and which allows playback to start before the whole answer is ready).
"""
import re
import unicodedata

MAX_SPEECH_CHARS = 6000
CHUNK_TARGET = 180
CHUNK_MIN = 24

_ONES = ["صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
_TEENS = ["ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده"]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
_SCALES = ["", "هزار", "میلیون", "میلیارد", "تریلیون"]

_LETTER_NAMES = {
    "a": "آ", "b": "بی", "c": "سی", "d": "دی", "e": "ای", "f": "اف", "g": "جی",
    "h": "اچ", "i": "آی", "j": "جی", "k": "کی", "l": "ال", "m": "ام", "n": "ان",
    "o": "او", "p": "پی", "q": "کیو", "r": "آر", "s": "اس", "t": "تی", "u": "یو",
    "v": "وی", "w": "دابلیو", "x": "ایکس", "y": "وای", "z": "زد",
}

# Common technical / everyday Latin words, written the way Persian speakers say them.
_LATIN_WORDS = {
    "python": "پایتون", "django": "جنگو", "ollama": "اولاما", "qwen": "کوئن",
    "windows": "ویندوز", "linux": "لینوکس", "mac": "مک", "macos": "مک او اس",
    "docker": "داکر", "github": "گیت هاب", "git": "گیت", "gitlab": "گیت لب",
    "javascript": "جاوا اسکریپت", "js": "جی اس", "typescript": "تایپ اسکریپت",
    "node": "نود", "react": "ریکت", "html": "اچ تی ام ال", "css": "سی اس اس",
    "api": "ای پی آی", "ai": "هوش مصنوعی", "gpu": "جی پی یو", "cpu": "سی پی یو",
    "ram": "رم", "json": "جیسون", "sql": "اس کیو ال", "sqlite": "اس کیو لایت",
    "http": "اچ تی تی پی", "https": "اچ تی تی پی اس", "url": "یو آر ال",
    "pdf": "پی دی اف", "zip": "زیپ", "wav": "وی", "mp3": "ام پی تری",
    "ok": "اوکی", "email": "ایمیل", "google": "گوگل", "chatgpt": "چت جی پی تی",
    "gpt": "جی پی تی", "claude": "کلود", "anthropic": "آنتروپیک", "openai": "اوپن ای آی",
    "vscode": "وی اس کد", "vs": "وی اس", "code": "کد", "whisper": "ویسپر",
    "xtts": "ایکس تی تی اس", "tts": "تی تی اس", "stt": "اس تی تی", "llm": "ال ال ام",
    "localchat": "لوکال چت", "parsvoice": "پارس وویس", "ollama": "اولاما",
    "fastapi": "فست ای پی آی", "firefox": "فایرفاکس", "chrome": "کروم", "edge": "اج",
    "wifi": "وای فای", "bluetooth": "بلوتوث", "usb": "یو اس بی", "pc": "پی سی",
    "app": "اپ", "web": "وب", "site": "سایت", "online": "آنلاین", "offline": "آفلاین",
    "pip": "پیپ", "npm": "ان پی ام", "venv": "وی انو", "bat": "بت", "ps1": "پی اس وان",
    "ui": "یو آی", "ux": "یو ایکس", "id": "آی دی", "ip": "آی پی", "dns": "دی ان اس",
    "ssh": "اس اس اچ", "os": "او اس", "ide": "آی دی ای", "cuda": "کودا",
    "nvidia": "انویدیا", "rtx": "آر تی ایکس", "gb": "گیگابایت", "mb": "مگابایت",
    "kb": "کیلوبایت", "ms": "میلی ثانیه",
}

_SYMBOLS = {
    "%": " درصد ", "٪": " درصد ", "+": " به علاوه ", "=": " مساوی ", "&": " و ",
    "@": " ات ", "$": " دلار ", "°": " درجه ", "€": " یورو ", "£": " پوند ",
    "→": " ", "←": " ", "↔": " ", "×": " ضرب در ", "÷": " تقسیم بر ",
}

_DIACRITICS = re.compile("[ً-ٰٟـۖ-ۭ]")
_EMOJI_AND_SYMBOLS = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍⃣←-⇿⌀-⏿]"
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def int_to_persian_words(number):
    """Spell a non-negative integer in Persian (e.g. 1405 -> هزار و چهارصد و پنج)."""
    if number < 0:
        return "منفی " + int_to_persian_words(-number)
    if number == 0:
        return _ONES[0]
    if number >= 1000 ** len(_SCALES):
        return " ".join(_ONES[int(digit)] for digit in str(number))
    groups = []
    index = 0
    value = number
    while value:
        value, part = divmod(value, 1000)
        if part:
            words = _group_to_words(part)
            if index == 1 and part == 1:
                words = ""  # "هزار" not "یک هزار"
            scale = _SCALES[index]
            groups.append(" ".join(item for item in (words, scale) if item))
        index += 1
    return " و ".join(reversed(groups))


def _group_to_words(part):
    hundreds, rest = divmod(part, 100)
    pieces = []
    if hundreds:
        pieces.append(_HUNDREDS[hundreds])
    if rest:
        if rest < 10:
            pieces.append(_ONES[rest])
        elif rest < 20:
            pieces.append(_TEENS[rest - 10])
        else:
            tens, ones = divmod(rest, 10)
            pieces.append(_TENS[tens])
            if ones:
                pieces.append(_ONES[ones])
    return " و ".join(pieces)


def _digits_one_by_one(text):
    return " ".join(_ONES[int(char)] for char in text if char.isdigit())


def _read_integer(text):
    if len(text) > 1 and text.startswith("0"):
        return _digits_one_by_one(text)
    return int_to_persian_words(int(text))


def _read_decimal(text):
    whole, _dot, fraction = text.partition(".")
    result = _read_integer(whole)
    if fraction:
        spoken = _digits_one_by_one(fraction) if fraction.startswith("0") or len(fraction) > 3 \
            else int_to_persian_words(int(fraction))
        result += " ممیز " + spoken
    return result


def _replace_number(match):
    token = match.group(0).rstrip(",")
    trailing = "," if match.group(0).endswith(",") else ""
    if token.count(".") >= 2:  # version numbers such as 1.8.5
        return " نقطه ".join(_read_integer(part) for part in token.split(".")) + trailing
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", token):
        return _read_decimal(token.replace(",", "")) + trailing
    if "," in token:
        return ", ".join(_read_decimal(part) for part in token.split(",") if part) + trailing
    return _read_decimal(token) + trailing


def _replace_latin(match):
    word = match.group(0)
    lowered = word.lower()
    if lowered in _LATIN_WORDS:
        return " " + _LATIN_WORDS[lowered] + " "
    if word.isupper() and len(word) <= 5:
        return " " + " ".join(_LETTER_NAMES.get(char.lower(), "") for char in word) + " "
    return " "  # unknown foreign word: the Persian model would only garble it


def _normalize_characters(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
    text = text.replace("٫", ".").replace("٬", ",")
    text = text.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک").replace("ة", "ه")
    text = text.replace("ؤ", "و").replace("أ", "ا").replace("إ", "ا")
    text = _DIACRITICS.sub("", text)
    text = text.replace("‌", " ").replace("‏", " ").replace("‎", " ")
    text = text.replace(" ", " ")
    return _EMOJI_AND_SYMBOLS.sub(" ", text)


def _strip_markdown(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\((?:https?://)[^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", " پیوند اینترنتی ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:[-*+•▪●○]|\d+[.)])\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*[-=_*]{3,}\s*$", " ", text, flags=re.M)
    text = re.sub(r"\[\d+\]", " ", text)  # citation markers like [1]
    text = text.replace("**", " ").replace("__", " ")
    return text


def normalize_for_speech(value, limit=MAX_SPEECH_CHARS):
    """Return plain, speakable Persian text with sentence-ending punctuation."""
    text = _strip_markdown(str(value or ""))
    text = _normalize_characters(text)

    text = re.sub(r"(\d{4})-(\d{1,2})-(\d{1,2})", r"\1, \2, \3", text)
    text = re.sub(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", r"\1 و \2 دقیقه", text)
    text = re.sub(r"(?<=\d)\s*[-–—]\s*(?=\d)", " تا ", text)
    text = re.sub(r"\d[\d,]*(?:\.\d+)*", _replace_number, text)

    for symbol, spoken in _SYMBOLS.items():
        text = text.replace(symbol, spoken)
    text = re.sub(r"[A-Za-z][A-Za-z0-9]*", _replace_latin, text)

    text = text.translate(str.maketrans({
        "،": ",", "؛": ",", ";": ",", "؟": "?", "…": ".", ":": ",",
        "(": ", ", ")": ", ", "[": ", ", "]": ", ", "{": ", ", "}": ", ",
        "«": " ", "»": " ", '"': " ", "“": " ", "”": " ", "'": " ", "’": " ", "‘": " ",
        "–": ",", "—": ",", "-": " ", "_": " ", "*": " ", "#": " ", "~": " ",
        "|": " ", ">": " ", "<": " ", "/": " ", "\\": " ", "^": " ",
    }))

    lines = []
    for raw_line in text.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip(" ,")
        if not line or not re.search(r"[؀-ۿ]", line):
            continue
        if line[-1] not in ".!?":
            line += "."
        lines.append(line)
    text = " ".join(lines)

    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?])(?:\s*[,.!?])+", lambda m: m.group(0)[0], text)
    text = re.sub(r"([,.!?])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text[:limit].strip()


def split_for_speech(text, target=CHUNK_TARGET, minimum=CHUNK_MIN):
    """Cut speakable text into short sentence groups (<= ~target characters)."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    pieces = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        pieces.extend(_split_long_sentence(sentence, target))

    chunks = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) < minimum or len(current) + 1 + len(piece) <= target:
            current = f"{current} {piece}"
        else:
            chunks.append(current)
            current = piece
    if current:
        if chunks and len(current) < minimum and len(chunks[-1]) + 1 + len(current) <= target + minimum:
            chunks[-1] = f"{chunks[-1]} {current}"
        else:
            chunks.append(current)
    return chunks


def _split_long_sentence(sentence, target):
    if len(sentence) <= target:
        return [sentence]
    parts = []
    for clause in re.split(r"(?<=,)\s+", sentence):
        if len(clause) <= target:
            parts.append(clause)
            continue
        words = clause.split(" ")
        current = ""
        for word in words:
            if current and len(current) + 1 + len(word) > target:
                parts.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            parts.append(current)
    merged = []
    current = ""
    for part in parts:
        if current and len(current) + 1 + len(part) > target:
            merged.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        merged.append(current)
    return merged
