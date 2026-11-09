"""Fast, rule-based intent detection for web search (no extra model call).

Why rules instead of asking the local model: with a slow 30B model a classification round trip costs
many seconds before the search even starts. The rules below only decide *what kind of results to
fetch* (video / music / movie / image / news / plain web) and which clean query to send.

classify(text) -> {"kind", "topic", "query", "lang", "explicit"}
"""
import re

KINDS = ("web", "news", "image", "video", "music", "movie")

_PERSIAN = re.compile(r"[\u0600-\u06FF]")


def _normalize(text):
    text = str(text or "")
    text = text.replace("\u200c", " ").replace("\u200f", " ").replace("\u200e", " ")
    text = text.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه").replace("ۀ", "ه")
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)      # harakat / tatweel
    return re.sub(r"\s+", " ", text).strip()


def _words(*items):
    return "|".join(re.escape(item) for item in items)


# kind keywords (already normalized: no ZWNJ, Persian ye/kaf)
_VIDEO = _words("ویدیو", "ویدئو", "ویدیوی", "کلیپ", "تریلر", "یوتیوب", "آپارات", "فیلمک", "video", "clip", "trailer", "youtube", "teaser")
_MUSIC = _words("آهنگ", "اهنگ", "موزیک", "موسیقی", "ترانه", "آلبوم", "البوم", "سینگل", "song", "songs", "music", "track", "album", "mp3", "playlist", "soundtrack")
_IMAGE = _words("عکس", "تصویر", "تصاویر", "پوستر", "والپیپر", "پس زمینه", "photo", "photos", "image", "images", "picture", "pictures", "wallpaper", "poster", "pic", "pics")
_MOVIE = _words("فیلم", "سریال", "انیمه", "انیمیشن", "مستند", "movie", "movies", "film", "films", "series", "anime", "documentary", "tv show")
_NEWS = _words("خبر", "اخبار", "خبرها", "news", "headlines", "breaking")

# cues that the person wants something FOUND (so «توضیح بده درباره فیلم X» is not hijacked)
_CUE = (
    "پیدا", "بیاب", "سرچ", "جستجو", "جست وجو", "لینک", "دانلود", "نشان بده", "نشونم بده", "نشانم بده", "ببینم", "ببین",
    "بده", "بفرست", "میخوام", "می خوام", "می خواهم", "میخواهم", "بخوام", "کجا", "کجاست", "معرفی", "پیشنهاد", "بگرد",
    "بیار", "بیاور", "برام", "برایم", "یه ", "یک ", "find", "search", "link", "download", "show me", "give me",
    "get me", "where can i", "where to", "watch", "listen", "recommend", "suggest", "looking for", "i want", "i need",
)

_FILLER = re.compile(
    r"(?:\b(?:لطفا|لطفاً|ممنون|مرسی|خواهشا|خواهشاً|سلام|برام|برایم|برای من|واسم|واسه من|من|یه|یک|چند|تا|"
    r"ها|های|هایی|کجا|ببینم|ببینیم|دید|تماشا|از|را|رو|که|هم|هست|هستش|هستند|باشه|باشد|می ?خوام|می ?خواهم|بخوام|میشه|می شه|می‌شود|"
    r"پیدا|بیاب|بیابی|کن|کنی|کنید|کردن|بکن|بگرد|بگردی|بده|بدی|بدید|بفرست|بفرستی|بیار|بیاور|"
    r"سرچ|جستجو|جست وجو|لینک|لینکش|لینکشو|لینکشون|لینکشونو|لینک‌ها|لینکها|دانلود|نشان|نشونم|نشانم|ببینم|"
    r"در|توی|تو|اینترنت|وب|گوگل|اون|اونو|این|آن|را|"
    r"please|pls|can you|could you|find|search|for|me|a|an|the|some|link|links|to|of|download|show|give|get|"
    r"i|want|need|looking|where|can|watch|listen|online|internet|web|on|google)\b)",
    re.IGNORECASE,
)


def _has(pattern, text):
    return re.search(rf"(?<![\w])(?:{pattern})(?![\w])", text, re.IGNORECASE) is not None


_STRONG = (
    "پیدا", "بیاب", "سرچ", "جستجو", "جست وجو", "لینک", "دانلود", "بگرد", "نشان بده", "نشونم بده", "نشانم بده",
    "find", "search", "link", "download", "show me", "where can i", "looking for", "recommend", "suggest",
    "کجا ببین", "کجا میشه", "تماشا", "watch", "listen",
)
_EXPLAIN = (
    "توضیح", "خلاصه", "نقد", "تحلیل", "درباره", "داستان", "چیست", "یعنی چی", "بگو", "explain", "summar",
    "review", "about", "plot of", "what is", "tell me",
)


def _has_cue(text):
    lowered = text.lower()
    if any(cue in lowered for cue in _STRONG):
        return True
    if any(word in lowered for word in _EXPLAIN):
        return False        # «درباره فیلم X توضیح بده» is a question, not a request for links
    return any(cue in lowered for cue in _CUE)


def _topic(text, kind_pattern):
    cleaned = re.sub(rf"(?<![\w])(?:{kind_pattern})(?![\w])", " ", text, flags=re.IGNORECASE)
    cleaned = _FILLER.sub(" ", cleaned)
    cleaned = re.sub(r"[؟?!.,،؛:;«»\"'“”()\[\]{}]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_query(text, limit=300):
    """A short single-line query (the engine and the 500-char server limit both like that)."""
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    end = max(cut.rfind(". "), cut.rfind("؟"), cut.rfind("?"), cut.rfind(" "))
    return (cut[:end] if end > 80 else cut).strip()


def classify(text):
    """Decide what the person is looking for. Conservative: unclear → plain web search."""
    original = clean_query(text)
    norm = _normalize(original)
    lang = "fa" if _PERSIAN.search(norm) else "en"
    plain = {"kind": "web", "topic": original, "query": original, "lang": lang, "explicit": False}
    if not norm:
        return plain
    cue = _has_cue(norm)

    order = (
        ("video", _VIDEO),
        ("music", _MUSIC),
        ("image", _IMAGE),
        ("movie", _MOVIE),
    )
    for kind, pattern in order:
        if _has(pattern, norm) and cue:
            topic = _topic(norm, "|".join(p for _, p in order))
            if len(topic) < 2:
                topic = _topic(norm, pattern) or norm
            if len(topic) < 2:
                continue
            kind_word = re.search(rf"(?<![\w])(?:{pattern})(?![\w])", norm, flags=re.IGNORECASE)
            return {"kind": kind, "topic": topic[:160], "query": topic[:160], "lang": lang, "explicit": True,
                    "kind_word": kind_word.group(0) if kind_word else ""}
    if re.search(r"where can i watch|watch .* online|کجا (?:میشه |می شه )?(?:ببینم|ببینیم|دید|تماشا)", norm, re.IGNORECASE):
        topic = _topic(norm, "movie")
        if len(topic) >= 2:
            return {"kind": "movie", "topic": topic[:160], "query": topic[:160], "lang": lang, "explicit": True,
                    "kind_word": "movie" if lang == "en" else "فیلم"}
    if _has(_NEWS, norm):
        return {"kind": "news", "topic": original, "query": original, "lang": lang, "explicit": False}
    return plain


def wants_media(text):
    return classify(text)["kind"] in ("image", "video", "music", "movie")
