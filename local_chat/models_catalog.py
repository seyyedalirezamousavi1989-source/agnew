"""Model catalog + capability heuristics for LocalChat (no network access here)."""
import re

# group: vision | audio | text.  Sizes are the approximate download size from the
# Ollama library (they can change); the real size is shown while downloading.
CATALOG = [
    # --- بینایی (تصویر) ---
    {"id": "qwen3-vl:8b", "group": "vision", "name": "Qwen3-VL 8B", "size": "≈ 6.1 GB",
     "desc": "بهترین کیفیت عمومی برای عکس، اسکرین‌شات و متن داخل تصویر؛ روی کارت ۱۲ گیگ راحت اجرا می‌شود.",
     "recommended": True, "vision": True},
    {"id": "qwen3-vl:4b", "group": "vision", "name": "Qwen3-VL 4B", "size": "≈ 3.3 GB",
     "desc": "سبک‌تر و سریع‌تر؛ مناسب کارت ۸ گیگ.", "vision": True},
    {"id": "qwen3-vl:2b", "group": "vision", "name": "Qwen3-VL 2B", "size": "≈ 1.9 GB",
     "desc": "خیلی سبک برای سیستم‌های ضعیف؛ در تصاویر پیچیده ضعیف‌تر است.", "vision": True},
    {"id": "qwen2.5vl:7b", "group": "vision", "name": "Qwen2.5-VL 7B", "size": "≈ 6.0 GB",
     "desc": "قوی در خواندن جدول، نمودار و اسناد.", "vision": True},
    {"id": "minicpm-v", "group": "vision", "name": "MiniCPM-V", "size": "≈ 5.5 GB",
     "desc": "مناسب OCR و خواندن متن داخل تصویر.", "vision": True},
    {"id": "llama3.2-vision:11b", "group": "vision", "name": "Llama 3.2 Vision 11B", "size": "≈ 7.8 GB",
     "desc": "مدل بینایی متا؛ برای تصویر انگلیسی بهتر است.", "vision": True},
    {"id": "llava:7b", "group": "vision", "name": "LLaVA 7B", "size": "≈ 4.7 GB",
     "desc": "قدیمی ولی سبک و سازگار.", "vision": True},
    {"id": "moondream", "group": "vision", "name": "Moondream 1.8B", "size": "≈ 1.7 GB",
     "desc": "بسیار کوچک؛ توضیح ساده تصویر.", "vision": True},
    # --- صوتی (به‌همراه بینایی) ---
    {"id": "gemma4:e4b", "group": "audio", "name": "Gemma 4 E4B", "size": "≈ 10 GB",
     "desc": "متن + تصویر + صدا. فایل صوتی را به متن تبدیل می‌کند (تا ۱۰ دقیقه) و بعد تحلیل می‌شود.",
     "recommended": True, "vision": True, "audio": True},
    {"id": "gemma4:e2b", "group": "audio", "name": "Gemma 4 E2B", "size": "≈ 7 GB",
     "desc": "نسخهٔ سبک‌تر: متن + تصویر + صدا. برای کارت‌های کم‌حافظه.", "vision": True, "audio": True},
    # --- مدل‌های زبانی جدیدتر و استدلالی ---
    {"id": "qwen3.5:9b", "group": "text", "name": "Qwen 3.5 9B", "size": "≈ 6.6–7.6 GB",
     "desc": "مدل چندزبانهٔ نسل جدید با توانایی استدلال و درک تصویر؛ گزینهٔ اصلی پیشنهادی برای سیستم‌های ۱۲GB VRAM با توجه به مصرف واقعی حافظه.", "recommended": True, "vision": True},
    {"id": "qwen3.5:4b", "group": "text", "name": "Qwen 3.5 4B", "size": "≈ 3.3–4.0 GB",
     "desc": "نسخهٔ سبک‌تر Qwen 3.5 برای پاسخ‌گویی سریع‌تر و سیستم‌های محدودتر؛ دارای قابلیت بینایی.", "vision": True},
    {"id": "deepseek-r1:14b", "group": "text", "name": "DeepSeek R1 14B", "size": "≈ 9 GB",
     "desc": "مدل استدلالی برای مسائل چندمرحله‌ای؛ روی GPU دوازده گیگ ممکن است به RAM سیستم نیز متکی شود.", "recommended": True},
    {"id": "qwen2.5-coder:14b", "group": "text", "name": "Qwen 2.5 Coder 14B", "size": "≈ 9 GB",
     "desc": "مدل تخصصی کدنویسی و تحلیل کد؛ برای استفادهٔ روان ممکن است به Offload بخشی از مدل روی RAM نیاز داشته باشد."},
    # --- متنی ---
    {"id": "qwen3:8b", "group": "text", "name": "Qwen3 8B", "size": "≈ 5.2 GB",
     "desc": "مدل متنی سریع و خوب برای فارسی.", "recommended": True},
    {"id": "qwen2.5:7b", "group": "text", "name": "Qwen2.5 7B", "size": "≈ 4.7 GB",
     "desc": "سبک و پایدار."},
    {"id": "llama3.1:8b", "group": "text", "name": "Llama 3.1 8B", "size": "≈ 4.9 GB",
     "desc": "مدل عمومی متا."},
    {"id": "qwen2.5-coder:7b", "group": "text", "name": "Qwen2.5 Coder 7B", "size": "≈ 4.7 GB",
     "desc": "ویژهٔ کدنویسی و تحلیل فایل‌های پایتون."},
    {"id": "gemma3:4b", "group": "text", "name": "Gemma 3 4B", "size": "≈ 3.3 GB",
     "desc": "کوچک؛ متن + تصویر.", "vision": True},
]

GROUP_LABELS = {"vision": "بینایی", "audio": "صوتی", "text": "متنی"}

_VISION_NAME = re.compile(
    r"(llava|bakllava|vision|[-_:.]vl(?:[-_:.:]|$)|\bvl[-_]|minicpm-v|moondream|qwen\d*(?:\.\d)?vl|"
    r"pixtral|granite3\.\d-vision|llama4|mistral-small3|internvl|glm-4\.\dv|gemma4|gemma3(?!n))",
    re.I,
)
_TEXT_ONLY_GEMMA3 = re.compile(r"gemma3:(?:1b|270m)", re.I)
_AUDIO_NAME = re.compile(r"(gemma4:e[24]b|gemma-?3n|audio|whisper)", re.I)
_THINKING_NAME = re.compile(r"(qwen3(?!-coder|-vl.*instruct)|deepseek-r1|gpt-oss|magistral|qwq|-thinking|gemma4|phi4-reasoning)", re.I)
_EMBED_NAME = re.compile(r"(embed|bge-|nomic|minilm|e5-|arctic-embed|snowflake|paraphrase-multilingual|mxbai|granite-embedding)", re.I)


def guess_capabilities(name):
    """Best-effort capabilities from a model name, used when /api/show is unavailable."""
    caps = []
    lowered = (name or "").lower()
    if _EMBED_NAME.search(lowered):
        return ["embedding"]
    caps.append("completion")
    if _VISION_NAME.search(lowered) and not _TEXT_ONLY_GEMMA3.search(lowered):
        caps.append("vision")
    if _AUDIO_NAME.search(lowered) and "whisper" not in lowered:
        caps.append("audio")
    if _THINKING_NAME.search(lowered):
        caps.append("thinking")
    return caps


def normalize_name(name):
    name = (name or "").strip().lower()
    return name[:-7] if name.endswith(":latest") else name


def catalog_with_status(installed_names):
    installed = {normalize_name(n) for n in installed_names}
    items = []
    for entry in CATALOG:
        item = dict(entry)
        item["installed"] = normalize_name(entry["id"]) in installed
        item["group_label"] = GROUP_LABELS[entry["group"]]
        items.append(item)
    return items


def pick_model(candidates, capability, preferred=None):
    """Choose an installed model name that has ``capability``.

    ``candidates`` is a mapping name -> list of capabilities. ``preferred`` wins
    when it has the capability.
    """
    if preferred and capability in candidates.get(preferred, ()):
        return preferred
    # prefer the smallest-looking tag first (e2b before e4b before 12b …) only when
    # nothing else distinguishes them: stable alphabetical order is good enough.
    for name in sorted(candidates):
        if capability in candidates[name]:
            return name
    return None


def pick_coding_model(candidates, preferred=None):
    """Best installed model for the tool-calling programming agent: keeps ``preferred`` if it can call tools,
    otherwise a tool-capable model, 'coder' models first, then the largest-looking tag."""
    if preferred and "tools" in candidates.get(preferred, ()):
        return preferred

    def weight(name):
        match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name.lower())
        return float(match.group(1)) if match else 0.0

    able = [name for name, caps in candidates.items() if "tools" in caps and "embedding" not in caps]
    able.sort(key=lambda n: (0 if "coder" in n.lower() else 1, -weight(n), n))
    return able[0] if able else None
