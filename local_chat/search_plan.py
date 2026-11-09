"""Translate natural-language search constraints into a deterministic search plan.

The parser is intentionally lightweight: it does not call the local LLM and never invents
URLs. The user's natural-language request remains the source of truth.
"""
import re
from urllib.parse import urlsplit

_PERSIAN = re.compile(r"[\u0600-\u06FF]")
_DOMAIN = re.compile(r"(?i)(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[\w./%-]*)?")
_ALIASES = {
    "دیجی کالا": "digikala.com", "دیجی‌کالا": "digikala.com", "digikala": "digikala.com",
    "ترب": "torob.com", "torob": "torob.com",
    "ایمالز": "emalls.ir", "emalls": "emalls.ir",
    "تکنولایف": "technolife.ir", "technolife": "technolife.ir",
    "لیون کامپیوتر": "lioncomputer.com", "lioncomputer": "lioncomputer.com",
    "مقداد آی تی": "meghdadit.com", "meghdadit": "meghdadit.com",
    "شهر کامپیوتر": "shahrcomputer.com", "shahrcomputer": "shahrcomputer.com",
    "آپارات": "aparat.com", "aparat": "aparat.com",
    "یوتیوب": "youtube.com", "youtube": "youtube.com",
    "ذره بین": "zarebin.ir", "ذره‌بین": "zarebin.ir", "zarebin": "zarebin.ir",
    "گردو": "gerdoo.me", "gerdoo": "gerdoo.me",
}
_KNOWN_FILE_EXTENSIONS = {
    "pdf", "doc", "docx", "odt", "rtf", "xls", "xlsx", "ods", "csv",
    "ppt", "pptx", "odp", "zip", "rar", "7z", "tar", "gz", "json", "xml",
    "txt", "md", "epub", "mp3", "mp4", "wav", "jpg", "jpeg", "png", "gif",
    "webp", "psd", "ai", "dwg", "dxf", "stl", "obj", "exe", "msi", "apk", "iso",
}
_FILE_EXTENSIONS = {
    "pdf": "pdf", "پی دی اف": "pdf", "پی‌دی‌اف": "pdf",
    "doc": "doc", "docx": "docx", "word": "docx", "ورد": "docx",
    "odt": "odt", "rtf": "rtf",
    "xls": "xls", "xlsx": "xlsx", "excel": "xlsx", "اکسل": "xlsx", "csv": "csv",
    "ppt": "ppt", "pptx": "pptx", "powerpoint": "pptx",
    "zip": "zip", "rar": "rar", "7z": "7z", "json": "json", "xml": "xml",
    "txt": "txt", "md": "md", "epub": "epub", "mp3": "mp3", "mp4": "mp4",
    "فایل": None, "فایل ها": None, "فایل‌ها": None, "سند": None, "اسناد": None,
}


def _norm(value):
    value = str(value or "").replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"\s+", " ", value).strip()


def _domain(value):
    value = str(value or "").strip().lower().rstrip(".,؛،)")
    if not value:
        return ""
    if "://" not in value:
        value = "https://" + value
    try:
        host = (urlsplit(value).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""
    if not host or "." not in host or len(host) > 253:
        return ""
    if not re.fullmatch(r"[a-z0-9.-]+", host) or ".." in host:
        return ""
    return host


def _extract_domains(text):
    found = []
    for raw in _DOMAIN.findall(text):
        host = _domain(raw)
        if host and host not in found:
            found.append(host)
    lowered = _norm(text).casefold()
    for alias, host in _ALIASES.items():
        if alias.casefold() in lowered and host not in found:
            found.append(host)
    return found


def parse_search_request(raw_query, supplied=None):
    """Translate natural-language restrictions and optional model arguments into a safe plan."""
    original = _norm(raw_query)[:1000]
    supplied = supplied if isinstance(supplied, dict) else {}
    lower = original.casefold()
    alias_pattern = "|".join(re.escape(name) for name in _ALIASES)

    only_site = bool(
        re.search(r"(?:فقط|صرفا|صرفاً)\s*(?:در\s*)?(?:سایت|دامنه|وبسایت|وب\s*سایت)|only\s+(?:the\s+)?(?:site|domain|website)", lower)
        or re.search(r"(?:فقط|صرفا|صرفاً)\s*(?:در\s*)?(?:" + alias_pattern + r")", lower)
        or re.search(r"only\s+(?:on\s+)?(?:" + alias_pattern + r")", lower)
    )
    exclude_site = bool(re.search(
        r"(?:به\s*جز|به\s*غیر\s*از|غیر\s*از|به\s*استثنای)|(?:except|excluding|exclude)", lower
    ))
    fa_only = bool(re.search(
        r"(?:فقط|صرفا|صرفاً)\s*(?:سایت(?:های|\s*های)?|منابع?)\s*فارسی|"
        r"(?:سایت(?:های|\s*های)?|منابع?)\s*فارسی\s*(?:فقط|باشه|باشد)|"
        r"only\s+persian\s+(?:sites|sources)", lower
    ))
    iran_only = bool(re.search(
        r"(?:فقط|صرفا|صرفاً)\s*(?:سایت(?:های|\s*های)?|فروشگاه(?:های|\s*های)?|منابع?)\s*ایرانی|"
        r"(?:سایت(?:های|\s*های)?|فروشگاه(?:های|\s*های)?|منابع?)\s*ایرانی\s*(?:فقط|باشه|باشد)|"
        r"iranian\s+(?:sites|stores|sources)\s+only", lower
    ))
    only_site = only_site and not (fa_only or iran_only)

    mentioned_domains = _extract_domains(original)
    allowed_raw = supplied.get("allowed_domains")
    excluded_raw = supplied.get("excluded_domains")
    allowed = [_domain(x) for x in allowed_raw] if isinstance(allowed_raw, list) else []
    excluded = [_domain(x) for x in excluded_raw] if isinstance(excluded_raw, list) else []
    if only_site and not allowed:
        if exclude_site:
            allow_clause = re.search(
                r"(?:فقط|صرفا|صرفاً)\s*(?:در\s*)?(?:سایت|دامنه|وبسایت|وب\s*سایت)?\s*(.*?)"
                r"(?=(?:به\s*جز|به\s*غیر\s*از|غیر\s*از|به\s*استثنای|except|excluding|exclude)|$)",
                original, flags=re.I,
            )
            allowed = _extract_domains(allow_clause.group(1)) if allow_clause else []
        else:
            allowed = mentioned_domains
    if exclude_site and not excluded:
        exclude_clause = re.search(
            r"(?:به\s*جز|به\s*غیر\s*از|غیر\s*از|به\s*استثنای|except|excluding|exclude)\s*(.*)$",
            original, flags=re.I,
        )
        excluded = _extract_domains(exclude_clause.group(1)) if exclude_clause else []

    content_types = supplied.get("content_types")
    if not isinstance(content_types, list):
        content_types = []
    content_types = [
        str(x).lower() for x in content_types
        if str(x).lower() in _KNOWN_FILE_EXTENSIONS
    ]
    file_search = bool(re.search(r"\bfiles?\b|\bdocuments?\b|فایل|اسناد?|مدارک", lower))
    for term, extension in _FILE_EXTENSIONS.items():
        term_found = (
            bool(re.search(r"\b" + re.escape(term) + r"\b", lower))
            if re.fullmatch(r"[a-z0-9]+", term) else term in lower
        )
        if term_found and extension and extension not in content_types:
            content_types.append(extension)
    if re.search(r"(?:فایل|file)\s+ai\b|\bfiletype:ai\b", lower):
        if "ai" not in content_types:
            content_types.append("ai")
    for extension in sorted(_KNOWN_FILE_EXTENSIONS):
        if extension == "ai":
            continue
        explicit_filetype = re.search(r"\bfiletype:\s*" + re.escape(extension) + r"\b", lower)
        mentioned_with_file = file_search and re.search(r"\b" + re.escape(extension) + r"\b", lower)
        if (explicit_filetype or mentioned_with_file) and extension not in content_types:
            content_types.append(extension)
    if content_types:
        file_search = True

    query = original
    patterns = [
        r"(?:فقط|صرفا|صرفاً)\s*(?:سایت(?:های|\s*های)?|منابع?)\s*(?:فارسی|ایرانی)",
        r"(?:فقط|صرفا|صرفاً)\s*(?:فروشگاه(?:های|\s*های)?|سایت(?:های|\s*های)?|منابع?)\s*ایرانی",
        r"(?:فقط|صرفا|صرفاً)\s*(?:در\s*)?(?:سایت|دامنه|وبسایت|وب\s*سایت)\s+[\w.-]+\.[a-z]{2,}",
        r"(?:به\s*جز|به\s*غیر\s*از|غیر\s*از|به\s*استثنای)\s+(?:سایت\s+)?[\w.-]+\.[a-z]{2,}",
        r"only\s+persian\s+(?:sites|sources)",
        r"iranian\s+(?:sites|stores|sources)\s+only",
        r"only\s+(?:the\s+)?(?:site|domain|website)\s+[\w.-]+",
        r"(?:except|excluding|exclude)\s+[\w.-]+",
    ]
    for pattern in patterns:
        query = re.sub(pattern, " ", query, flags=re.I)
    if only_site:
        query = re.sub(r"(?:فقط|صرفا|صرفاً)\s*(?:در\s*)?(?:سایت\s*)?(?:" + alias_pattern + r")", " ", query, flags=re.I)
        query = re.sub(r"only\s+(?:on\s+)?" + r"(?:" + alias_pattern + r")", " ", query, flags=re.I)
    if exclude_site:
        query = re.sub(r"(?:به\s*جز|به\s*غیر\s*از|غیر\s*از|به\s*استثنای)\s*(?:سایت\s*)?(?:" + alias_pattern + r")", " ", query, flags=re.I)
        query = re.sub(r"(?:except|excluding|exclude)\s+(?:" + alias_pattern + r")", " ", query, flags=re.I)
    if only_site or exclude_site:
        for raw in _DOMAIN.findall(original):
            query = query.replace(raw, " ")
        for alias in _ALIASES:
            query = re.sub(re.escape(alias), " ", query, flags=re.I)
        query = re.sub(r"(?:فقط|صرفا|صرفاً)\s*(?:در\s*)?(?:سایت|دامنه|وبسایت|وب\s*سایت)", " ", query, flags=re.I)
        query = re.sub(r"(?:به\s*جز|به\s*غیر\s*از|غیر\s*از|به\s*استثنای)", " ", query, flags=re.I)
    query = re.sub(r"[،,;؛]+", " ", query)
    query = re.sub(r"\s+", " ", query).strip(" -:")
    if len(query) < 2:
        query = original

    allowed = list(dict.fromkeys(x for x in allowed if x))
    excluded = list(dict.fromkeys(x for x in excluded if x and x not in allowed))
    language = "fa" if fa_only else supplied.get("language")
    if language not in ("fa", "en"):
        language = None
    region = "IR" if iran_only else supplied.get("region")
    if region != "IR":
        region = None

    return {
        "query": query[:500],
        "constraints": {
            "language": language,
            "region": region,
            "allowed_domains": allowed,
            "excluded_domains": excluded,
            "content_types": content_types,
            "file_search": file_search,
        },
        "strict_domains": bool(only_site or supplied.get("strict_domains")),
        "strategy": str(supplied.get("strategy") or "auto"),
    }


def host_allowed(host, constraints):
    """Apply explicit domain allow/deny constraints without fuzzy substring matching."""
    host = str(host or "").lower().rstrip(".")
    allowed = constraints.get("allowed_domains") or []
    excluded = constraints.get("excluded_domains") or []
    if allowed and not any(host == d or host.endswith("." + d) for d in allowed):
        return False
    if any(host == d or host.endswith("." + d) for d in excluded):
        return False
    return True


def result_matches_constraints(item, constraints):
    host = _domain(item.get("url", ""))
    if not host_allowed(host, constraints):
        return False
    language = constraints.get("language")
    if language == "fa":
        sample = " ".join((item.get("title") or "", item.get("snippet") or "", item.get("content") or ""))
        letters = [c for c in sample if c.isalpha()]
        if len(letters) >= 10 and sum(1 for c in letters if "\u0600" <= c <= "\u06ff") / len(letters) < 0.08:
            return False
    region = constraints.get("region")
    if region == "IR":
        iranian_hosts = ("digikala.com", "torob.com", "emalls.ir", "technolife.ir", "lioncomputer.com",
                         "meghdadit.com", "shahrcomputer.com", "zarebin.ir", "gerdoo.me")
        if not (host.endswith(".ir") or any(host == d or host.endswith("." + d) for d in iranian_hosts)):
            return False
    file_search = bool(constraints.get("file_search"))
    types = constraints.get("content_types") or []
    if file_search:
        path = urlsplit(item.get("url", "")).path.lower()
        mime = str(item.get("mime_type") or "").lower()
        if not any(path.endswith("." + ext) for ext in _KNOWN_FILE_EXTENSIONS) and not (
            mime.startswith(("application/", "audio/", "video/", "image/")) and mime != "application/xhtml+xml"
        ):
            return False
    if types:
        path = urlsplit(item.get("url", "")).path.lower()
        if not any(path.endswith("." + ext) for ext in types):
            # Search snippets sometimes link to landing pages rather than files; keep them if the provider labels the type.
            if str(item.get("mime_type") or "").lower() not in {"application/pdf", "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "application/zip"}:
                return False
    return True
