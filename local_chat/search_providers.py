"""Legacy Google search provider adapters used by the multi-provider search engine.

Serper and Google Custom Search return structured JSON. The brittle keyless Google HTML
parser remains for compatibility and focused parser tests, but the active search router
does not rely on scraping Google result pages. Provider failures are isolated by
local_chat.search_engine and never expose API keys in error messages.

Every provider returns a list of {"title", "url", "snippet"} dictionaries or raises
SearchProviderError.
"""
import json
import re
from html import unescape
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

MAX_RESULTS = 5
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_PERSIAN = re.compile(r"[\u0600-\u06FF]")


class SearchProviderError(Exception):
    pass


def language_for(query):
    return "fa" if _PERSIAN.search(query or "") else "en"


def _read_json(request, timeout=15):
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES)
    except HTTPError as exc:
        raise SearchProviderError(_http_message(exc.code)) from None
    except (URLError, TimeoutError, OSError):
        raise SearchProviderError("اتصال به سرویس جست‌وجوی گوگل برقرار نشد.") from None
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        raise SearchProviderError("پاسخ سرویس جست‌وجوی گوگل معتبر نیست.") from None
    if not isinstance(data, dict):
        raise SearchProviderError("پاسخ سرویس جست‌وجوی گوگل معتبر نیست.")
    return data


def _http_message(code):
    if code in (401, 403):
        return "کلید API گوگل/Serper نامعتبر است یا سهمیه یا دسترسی ندارد (HTTP %d)." % code
    if code == 429:
        return "سهمیهٔ جست‌وجو تمام شده یا درخواست‌ها زیاد بوده‌اند (HTTP 429)."
    return "سرویس جست‌وجوی گوگل خطا داد (HTTP %d)." % code


def _clean(results):
    cleaned = []
    seen = set()
    for item in results:
        url = str(item.get("url") or "")[:2048]
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            continue
        if parts.username is not None or parts.password is not None:
            continue
        key = (parts.hostname, parts.path.rstrip("/"), parts.query)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({
            "title": " ".join(unescape(str(item.get("title") or "")).split())[:300] or url,
            "url": url,
            "snippet": " ".join(unescape(str(item.get("snippet") or "")).split())[:1000],
            "content": "",
        })
        if len(cleaned) >= MAX_RESULTS:
            break
    return cleaned


def search_serper(query, api_key):
    body = json.dumps({"q": query, "num": MAX_RESULTS + 3, "hl": language_for(query)}).encode("utf-8")
    request = Request(
        "https://google.serper.dev/search", data=body, method="POST",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    data = _read_json(request)
    organic = data.get("organic")
    if not isinstance(organic, list):
        raise SearchProviderError("پاسخ Serper نتیجه‌ای نداشت.")
    results = _clean({"title": r.get("title"), "url": r.get("link"), "snippet": r.get("snippet")}
                     for r in organic if isinstance(r, dict))
    # a direct answer box is very useful context for the model
    answer = data.get("answerBox")
    if isinstance(answer, dict) and results:
        text = answer.get("answer") or answer.get("snippet")
        if text and not results[0]["snippet"]:
            results[0]["snippet"] = str(text)[:1000]
    return results


def search_google_cse(query, api_key, cx):
    params = urlencode({
        "key": api_key, "cx": cx, "q": query, "num": MAX_RESULTS, "hl": language_for(query),
    })
    request = Request(f"https://www.googleapis.com/customsearch/v1?{params}",
                      headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    data = _read_json(request)
    items = data.get("items")
    if items is None:
        return []
    if not isinstance(items, list):
        raise SearchProviderError("پاسخ Google CSE معتبر نیست.")
    return _clean({"title": r.get("title"), "url": r.get("link"), "snippet": r.get("snippet")}
                  for r in items if isinstance(r, dict))


class _GoogleHtmlParser(HTMLParser):
    """Very tolerant: collects <a href> that wrap a heading and the text after it."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._href = None
        self._depth = 0
        self._parts = []
        self._heading = []
        self._in_heading = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href") and self._href is None:
            self._href = attrs["href"]
            self._parts = []
            self._heading = []
            self._depth = 1
        elif self._href is not None:
            self._depth += 1
            if tag in ("h3", "h2"):
                self._in_heading += 1

    def handle_endtag(self, tag):
        if self._href is None:
            return
        if tag in ("h3", "h2") and self._in_heading:
            self._in_heading -= 1
        self._depth -= 1
        if self._depth <= 0:
            title = " ".join(" ".join(self._heading).split())
            if title:
                self.results.append({"href": self._href, "title": title})
            self._href = None
            self._in_heading = 0

    def handle_data(self, data):
        if self._href is not None and self._in_heading:
            self._heading.append(data)


_BLOCK_MARKERS = (
    "enablejs", "unusual traffic", "/sorry/", "consent.google.com", "captcha",
    "httpservice/retry/enablejs", "noscript",
)


def parse_google_html(page):
    lowered = page.lower()
    parser = _GoogleHtmlParser()
    parser.feed(page)
    results = []
    for item in parser.results:
        href = item["href"]
        if href.startswith("/url?"):
            href = parse_qs(urlsplit(href).query).get("q", [""])[0]
        parts = urlsplit(href)
        host = (parts.hostname or "").lower()
        if parts.scheme not in ("http", "https") or not host:
            continue
        if host.endswith("google.com") or host.endswith("googleusercontent.com") or host.endswith("gstatic.com"):
            continue
        results.append({"title": item["title"], "url": href, "snippet": ""})
    cleaned = _clean(results)
    if not cleaned and any(marker in lowered for marker in _BLOCK_MARKERS):
        raise SearchProviderError("گوگل بدون جاوااسکریپت پاسخ نمی‌دهد یا درخواست را مسدود کرده است.")
    return cleaned


def search_google_keyless(query):
    params = urlencode({"q": query, "hl": language_for(query), "num": 10, "pws": 0})
    request = Request(
        f"https://www.google.com/search?{params}",
        headers={"User-Agent": USER_AGENT, "Accept": "text/html", "Accept-Language": "fa,en;q=0.8"},
    )
    try:
        with urlopen(request, timeout=12) as response:
            page = response.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise SearchProviderError(
            "گوگل جست‌وجوی بدون کلید را رد کرد (HTTP %d)." % exc.code) from None
    except (URLError, TimeoutError, OSError):
        raise SearchProviderError("اتصال به گوگل برقرار نشد.") from None
    results = parse_google_html(page)
    if not results:
        raise SearchProviderError("گوگل نتیجه‌ای که قابل خواندن باشد برنگرداند.")
    return results

def search_bing_rss(query):
    """Keyless Bing RSS adapter. It is a best-effort public endpoint, not a guaranteed API."""
    from xml.etree import ElementTree

    params = urlencode({"q": query, "format": "rss", "setlang": language_for(query)})
    request = Request(
        "https://www.bing.com/search?" + params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml"},
    )
    try:
        with urlopen(request, timeout=12) as response:
            raw = response.read(MAX_RESPONSE_BYTES)
    except HTTPError as exc:
        raise SearchProviderError("Bing RSS خطا داد (HTTP %d)." % exc.code) from None
    except (URLError, TimeoutError, OSError):
        raise SearchProviderError("اتصال به Bing RSS برقرار نشد.") from None
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        raise SearchProviderError("پاسخ Bing RSS معتبر نیست.") from None
    rows = []
    for node in root.findall(".//item"):
        title = node.findtext("title") or ""
        url = node.findtext("link") or ""
        snippet = node.findtext("description") or ""
        if url:
            rows.append({"title": title, "url": url, "snippet": snippet})
        if len(rows) >= MAX_RESULTS:
            break
    return _clean(rows)


def search_searxng(query, instance_url, language=None, categories=None):
    """Query a user-configured SearXNG instance; public instances may disable JSON output."""
    from urllib.parse import urlsplit, urlunsplit

    base = str(instance_url or "").strip().rstrip("/")
    parts = urlsplit(base)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise SearchProviderError("آدرس SearXNG باید یک نشانی HTTPS معتبر باشد.")
    params = {"q": query, "format": "json", "language": language or language_for(query), "safesearch": "1"}
    if categories:
        params["categories"] = categories
    request = Request(
        base + "/search?" + urlencode(params),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=12) as response:
            raw = response.read(MAX_RESPONSE_BYTES)
    except HTTPError as exc:
        raise SearchProviderError("SearXNG خطا داد (HTTP %d؛ ممکن است خروجی JSON غیرفعال باشد)." % exc.code) from None
    except (URLError, TimeoutError, OSError):
        raise SearchProviderError("اتصال به نمونهٔ SearXNG برقرار نشد.") from None
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        raise SearchProviderError("پاسخ SearXNG JSON معتبر نیست.") from None
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise SearchProviderError("ساختار پاسخ SearXNG معتبر نیست.")
    return _clean(
        {"title": item.get("title"), "url": item.get("url"), "snippet": item.get("content")}
        for item in results if isinstance(item, dict)
    )
