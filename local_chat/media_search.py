"""Keyless media search for LocalChat: videos, music, movies and images with real links.

Providers (all free, no API key, fixed well-known hosts so there is no SSRF surface):
* YouTube results page  - video / music / trailer links (parsed from ytInitialData; may break if YouTube changes)
* iTunes Search API     - official song / movie records with Apple links and 30 s previews
* Openverse API         - openly licensed images (direct image URL + landing page)
* Wikimedia Commons API - images fallback
* Wikipedia opensearch  - information page for a movie / artist

Every provider returns normalized items:
    {"title", "url", "snippet", "content": "", "kind", "source"}
and never raises: a failing provider only returns []. search_media() runs them in parallel, merges,
de-duplicates and caps the list so the model context stays small (important for slow local models).
"""
import concurrent.futures
import json
import re
import time
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 LocalChat/2.0"
)
TIMEOUT = 8
MAX_BYTES = 2_500_000
MAX_ITEMS = 8
_CACHE = {}
_CACHE_TTL = 600
_CACHE_MAX = 64

KIND_LABELS = {"video": "ویدیو", "music": "آهنگ", "movie": "فیلم / سریال", "image": "تصویر", "web": "وب"}


def _get(url, headers=None, timeout=TIMEOUT):
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "fa,en;q=0.8", **(headers or {})})
    with urlopen(request, timeout=timeout) as response:
        return response.read(MAX_BYTES).decode("utf-8", errors="replace")


def _get_json(url, headers=None, timeout=TIMEOUT):
    return json.loads(_get(url, headers, timeout))


def _item(title, url, snippet, kind, source):
    parts = urlsplit(str(url or ""))
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        return None
    title = " ".join(str(title or "").split())[:200] or url
    return {
        "title": title,
        "url": str(url)[:2048],
        "snippet": " ".join(str(snippet or "").split())[:300],
        "content": "",
        "kind": kind,
        "source": source,
    }


# ---------------------------------------------------------------- YouTube
def _walk(node, key):
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if key in current:
                yield current[key]
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def parse_youtube(page, kind="video", limit=6):
    marker = "ytInitialData"
    start = page.find(marker)
    if start < 0:
        return []
    brace = page.find("{", start)
    if brace < 0:
        return []
    try:
        data, _ = json.JSONDecoder().raw_decode(page[brace:])
    except ValueError:
        return []
    items, seen = [], set()
    for renderer in _walk(data, "videoRenderer"):
        if not isinstance(renderer, dict):
            continue
        video_id = renderer.get("videoId")
        if not isinstance(video_id, str) or not re.fullmatch(r"[\w-]{11}", video_id) or video_id in seen:
            continue
        seen.add(video_id)
        runs = (renderer.get("title") or {}).get("runs") or [{}]
        title = runs[0].get("text", "")
        channel = (((renderer.get("ownerText") or {}).get("runs")) or [{}])[0].get("text", "")
        length = (renderer.get("lengthText") or {}).get("simpleText", "")
        views = (renderer.get("viewCountText") or {}).get("simpleText", "")
        published = (renderer.get("publishedTimeText") or {}).get("simpleText", "")
        snippet = " · ".join(part for part in ("YouTube", channel, length, views, published) if part)
        item = _item(title, f"https://www.youtube.com/watch?v={video_id}", snippet, kind, "YouTube")
        if item:
            items.append(item)
        if len(items) >= limit:
            break
    return items


def youtube(query, kind="video", limit=5):
    try:
        url = "https://www.youtube.com/results?" + urlencode({"search_query": query, "hl": "en"})
        page = _get(url, headers={"Cookie": "CONSENT=YES+1; SOCS=CAI", "Accept": "text/html"})
        return parse_youtube(page, kind, limit)
    except Exception:
        return []


# ---------------------------------------------------------------- iTunes
def parse_itunes(data, kind):
    items = []
    for record in (data or {}).get("results", []):
        if not isinstance(record, dict):
            continue
        title = record.get("trackName") or record.get("collectionName")
        url = record.get("trackViewUrl") or record.get("collectionViewUrl")
        artist = record.get("artistName") or ""
        album = record.get("collectionName") if kind == "music" else ""
        year = str(record.get("releaseDate") or "")[:4]
        genre = record.get("primaryGenreName") or ""
        extra = record.get("shortDescription") or record.get("longDescription") or ""
        snippet = " · ".join(part for part in ("Apple", artist, album, year, genre, str(extra)[:140]) if part)
        item = _item(f"{title} — {artist}" if artist and kind == "music" else title, url, snippet, kind, "Apple iTunes")
        if item:
            preview = record.get("previewUrl")
            if isinstance(preview, str) and preview.startswith("https://"):
                item["preview_url"] = preview[:2048]
            items.append(item)
    return items


def itunes(term, kind, limit=4):
    try:
        media, entity = ("music", "song") if kind == "music" else ("movie", "movie")
        params = {"term": term, "media": media, "entity": entity, "limit": limit}
        return parse_itunes(_get_json("https://itunes.apple.com/search?" + urlencode(params)), kind)
    except Exception:
        return []


# ---------------------------------------------------------------- Wikipedia
def parse_opensearch(data, source, kind):
    try:
        titles, descs, urls = data[1], data[2], data[3]
    except (TypeError, IndexError):
        return []
    items = []
    for title, desc, url in zip(titles, descs, urls):
        item = _item(title, url, desc or source, kind, source)
        if item:
            items.append(item)
    return items


def wikipedia(term, lang, kind, limit=2):
    try:
        host = "fa" if lang == "fa" else "en"
        url = f"https://{host}.wikipedia.org/w/api.php?" + urlencode(
            {"action": "opensearch", "search": term, "limit": limit, "namespace": 0, "format": "json"})
        return parse_opensearch(_get_json(url), f"Wikipedia ({host})", kind)
    except Exception:
        return []


# ---------------------------------------------------------------- images
def parse_openverse(data):
    items = []
    for record in (data or {}).get("results", []):
        if not isinstance(record, dict):
            continue
        direct = record.get("url")
        landing = record.get("foreign_landing_url") or direct
        creator = record.get("creator") or ""
        lic = " ".join(str(part) for part in (record.get("license"), record.get("license_version")) if part)
        snippet = " · ".join(part for part in (record.get("source") or "Openverse", creator, lic.upper()) if part)
        item = _item(record.get("title") or "تصویر", landing, snippet, "image", "Openverse")
        if item:
            if isinstance(direct, str) and direct.startswith(("http://", "https://")):
                item["image_url"] = direct[:2048]
            items.append(item)
    return items


def openverse(term, limit=6):
    try:
        url = "https://api.openverse.org/v1/images/?" + urlencode(
            {"q": term, "page_size": limit, "mature": "false"})
        return parse_openverse(_get_json(url))
    except Exception:
        return []


def parse_commons(data):
    pages = ((data or {}).get("query") or {}).get("pages") or {}
    items = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 0) if isinstance(p, dict) else 0):
        if not isinstance(page, dict):
            continue
        info = (page.get("imageinfo") or [{}])[0]
        title = re.sub(r"^File:", "", str(page.get("title") or "")).rsplit(".", 1)[0]
        landing = info.get("descriptionurl") or info.get("url")
        item = _item(title or "تصویر", landing, "Wikimedia Commons", "image", "Wikimedia Commons")
        if item:
            if str(info.get("url", "")).startswith("https://"):
                item["image_url"] = info["url"][:2048]
            items.append(item)
    return items


def commons(term, limit=5):
    try:
        url = "https://commons.wikimedia.org/w/api.php?" + urlencode({
            "action": "query", "generator": "search", "gsrsearch": f"{term} filetype:bitmap",
            "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url", "format": "json"})
        return parse_commons(_get_json(url))
    except Exception:
        return []


# ---------------------------------------------------------------- orchestration
def _dedupe(items):
    seen, out = set(), []
    for item in items:
        key = item["url"].split("#")[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _run(tasks):
    """Run provider callables in parallel; keep the order of `tasks` in the merged output."""
    results = [[] for _ in tasks]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(tasks))) as pool:
        futures = {pool.submit(task): index for index, task in enumerate(tasks)}
        try:
            for future in concurrent.futures.as_completed(futures, timeout=TIMEOUT + 4):
                try:
                    results[futures[future]] = future.result() or []
                except Exception:
                    results[futures[future]] = []
        except concurrent.futures.TimeoutError:
            pass
    return [item for group in results for item in group]


def search_media(kind, topic, lang="en", kind_word="", web_fallback=None):
    """Return up to MAX_ITEMS link items for kind in {video, music, movie, image}.

    web_fallback(query) -> list of {"title","url","snippet"} is used only when the specialised
    providers returned nothing (e.g. all blocked), so the person still gets links.
    """
    topic = " ".join(str(topic or "").split())[:160]
    if kind not in ("video", "music", "movie", "image") or not topic:
        return []
    cache_key = (kind, topic.lower(), lang)
    cached = _CACHE.get(cache_key)
    if cached and time.monotonic() - cached[0] < _CACHE_TTL:
        return [dict(item) for item in cached[1]]

    if kind == "video":
        tasks = [lambda: youtube(topic, "video", 6)]
        fallback_query = f"{topic} video"
    elif kind == "music":
        tasks = [lambda: itunes(topic, "music", 3), lambda: youtube(f"{topic} official audio", "music", 5)]
        fallback_query = f"{kind_word or 'آهنگ'} {topic} listen"
    elif kind == "movie":
        tasks = [
            lambda: itunes(topic, "movie", 2),
            lambda: wikipedia(f"{topic} film" if lang != "fa" else f"{topic} فیلم", lang, "movie", 2),
            lambda: youtube(f"{topic} official trailer", "video", 3),
        ]
        fallback_query = f"{kind_word or 'فیلم'} {topic} imdb"
    else:
        tasks = [lambda: openverse(topic, 6), lambda: commons(topic, 4)]
        fallback_query = f"{topic} images"

    items = _dedupe(_run(tasks))
    if not items and callable(web_fallback):
        try:
            for raw in web_fallback(fallback_query) or []:
                item = _item(raw.get("title"), raw.get("url"), raw.get("snippet"), kind, "وب")
                if item:
                    items.append(item)
        except Exception:
            pass
    # images: keep a mix of sources; everything else: provider order is already by relevance
    items = items[:MAX_ITEMS]
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)), None)
    if items:
        _CACHE[cache_key] = (time.monotonic(), [dict(item) for item in items])
    return items
