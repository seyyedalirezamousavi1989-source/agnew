"""Multi-provider, intent-aware web search for LocalChat.

The local model never invents result URLs. Providers supply URLs; this module normalizes,
deduplicates, ranks and labels them. Product and video searches expand into domain-targeted
queries, especially for Persian/Iranian commerce and Aparat.
"""
import concurrent.futures
import ipaddress
import json
import re
import time
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from . import search_providers, search_plan

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 LocalChat/2.1"
)
TIMEOUT = 12
MAX_RESULTS = 12
MAX_QUERIES = 12
MAX_BYTES = 2 * 1024 * 1024
CACHE_TTL = 180
_CACHE = {}
_CACHE_LIMIT = 96

_PRODUCT = re.compile(
    r"(?:کارت\s*گرافیک|گرافیک|پردازنده|سی\s*پی\s*یو|مادربرد|رم|حافظه|اس\s*اس\s*دی|"
    r"مانیتور|لپ\s*تاپ|موس|کیبورد|پاور|کیس|محصول|گوشی|موبایل|تبلت|آیفون|شیائومی|هواوی|هدفون|هندزفری|اسپیکر|"
    r"دوربین|کنسول|تلویزیون|ساعت\s*هوشمند|کفش|لباس|کتاب|لوازم\s*خانگی|"
    r"iphone|ipad|macbook|galaxy|pixel|playstation|ps5|xbox|headphones|camera|product|gpu|cpu|"
    r"rtx\s*\d{3,4}|gtx\s*\d{3,4}|rx\s*\d{3,4})", re.I
)
_VIDEO = re.compile(r"(?:آپارات|ویدیو|ویدئو|کلیپ|فیلم|تریلر|aparat|youtube|video|watch)", re.I)
_PERSIAN = re.compile(r"[\u0600-\u06FF]")
_IRAN_STORES = (
    "digikala.com/product",
    "torob.com/p",
    "emalls.ir",
    "technolife.ir",
    "lioncomputer.com",
    "meghdadit.com",
    "shahrcomputer.com",
    "mobile140.com",
    "kalatik.com",
    "janebi.com",
    "basalam.com",
)
_VIDEO_SITES = ("aparat.com/v", "youtube.com/watch", "namasha.com", "tamasha.com")


class SearchError(RuntimeError):
    pass


def _normal(text):
    text = str(text or "").replace("\u200c", " ").replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")
    return re.sub(r"\s+", " ", text).strip()


def classify(query):
    q = _normal(query)
    if _PRODUCT.search(q):
        return "product"
    if _VIDEO.search(q):
        return "video"
    return "web"


def _safe_url(value):
    try:
        raw = str(value or "")
        p = urlsplit(raw)
        host = (p.hostname or "").lower().rstrip(".")
        if p.scheme not in ("http", "https") or not host or len(raw) > 2048:
            return False
        if p.username is not None or p.password is not None:
            return False
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            return False
        try:
            address = ipaddress.ip_address(host)
            if not address.is_global:
                return False
        except ValueError:
            pass
        return True
    except ValueError:
        return False


def _normalize_item(item, provider, query, kind):
    if not isinstance(item, dict):
        return None
    url = str(item.get("url") or item.get("link") or "").strip()
    if not _safe_url(url):
        return None
    p = urlsplit(url)
    host = (p.hostname or "").lower().rstrip(".")
    title = " ".join(str(item.get("title") or "").split())[:300]
    snippet = " ".join(str(item.get("snippet") or item.get("description") or "").split())[:1000]
    if not title:
        title = host
    product_path = bool(re.search(r"/(?:product|products|p|item|dp|prd|کالا)(?:/|-|_|\?|$)", p.path, re.I))
    video_url = (
        ("aparat.com" in host and re.search(r"/(?:v|video)/", p.path, re.I))
        or ("youtube.com" in host and p.path == "/watch" and "v=" in p.query)
        or ("youtu.be" in host and len(p.path.strip("/")) >= 6)
    )
    mime_type = str(item.get("mime_type") or "").lower()
    file_type = next((ext for ext in sorted(search_plan._KNOWN_FILE_EXTENSIONS)
                      if p.path.lower().endswith("." + ext)), "")
    is_file = bool(file_type or (
        mime_type.startswith(("application/", "audio/", "video/", "image/"))
        and mime_type != "application/xhtml+xml"
    ))
    title_match = _token_overlap(query, title + " " + snippet)
    score = 0.25 + min(title_match, 0.45)
    if kind == "product":
        if any(host == d or host.endswith("." + d) for d in (
            "digikala.com", "torob.com", "emalls.ir", "technolife.ir",
            "lioncomputer.com", "meghdadit.com", "shahrcomputer.com", "mobile140.com", "kalatik.com",
            "janebi.com", "basalam.com"
        )):
            score += 0.16
        if product_path:
            score += 0.22
        if not product_path and any(x in p.path.lower() for x in ("/search", "/category", "/brand")):
            score -= 0.08
    elif kind == "video":
        if video_url:
            score += 0.28
        if "aparat.com" in host:
            score += 0.18
    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "content": "",
        "source": provider,
        "provider": provider,
        "search_kind": kind,
        "match_score": round(min(1.0, score), 3),
        "link_type": "product" if kind == "product" and product_path else (
            "video" if kind == "video" and video_url else ("file" if is_file else "webpage")
        ),
        "file_type": file_type,
        "verified": False,
        "query": query,
        "mime_type": mime_type,
    }


def _token_overlap(query, text):
    def tokens(value):
        return {x.casefold() for x in re.findall(r"[a-zA-Z0-9\u0600-\u06FF]+", value) if len(x) > 1}
    a, b = tokens(query), tokens(text)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _dedupe(items):
    out, seen = [], set()
    for item in items:
        p = urlsplit(item["url"])
        key = (p.hostname.lower(), p.path.rstrip("/"), p.query)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def search_brave(query, api_key, limit=8):
    """Brave's independent index; API key is optional and stored only server-side."""
    if not api_key:
        raise search_providers.SearchProviderError("کلید Brave Search تنظیم نشده است.")
    params = {
        "q": query, "count": min(20, max(1, limit)),
        "safesearch": "moderate",
        "search_lang": "fa" if _PERSIAN.search(query) else "en",
        "country": "IR" if _PERSIAN.search(query) else "US",
    }
    request = Request(
        "https://api.search.brave.com/res/v1/web/search?" + urlencode(params),
        headers={"Accept": "application/json", "X-Subscription-Token": api_key, "User-Agent": USER_AGENT},
    )
    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read(MAX_BYTES)
    except HTTPError as exc:
        raise search_providers.SearchProviderError(
            "Brave Search خطا داد (HTTP %d)." % exc.code
        ) from None
    except (URLError, TimeoutError, OSError):
        raise search_providers.SearchProviderError("اتصال به Brave Search برقرار نشد.") from None
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        raise search_providers.SearchProviderError("پاسخ Brave Search معتبر نیست.") from None
    web = data.get("web") if isinstance(data, dict) else None
    rows = web.get("results", []) if isinstance(web, dict) else []
    return [
        {"title": row.get("title"), "url": row.get("url"), "snippet": row.get("description")}
        for row in rows if isinstance(row, dict)
    ]


_PROVIDER_STATS = {}
_STATS_LOCK = threading.Lock()
_PROVIDER_PRIOR = {
    "SearXNG": 0.72, "Bing RSS": 0.67, "DuckDuckGo": 0.62,
    "Google / Serper": 0.60, "Brave Search": 0.60, "Google CSE": 0.55,
}


def _record_provider(name, language, ok, elapsed, result_count):
    key = (name, language or "unknown")
    with _STATS_LOCK:
        state = _PROVIDER_STATS.setdefault(key, {
            "successes": 0, "failures": 0, "latency": max(0.1, elapsed), "results": 0.0,
        })
        state["successes" if ok else "failures"] += 1
        state["latency"] = 0.7 * state["latency"] + 0.3 * max(0.01, elapsed)
        state["results"] = 0.7 * state["results"] + 0.3 * max(0, result_count)


def _provider_score(name, language):
    with _STATS_LOCK:
        state = _PROVIDER_STATS.get((name, language or "unknown"))
        if not state:
            return _PROVIDER_PRIOR.get(name, 0.5)
        total = state["successes"] + state["failures"]
        reliability = state["successes"] / max(1, total)
        volume = min(1.0, state["results"] / 5.0)
        latency_bonus = 1.0 / (1.0 + state["latency"] / 8.0)
        return 0.55 * reliability + 0.30 * volume + 0.15 * latency_bonus


def _provider_calls(query, config, ddg_search, constraints=None):
    """Free providers are tried first; paid APIs are a fallback, not the default."""
    constraints = constraints or {}
    engine = config.get("engine", "auto")
    language = constraints.get("language") or ("fa" if _PERSIAN.search(query) else "en")
    calls = []
    if engine in ("auto", "duckduckgo"):
        calls.append(("DuckDuckGo", lambda q=query: ddg_search(q)))
    if engine == "auto":
        calls.append(("Bing RSS", lambda q=query: search_providers.search_bing_rss(q)))
    if engine in ("auto", "searxng") and config.get("searxng_url"):
        calls.append(("SearXNG", lambda q=query: search_providers.search_searxng(
            q, config["searxng_url"], language=language)))
    if engine in ("google", "brave"):
        if config.get("serper_api_key") and engine == "google":
            calls.append(("Google / Serper", lambda q=query: search_providers.search_serper(q, config["serper_api_key"])))
        if config.get("google_api_key") and config.get("google_cx") and engine == "google":
            calls.append(("Google CSE", lambda q=query: search_providers.search_google_cse(
                q, config["google_api_key"], config["google_cx"])))
        if config.get("brave_api_key") and engine in ("google", "brave"):
            calls.append(("Brave Search", lambda q=query: search_brave(q, config["brave_api_key"])))
    if engine == "brave" and not config.get("brave_api_key"):
        raise SearchError("برای انتخاب مستقیم Brave باید کلید API تنظیم شود.")
    if engine == "searxng" and not config.get("searxng_url"):
        raise SearchError("برای انتخاب مستقیم SearXNG باید نشانی نمونهٔ آن در تنظیمات ثبت شود.")
    if engine not in ("auto", "duckduckgo") and not any(name == "DuckDuckGo" for name, _ in calls):
        calls.append(("DuckDuckGo", lambda q=query: ddg_search(q)))
    return calls


def _paid_provider_calls(query, config):
    """Optional paid/keyed sources, invoked only if the free pass is weak."""
    calls = []
    if config.get("serper_api_key"):
        calls.append(("Google / Serper", lambda q=query: search_providers.search_serper(q, config["serper_api_key"])))
    if config.get("google_api_key") and config.get("google_cx"):
        calls.append(("Google CSE", lambda q=query: search_providers.search_google_cse(
            q, config["google_api_key"], config["google_cx"])))
    if config.get("brave_api_key"):
        calls.append(("Brave Search", lambda q=query: search_brave(q, config["brave_api_key"])))
    return calls


def _query_plan(query, kind, constraints=None):
    """Expand only where intent/constraints justify it; never override an explicit site scope."""
    constraints = constraints or {}
    q = _normal(query)
    queries = [q]
    domains = list(constraints.get("allowed_domains") or [])
    if domains:
        queries.extend(f"site:{domain} {q}" for domain in domains)
    elif constraints.get("region") == "IR":
        iran_domains = (
            "digikala.com", "torob.com", "emalls.ir", "technolife.ir",
            "meghdadit.com", "shahrcomputer.com",
        )
        queries.extend(f"site:{domain} {q}" for domain in iran_domains)
        queries.append(f"site:.ir {q}")
    elif kind == "product":
        queries.extend(f"site:{domain} {q}" for domain in _IRAN_STORES)
    elif kind == "video":
        queries.extend(f"site:{domain} {q}" for domain in _VIDEO_SITES)
    for ext in constraints.get("content_types") or []:
        queries[0] += f" filetype:{ext}"
    if constraints.get("file_search") and not constraints.get("content_types"):
        queries.extend(f"{q} filetype:{ext}" for ext in ("pdf", "docx", "xlsx", "zip"))
    return list(dict.fromkeys(queries))[:MAX_QUERIES]


def _run_query(calls, language=None):
    """Run providers concurrently and learn their reliability/latency/result yield."""
    results, errors, successful = [], [], []
    if not calls:
        return results, errors, successful
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(calls))) as pool:
        submitted = {}
        for name, call in calls:
            submitted[pool.submit(call)] = (name, time.monotonic())
        for future in concurrent.futures.as_completed(submitted):
            name, started = submitted[future]
            elapsed = time.monotonic() - started
            try:
                rows = future.result() or []
                successful.append(name)
                _record_provider(name, language, True, elapsed, len(rows))
                results.extend((name, row) for row in rows)
            except Exception as exc:
                _record_provider(name, language, False, elapsed, 0)
                errors.append(f"{name}: {str(exc)[:180]}")
    return results, errors, successful


def search(query, config, ddg_search, page_fetch=None, max_results=MAX_RESULTS, search_options=None):
    """Search using natural-language constraints; URLs always originate from providers."""
    original_query = _normal(query)[:1000]
    if not original_query:
        raise SearchError("عبارت جست‌وجو خالی است.")
    plan = search_plan.parse_search_request(original_query, search_options)
    query = _normal(plan["query"])[:500]
    constraints = plan["constraints"]
    if plan.get("strict_domains") and not constraints.get("allowed_domains"):
        raise SearchError("برای جست‌وجوی محدود به یک سایت، نام دامنه یا نشانی سایت را هم مشخص کن.")
    if len(query) < 2:
        raise SearchError("عبارت جست‌وجوی قابل‌استفاده پس از اعمال محدودیت‌ها باقی نماند.")
    kind = classify(query)
    cache_key = (
        query.casefold(), kind, json.dumps(constraints, sort_keys=True, ensure_ascii=False),
        config.get("engine", "auto"), str(config.get("searxng_url") or ""),
        str(config.get("serper_api_key") or "")[-4:], str(config.get("brave_api_key") or "")[-4:],
        str(config.get("google_api_key") or "")[-4:], str(config.get("google_cx") or ""),
    )
    cached = _CACHE.get(cache_key)
    if cached and time.monotonic() - cached[0] < CACHE_TTL:
        payload = dict(cached[1])
        payload["results"] = [dict(x) for x in cached[1]["results"]]
        payload["cached"] = True
        return payload

    all_rows, errors = [], []
    query_plan = _query_plan(query, kind, constraints)
    base_calls = _provider_calls(query_plan[0], config, ddg_search, constraints)
    language = constraints.get("language") or ("fa" if _PERSIAN.search(query) else "en")
    base_rows, base_errors, successful = _run_query(base_calls, language)
    all_rows.extend(base_rows)
    errors.extend(base_errors)

    providers_with_hits = list(dict.fromkeys(name for name, _row in all_rows))
    targeted_provider = max(
        providers_with_hits or ["DuckDuckGo"],
        key=lambda name: _provider_score(name, language),
    )
    if len(query_plan) > 1 and targeted_provider:
        targeted_calls = []
        for targeted_query in query_plan[1:]:
            def call_targeted(q=targeted_query, provider=targeted_provider):
                if provider == "SearXNG":
                    language = constraints.get("language") or ("fa" if _PERSIAN.search(q) else "en")
                    return search_providers.search_searxng(q, config["searxng_url"], language=language)
                if provider == "Bing RSS":
                    return search_providers.search_bing_rss(q)
                if provider == "Google / Serper":
                    return search_providers.search_serper(q, config["serper_api_key"])
                if provider == "Brave Search":
                    return search_brave(q, config["brave_api_key"])
                if provider == "Google CSE":
                    return search_providers.search_google_cse(q, config["google_api_key"], config["google_cx"])
                return ddg_search(q)
            targeted_calls.append((targeted_provider, call_targeted))
        targeted_rows, targeted_errors, _targeted_successful = _run_query(targeted_calls, language)
        all_rows.extend(targeted_rows)
        errors.extend(targeted_errors)

    # Paid APIs are only consulted after free base + domain-targeted discovery remains weak.
    preliminary = []
    for provider, row in all_rows:
        item = _normalize_item(row, provider, query, kind)
        if item and search_plan.result_matches_constraints(item, constraints):
            preliminary.append(item)
    strong_hits = [item for item in _dedupe(preliminary) if item["match_score"] >= 0.35]
    if config.get("engine", "auto") == "auto" and len(strong_hits) < 3:
        paid_calls = _paid_provider_calls(query, config)
        if paid_calls:
            paid_rows, paid_errors, paid_successful = _run_query(paid_calls, language)
            all_rows.extend(paid_rows)
            errors.extend(paid_errors)

    normalized = []
    for provider, row in all_rows:
        item = _normalize_item(row, provider, query, kind)
        if item and search_plan.result_matches_constraints(item, constraints):
            normalized.append(item)
    results = _dedupe(normalized)
    results.sort(key=lambda item: item["match_score"], reverse=True)
    results = results[:max(1, min(12, int(max_results)))]

    fetchable_pages = [item for item in results if item.get("link_type") != "file"]
    if callable(page_fetch) and fetchable_pages:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            future_map = {pool.submit(page_fetch, item["url"]): item for item in fetchable_pages[:3]}
            for future, item in future_map.items():
                try:
                    item["content"] = future.result() or ""
                    item["verified"] = bool(item["content"])
                except Exception:
                    item["content"] = ""
                    item["verified"] = False
        # Re-check language against extracted text when it was not clear from snippets.
        results = [item for item in results if search_plan.result_matches_constraints(item, constraints)]

    if constraints.get("language") == "fa":
        for item in results:
            sample = " ".join((item.get("title") or "", item.get("snippet") or "", item.get("content") or ""))
            letters = [char for char in sample if char.isalpha()]
            persian_ratio = (
                sum(1 for char in letters if "\u0600" <= char <= "\u06ff") / len(letters)
                if letters else 0.0
            )
            item["language_verified"] = bool(len(letters) >= 10 and persian_ratio >= 0.08)

    provider_names = list(dict.fromkeys(item["provider"] for item in results))
    payload = {
        "results": results,
        "engine": "multi" if len(provider_names) > 1 else (provider_names[0] if provider_names else "none"),
        "provider": ", ".join(provider_names) if provider_names else "none",
        "fallback": bool(errors or (provider_names and len(provider_names) == 1 and provider_names[0] in ("DuckDuckGo", "Bing RSS"))),
        "notice": (
            "نتایج فقط از پیوندهای برگشتی موتورهای جست‌وجو ساخته شده‌اند. "
            "دریافت متن صفحه، موجودی یا قیمت زنده را تضمین نمی‌کند."
        ),
        "intent": {"kind": kind, "topic": query},
        "constraints": constraints,
        "errors": errors[:5],
        "cached": False,
    }
    if errors:
        payload["notice"] += " برخی منابع پاسخ ندادند: " + "؛ ".join(errors[:2])
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)), None)
    _CACHE[cache_key] = (time.monotonic(), payload)
    return payload

