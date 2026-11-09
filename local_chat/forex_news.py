"""Economic-calendar awareness for the forex desk.

The price-based model knows nothing about scheduled news. This module reads a public
weekly calendar feed (JSON list of events: title, country, date, impact, forecast,
previous — the format used by the Forex Factory community feed), keeps the events that
move the pair's currencies and reports:

* the next high-impact events (what, when, forecast/previous);
* ``imminent``: a high-impact release within -15 … +30 minutes → fresh entries are risky;
* ``in_horizon``: high-impact releases inside the model's horizon → the empirical
  probabilities are less reliable, so the desk damps them.

Everything is fail-soft: no network, a changed feed or a rate limit gives ``None`` and the
desk says the calendar is unavailable instead of guessing. Responses are cached.
"""
import json
import os
import threading
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_URLS = (
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
)
USER_AGENT = "LocalChat-ForexDesk/1.0"
CACHE_TTL = 3600          # the weekly feed rarely changes and is rate limited
FAILURE_TTL = 300         # do not hammer a failing feed
LOOK_BEHIND = 15 * 60     # a release this recent is still "imminent"
LOOK_AHEAD = 24 * 3600    # events listed up to a day ahead
IMMINENT_AHEAD = 30 * 60
HORIZON_DAMPING = 0.7
MAX_LISTED = 4
_IMPACT = {"high": "high", "medium": "medium", "low": "low", "holiday": "holiday", "non-economic": "low"}
_COMMODITY_QUOTES = {"XAU": ["USD"], "XAG": ["USD"], "XPT": ["USD"], "XPD": ["USD"]}

_lock = threading.Lock()
_cache = {}               # url -> (stored_at, events or None)


class NewsError(Exception):
    """Calendar could not be read (message is Persian, safe to show)."""


def offline():
    """Tests and offline installs set LOCALCHAT_OFFLINE=1 to skip every network request."""
    return bool(os.environ.get("LOCALCHAT_OFFLINE"))


def currencies_of(symbol):
    """Currency codes whose news moves ``symbol`` (EURUSD → EUR, USD; XAUUSD → USD)."""
    base, quote = symbol[:3].upper(), symbol[3:6].upper()
    if base in _COMMODITY_QUOTES:
        return _COMMODITY_QUOTES[base]
    return [c for c in (base, quote) if len(c) == 3]


def _epoch(value):
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def parse_events(data):
    """Normalised, time-sorted events from the feed's JSON (unknown rows are skipped)."""
    if not isinstance(data, list):
        raise NewsError("قالب تقویم اقتصادی شناخته نشد.")
    events = []
    for row in data:
        if not isinstance(row, dict):
            continue
        when = _epoch(row.get("date"))
        impact = _IMPACT.get(str(row.get("impact", "")).strip().lower())
        currency = str(row.get("country") or row.get("currency") or "").strip().upper()
        title = str(row.get("title") or "").strip()
        if when is None or impact is None or len(currency) != 3 or not title:
            continue
        events.append({
            "time": when, "currency": currency, "title": title[:90], "impact": impact,
            "forecast": str(row.get("forecast") or "")[:20], "previous": str(row.get("previous") or "")[:20],
        })
    events.sort(key=lambda e: e["time"])
    return events


def _download(url, opener):
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with opener(request, timeout=8) as response:
            body = response.read(2_000_000)
    except HTTPError as exc:
        raise NewsError("تقویم اقتصادی در حال حاضر محدودیت درخواست دارد؛ بعداً تلاش می‌شود."
                        if exc.code == 429 else f"سرویس تقویم اقتصادی خطای {exc.code} داد.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise NewsError("ارتباط با سرویس تقویم اقتصادی برقرار نشد.") from exc
    try:
        return parse_events(json.loads(body.decode("utf-8", "replace")))
    except json.JSONDecodeError as exc:
        raise NewsError("پاسخ تقویم اقتصادی JSON معتبر نبود.") from exc


def fetch_events(urls=DEFAULT_URLS, now=None, opener=urlopen):
    """All events of the configured feeds, cached. Raises ``NewsError`` only when no feed works."""
    now = time.time() if now is None else now
    collected, last_error, any_ok = [], None, False
    for url in urls:
        with _lock:
            stored = _cache.get(url)
        if stored and now - stored[0] < (CACHE_TTL if stored[1] is not None else FAILURE_TTL):
            events = stored[1]
        else:
            try:
                events = _download(url, opener)
                with _lock:
                    _cache[url] = (now, events)
            except NewsError as exc:
                last_error = exc
                events = stored[1] if stored else None      # keep serving the stale copy
                with _lock:
                    _cache[url] = (now, events)
        if events is not None:
            any_ok = True
            collected.extend(events)
    if not any_ok:
        raise last_error or NewsError("تقویم اقتصادی در دسترس نیست.")
    return sorted(collected, key=lambda e: e["time"])


def clear_cache():
    with _lock:
        _cache.clear()


def assess(symbol, now_epoch, horizon_seconds, urls=DEFAULT_URLS, opener=urlopen):
    """News situation of ``symbol`` now: dict, or None when the calendar is unavailable/off."""
    if offline():
        return None
    try:
        events = fetch_events(urls, now_epoch, opener)
    except NewsError as exc:
        return {"ok": False, "error": exc.args[0]}
    return evaluate(events, symbol, now_epoch, horizon_seconds)


def evaluate(events, symbol, now_epoch, horizon_seconds):
    """Pure part of ``assess`` (testable without network)."""
    currencies = currencies_of(symbol)
    listed, imminent, in_horizon = [], None, []
    for event in events:
        if event["currency"] not in currencies or event["impact"] not in ("high", "medium"):
            continue
        delta = event["time"] - now_epoch
        if delta < -LOOK_BEHIND or delta > LOOK_AHEAD:
            continue
        item = {**event, "utc": datetime.fromtimestamp(event["time"], timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "minutes": round(delta / 60)}
        if event["impact"] == "medium" and delta > 3 * 3600:
            continue            # medium releases matter only close to now
        listed.append(item)
        if event["impact"] == "high":
            if -LOOK_BEHIND <= delta <= IMMINENT_AHEAD and imminent is None:
                imminent = item
            if 0 <= delta <= horizon_seconds:
                in_horizon.append(item)
    listed.sort(key=lambda e: (e["impact"] != "high", e["time"]))
    listed = sorted(listed[:MAX_LISTED], key=lambda e: e["time"])
    return {
        "ok": True, "currencies": currencies, "events": listed, "imminent": imminent,
        "in_horizon": in_horizon, "damping": HORIZON_DAMPING if (in_horizon or imminent) else 1.0,
    }


def describe(event):
    """One Persian line about an event."""
    minutes = event["minutes"]
    if minutes >= 0:
        when = f"{minutes // 60} ساعت و {minutes % 60} دقیقه دیگر" if minutes >= 60 else f"{minutes} دقیقه دیگر"
    else:
        when = f"{-minutes} دقیقه پیش"
    extra = ""
    if event["forecast"] or event["previous"]:
        extra = f" | پیش‌بینی {event['forecast'] or '-'} قبلی {event['previous'] or '-'}"
    level = "اهمیت بالا" if event["impact"] == "high" else "اهمیت متوسط"
    return f"{event['currency']} {event['title']} ({level}) {event['utc'][11:]} UTC — {when}{extra}"
