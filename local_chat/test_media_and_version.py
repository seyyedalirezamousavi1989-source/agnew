"""Tests for 2.0.5: version consistency, smart search intent, media parsers, search order, voice endpoint."""
import json
import re
import sys
import threading
import types
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from local_chat import media_search, search_intent, search_engine, version  # noqa: E402


# ------------------------------------------------------------------ version
def test_version_is_consistent_everywhere():
    v = version.APP_VERSION
    assert re.search(rf'const UI_VERSION = "{re.escape(v)}"', (ROOT / "static" / "extras.js").read_text(encoding="utf-8"))
    assert f'#define AppVersion "{v}"' in (PROJECT / "packaging" / "installer.iss").read_text(encoding="utf-8")
    assert f"LocalChat-Setup-{v}.exe" in (PROJECT / "build_installer.bat").read_text(encoding="utf-8")
    assert "APP_VERSION =" not in (ROOT / "launcher.py").read_text(encoding="utf-8")   # imported, not duplicated


def test_new_modules_are_in_both_specs():
    for name in ("localchat.spec", "localchat_onefile.spec"):
        text = (PROJECT / "packaging" / name).read_text(encoding="utf-8")
        for module in ("local_chat.version", "local_chat.media_search", "local_chat.search_intent", "local_chat.search_engine", "edge_tts"):
            assert module in text, (name, module)


# ------------------------------------------------------------------ intent
@pytest.mark.parametrize("text,kind,topic", [
    ("یه فیلم ترسناک پیدا کن برام", "movie", "ترسناک"),
    ("لینک آهنگ شادمهر عقیلی رو بده", "music", "شادمهر عقیلی"),
    ("یه عکس از گربه میخوام", "image", "گربه"),
    ("کلیپ آموزش پایتون رو پیدا کن", "video", "آموزش پایتون"),
    ("find me a trailer for Dune 2", "video", "Dune 2"),
    ("download the song Bohemian Rhapsody", "music", "Bohemian Rhapsody"),
    ("عکس‌های برج میلاد را سرچ کن", "image", "برج میلاد"),
    ("where can I watch Inception", "movie", "Inception"),
])
def test_media_intent(text, kind, topic):
    result = search_intent.classify(text)
    assert result["kind"] == kind
    assert result["topic"] == topic


@pytest.mark.parametrize("text", [
    "درباره فیلم تایتانیک توضیح بده",      # a question, not a request for links
    "قیمت دلار امروز چنده",
    "پایتون چیه؟",
    "یک تابع بنویس که لیست را مرتب کند",
])
def test_non_media_stays_web(text):
    assert search_intent.classify(text)["kind"] in ("web", "news")


# ------------------------------------------------------------------ parsers (canned provider payloads)
YT_PAGE = "<script>var ytInitialData = " + json.dumps({"contents": [{"videoRenderer": {
    "videoId": "dQw4w9WgXcQ", "title": {"runs": [{"text": "Never Gonna Give You Up"}]},
    "ownerText": {"runs": [{"text": "Rick Astley"}]}, "lengthText": {"simpleText": "3:33"},
    "viewCountText": {"simpleText": "1B views"}}}, {"videoRenderer": {"videoId": "bad id"}}]}) + ";</script>"


def test_parse_youtube():
    items = media_search.parse_youtube(YT_PAGE, "music")
    assert len(items) == 1
    assert items[0]["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert items[0]["kind"] == "music" and "Rick Astley" in items[0]["snippet"]


def test_parse_itunes():
    data = {"results": [{"trackName": "Song", "artistName": "Artist", "collectionName": "Album", "releaseDate": "2020-01-01T00:00:00Z",
                         "trackViewUrl": "https://music.apple.com/x", "previewUrl": "https://audio.apple.com/p.m4a"},
                        {"trackName": "No link"}]}
    items = media_search.parse_itunes(data, "music")
    assert len(items) == 1 and items[0]["preview_url"].endswith(".m4a")


def test_parse_openverse_and_commons_and_wikipedia():
    ov = media_search.parse_openverse({"results": [{"title": "Cat", "url": "https://img/x.jpg",
                                                    "foreign_landing_url": "https://site/cat", "license": "by", "license_version": "2.0"}]})
    assert ov[0]["url"] == "https://site/cat" and ov[0]["image_url"] == "https://img/x.jpg"
    cm = media_search.parse_commons({"query": {"pages": {"1": {"title": "File:Tower.jpg", "index": 1, "imageinfo": [
        {"url": "https://upload.wikimedia.org/t.jpg", "descriptionurl": "https://commons.wikimedia.org/wiki/File:Tower.jpg"}]}}}})
    assert cm[0]["title"] == "Tower"
    wp = media_search.parse_opensearch(["q", ["Dune"], ["film"], ["https://en.wikipedia.org/wiki/Dune"]], "Wikipedia (en)", "movie")
    assert wp[0]["url"].startswith("https://en.wikipedia.org")


def test_items_reject_unsafe_urls():
    assert media_search._item("x", "javascript:alert(1)", "", "video", "s") is None
    assert media_search._item("x", "https://user:pw@host/x", "", "video", "s") is None


def test_search_media_merges_dedupes_and_falls_back(monkeypatch):
    media_search._CACHE.clear()
    monkeypatch.setattr(media_search, "youtube", lambda q, kind="video", limit=5: [media_search._item("A", "https://y/1", "", kind, "YouTube")])
    monkeypatch.setattr(media_search, "itunes", lambda t, kind, limit=4: [media_search._item("A2", "https://y/1", "", kind, "iTunes")])
    items = media_search.search_media("music", "topic", "en")
    assert [i["url"] for i in items] == ["https://y/1"]          # same URL from two providers appears once
    media_search._CACHE.clear()
    monkeypatch.setattr(media_search, "youtube", lambda *a, **k: [])
    monkeypatch.setattr(media_search, "itunes", lambda *a, **k: [])
    fallback = media_search.search_media("music", "other", "en", web_fallback=lambda q: [{"title": "W", "url": "https://w/1", "snippet": "s"}])
    assert fallback and fallback[0]["kind"] == "music"


# ------------------------------------------------------------------ server integration
@pytest.fixture()
def server(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from local_chat import server as srv
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.ChatHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield srv, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url, payload):
    request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=20) as response:
            return response.status, dict(response.headers), response.read()
    except Exception as exc:                                    # HTTPError carries the body
        if hasattr(exc, "code"):
            return exc.code, dict(exc.headers), exc.read()
        raise


def test_status_reports_version_and_build(server):
    srv, base = server
    with urlopen(base + "/api/status", timeout=10) as response:
        data = json.loads(response.read())
    assert data["version"] == version.APP_VERSION and data["build"] and data["pid"] > 0 and data["changelog"]


def test_forex_capture_script_is_served(server):
    _srv, base = server
    with urlopen(base + "/static/forex_capture.js", timeout=10) as response:
        assert response.status == 200 and b"function" in response.read()


def test_search_api_returns_media_with_intent(server, monkeypatch):
    srv, base = server
    media_search._CACHE.clear()
    monkeypatch.setattr(media_search, "youtube", lambda q, kind="video", limit=5: [media_search._item("Clip", "https://www.youtube.com/watch?v=abcdefghijk", "YouTube", kind, "YouTube")])
    monkeypatch.setattr(srv, "_ddg_results", lambda _query: [])
    status, _h, body = _post(base + "/api/search", {"query": "کلیپ آموزش پایتون رو پیدا کن"})
    data = json.loads(body)
    assert status == 200 and data["intent"]["kind"] == "video"
    assert data["results"][0]["kind"] == "video" and data["engine"] == "media"


def test_voice_edge_tts_returns_mp3_with_matching_mime(server, monkeypatch):
    srv, base = server
    fake = types.ModuleType("edge_tts")

    class Communicate:
        def __init__(self, text, voice, rate="+0%"):
            pass

        async def save(self, path):
            Path(path).write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" * 64)

    fake.Communicate = Communicate
    monkeypatch.setitem(sys.modules, "edge_tts", fake)
    status, headers, body = _post(base + "/api/voice/synthesize", {"text": "سلام، این یک آزمایش است."})
    assert status == 200 and headers["Content-Type"] == "audio/mpeg" and body[:3] == b"ID3"


def test_voice_failure_is_json_not_a_dropped_connection(server, monkeypatch):
    srv, base = server
    monkeypatch.setattr(srv, "normalize_for_speech", lambda text: (_ for _ in ()).throw(ValueError("boom")))
    status, _h, body = _post(base + "/api/voice/synthesize", {"text": "سلام"})
    assert status == 502 and "boom" in json.loads(body)["error"]


def test_voice_diagnose_lists_checks(server):
    _srv, base = server
    with urlopen(base + "/api/voice/diagnose", timeout=10) as response:
        data = json.loads(response.read())
    assert {c["name"] for c in data["checks"]} >= {"کتابخانهٔ edge-tts", "دسترسی DNS به سرویس Edge TTS"}


# ------------------------------------------------------------------ UI contract (static checks, JS is also node --check'ed)
def test_js_sources_go_below_the_answer():
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "renderSources(userArticle" not in js                      # old behaviour: links above the answer
    assert 'if (role !== "user") renderSources(article, sources);' in js
    assert "renderSources(article, userMessage.sources)" in js
    assert 'sniffAudioKind' in js and 'String.fromCharCode(...bytes.slice(0,4))==="RIFF"' not in js
