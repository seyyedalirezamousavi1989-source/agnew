"""Regression tests for the multi-provider search engine."""
import pytest

from local_chat import search_engine


def test_search_intent_and_domain_query_plans():
    product = "قیمت کارت گرافیک RTX 3060"
    video = "ویدیو آموزش پایتون در آپارات پیدا کن"
    assert search_engine.classify(product) == "product"
    assert search_engine.classify(video) == "video"
    assert any("digikala.com/product" in q for q in search_engine._query_plan(product, "product"))
    assert any("torob.com/p" in q for q in search_engine._query_plan(product, "product"))
    assert any("aparat.com/v" in q for q in search_engine._query_plan(video, "video"))


def test_normalizer_only_accepts_real_http_links_and_marks_product_pages():
    assert search_engine._normalize_item({"title": "GPU", "url": "javascript:alert(1)"}, "fake", "GPU", "product") is None
    assert search_engine._normalize_item({"title": "GPU", "url": "https://user:pass@example.com/x"}, "fake", "GPU", "product") is None
    assert search_engine._normalize_item({"title": "local", "url": "http://127.0.0.1/admin"}, "fake", "GPU", "product") is None
    item = search_engine._normalize_item(
        {"title": "کارت گرافیک RTX 3060", "url": "https://www.digikala.com/product/dkp-123/"},
        "DuckDuckGo", "کارت گرافیک RTX 3060", "product")
    assert item["link_type"] == "product"
    assert item["verified"] is False


def test_direct_downloads_are_labeled_as_files():
    item = search_engine._normalize_item(
        {"title": "Research paper", "url": "https://example.com/paper.pdf"},
        "Bing RSS", "research paper", "web",
    )
    assert item["link_type"] == "file"
    assert item["file_type"] == "pdf"


def test_product_search_uses_targeted_queries_and_never_invents_urls(monkeypatch):
    search_engine._CACHE.clear()
    monkeypatch.setattr(search_engine.search_providers, "search_bing_rss", lambda _q: [])
    seen = []
    actual_url = "https://www.digikala.com/product/dkp-123/"
    def ddg(query):
        seen.append(query)
        if "site:digikala.com/product" in query:
            return [{"title": "کارت گرافیک RTX 3060", "url": actual_url, "snippet": "صفحه محصول"}]
        return []
    result = search_engine.search(
        "قیمت کارت گرافیک RTX 3060", {"engine": "auto"}, ddg,
        page_fetch=lambda url: "مشخصات محصول", max_results=8)
    assert len(seen) >= 2
    assert any("site:digikala.com/product" in q for q in seen)
    assert [item["url"] for item in result["results"]] == [actual_url]
    assert result["results"][0]["link_type"] == "product"
    assert result["results"][0]["verified"] is True


def test_video_search_expands_to_aparat_and_preserves_provider_url(monkeypatch):
    search_engine._CACHE.clear()
    monkeypatch.setattr(search_engine.search_providers, "search_bing_rss", lambda _q: [])
    seen = []
    actual_url = "https://www.aparat.com/v/abc123"
    def ddg(query):
        seen.append(query)
        if "site:aparat.com/v" in query:
            return [{"title": "آموزش پایتون", "url": actual_url, "snippet": "ویدیو در آپارات"}]
        return []
    result = search_engine.search("ویدیو آموزش پایتون در آپارات پیدا کن", {"engine": "auto"}, ddg)
    assert any("site:aparat.com/v" in q for q in seen)
    assert result["results"][0]["url"] == actual_url
    assert result["results"][0]["link_type"] == "video"


def test_explicit_site_filter_is_strict_and_dynamic(monkeypatch):
    search_engine._CACHE.clear()
    monkeypatch.setattr(search_engine.search_providers, "search_bing_rss", lambda _q: [])
    seen = []
    def ddg(query):
        seen.append(query)
        return [
            {"title": "Allowed", "url": "https://shop.example.com/product/1", "snippet": "RTX 5070"},
            {"title": "Other", "url": "https://other.example.org/product/2", "snippet": "RTX 5070"},
        ]
    result = search_engine.search(
        "قیمت RTX 5070 فقط سایت example.com",
        {"engine": "auto"}, ddg,
    )
    assert any("site:example.com" in q for q in seen)
    assert [row["url"] for row in result["results"]] == ["https://shop.example.com/product/1"]


def test_auto_prefers_keyless_providers_before_paid_fallback():
    calls = search_engine._provider_calls(
        "RTX 5070", {
            "engine": "auto",
            "serper_api_key": "serper-test-key",
            "brave_api_key": "brave-test-key",
        }, lambda _q: [],
    )
    names = [name for name, _call in calls]
    assert "DuckDuckGo" in names
    assert "Bing RSS" in names
    assert "Google / Serper" not in names
    assert "Brave Search" not in names
    paid = search_engine._paid_provider_calls(
        "RTX 5070", {
            "serper_api_key": "serper-test-key",
            "brave_api_key": "brave-test-key",
        },
    )
    assert {name for name, _call in paid} == {"Google / Serper", "Brave Search"}


def test_product_detection_includes_common_consumer_products():
    assert search_engine.classify("قیمت گوشی آیفون 17") == "product"
    assert search_engine.classify("قیمت آیفون 17") == "product"
    assert search_engine.classify("خرید کفش ورزشی") == "product"
    assert search_engine.classify("خرید بلیط هواپیما") == "web"


def test_empty_search_does_not_call_provider():
    with pytest.raises(search_engine.SearchError):
        search_engine.search("   ", {"engine": "auto"}, lambda _q: pytest.fail("must not search"))
