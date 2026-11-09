"""Tests for keyless and configurable search provider adapters."""
import json
from unittest.mock import patch

from local_chat import search_providers


class FakeResponse:
    def __init__(self, body):
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        return False
    def read(self, _limit=-1):
        return self.body


def test_bing_rss_parses_provider_returned_urls():
    xml = b"""<?xml version="1.0"?><rss><channel><item>
    <title>Persian result</title><link>https://example.com/article</link>
    <description>Snippet</description></item></channel></rss>"""
    with patch("local_chat.search_providers.urlopen", return_value=FakeResponse(xml)):
        rows = search_providers.search_bing_rss("موضوع فارسی")
    assert rows == [{
        "title": "Persian result",
        "url": "https://example.com/article",
        "snippet": "Snippet",
        "content": "",
    }]


def test_searxng_json_adapter_uses_configured_https_instance():
    payload = {"results": [{
        "title": "Example",
        "url": "https://example.com/page",
        "content": "A snippet",
    }]}
    with patch("local_chat.search_providers.urlopen", return_value=FakeResponse(json.dumps(payload).encode())) as opened:
        rows = search_providers.search_searxng("query", "https://search.example.org", language="fa")
    assert rows[0]["url"] == "https://example.com/page"
    assert "format=json" in opened.call_args.args[0].full_url
    assert "language=fa" in opened.call_args.args[0].full_url
