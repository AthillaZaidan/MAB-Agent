from urllib.error import HTTPError

import modelwatch.connectors as connectors
from modelwatch.connectors import HuggingFaceConnector, OpenRouterConnector, RssConnector


def test_huggingface_connector_keeps_only_configured_model_prefixes(monkeypatch):
    monkeypatch.setattr(
        connectors,
        "get_json",
        lambda *_args, **_kwargs: [
            {"modelId": "Qwen/Qwen3-30B-A3B", "lastModified": "2026-07-06T01:00:00Z"},
            {"modelId": "random-user/cool-lora", "lastModified": "2026-07-06T01:00:00Z"},
        ],
    )

    items = HuggingFaceConnector(max_items=20, model_prefixes=["Qwen/", "deepseek-ai/"])(48)

    assert [item.title for item in items] == ["Qwen/Qwen3-30B-A3B"]


def test_openrouter_connector_keeps_only_configured_model_prefixes(monkeypatch):
    monkeypatch.setattr(
        connectors,
        "get_json",
        lambda *_args, **_kwargs: {
            "data": [
                {"id": "openai/gpt-5-mini", "name": "GPT-5 mini", "created": 1783300000},
                {"id": "somebody/test-model", "name": "test", "created": 1783300000},
            ]
        },
    )

    items = OpenRouterConnector(max_items=20, model_prefixes=["openai/", "anthropic/"])(48)

    assert [item.title for item in items] == ["GPT-5 mini"]


def test_rss_connector_keeps_valid_feeds_when_one_feed_fails(monkeypatch):
    def fake_get_text(url):
        if "bad.test" in url:
            raise HTTPError(url, 404, "Not Found", {}, None)
        return """
        <rss><channel>
          <item>
            <title>New model release</title>
            <link>https://good.test/model</link>
            <pubDate>Mon, 06 Jul 2026 01:00:00 GMT</pubDate>
            <description>Qwen release notes</description>
          </item>
        </channel></rss>
        """

    monkeypatch.setattr(connectors, "get_text", fake_get_text)
    connector = RssConnector(["https://good.test/rss.xml", "https://bad.test/rss.xml"])

    items = connector(48)

    assert [item.title for item in items] == ["New model release"]
    assert connector.failures == {"https://bad.test/rss.xml": "HTTPError: HTTP Error 404: Not Found"}
