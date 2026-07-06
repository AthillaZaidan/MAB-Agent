from urllib.error import HTTPError

import modelwatch.connectors as connectors
from modelwatch.connectors import RssConnector


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
