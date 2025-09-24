import asyncio
import sys
import textwrap
import types

import pytest

requests_stub = types.ModuleType("requests")
requests_stub.exceptions = types.SimpleNamespace(RequestException=Exception)
requests_stub.get = None
sys.modules.setdefault("requests", requests_stub)

from app import arxiv_fetcher


class DummyResponse:
    def __init__(self, payload: str):
        self.content = payload.encode("utf-8")

    def raise_for_status(self) -> None:
        return None


class DummyAnalyzer:
    def __init__(self) -> None:
        self.calls = []

    async def analyse(self, abstract: str) -> str:
        self.calls.append(abstract)
        return f"ANALYSED::{len(self.calls)}"


def test_fetch_arxiv_papers_parses_feed(monkeypatch):
    xml_payload = textwrap.dedent(
        """
        <?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/1234.5678v1</id>
            <title>First Sample Paper</title>
            <summary>First abstract.\nWith a newline.</summary>
            <published>2024-05-01T12:00:00Z</published>
            <author><name>Alice</name></author>
            <author><name>Bob</name></author>
          </entry>
          <entry>
            <id>http://arxiv.org/abs/2345.6789v1</id>
            <title>Second Sample Paper</title>
            <summary>Second abstract body.</summary>
            <published>2024-06-01T08:30:00Z</published>
            <author><name>Carol</name></author>
          </entry>
        </feed>
        """
    ).strip()

    def fake_get(url, params, timeout):
        assert url == arxiv_fetcher.ARXIV_API_URL
        assert params["search_query"] == 'all:"test"'
        assert params["max_results"] == "5"
        return DummyResponse(xml_payload)

    monkeypatch.setattr(arxiv_fetcher.requests, "get", fake_get)

    analyzer = DummyAnalyzer()
    papers = asyncio.run(
        arxiv_fetcher.fetch_arxiv_papers(
            'all:"test"',
            5,
            novelty_analyzer=analyzer,
        )
    )

    assert [paper["title"] for paper in papers] == [
        "Second Sample Paper",
        "First Sample Paper",
    ]
    assert papers[0]["published_date"] == "2024-06-01"
    assert papers[0]["summary"] == "Second abstract body."
    assert analyzer.calls == [
        "First abstract. With a newline.",
        "Second abstract body.",
    ]
    assert papers[0]["novelty"] == "ANALYSED::2"


def test_fetch_arxiv_papers_validates_query():
    with pytest.raises(ValueError):
        asyncio.run(arxiv_fetcher.fetch_arxiv_papers("", 3))
