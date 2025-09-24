"""Utility functions for querying arXiv and analysing novelty on GPU."""
from __future__ import annotations

import asyncio
import logging
import threading
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ARXIV_API_URL = "http://export.arxiv.org/api/query?"
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
ARXIV_NAMESPACE = "http://arxiv.org/schemas/atom"
NAMESPACES = {"atom": ATOM_NAMESPACE, "arxiv": ARXIV_NAMESPACE}


class NoveltyAnalyzer:
    """Summarise an abstract with a Transformer model running on GPU when available.

    The class lazily loads a Hugging Face summarisation pipeline.  Loading the
    model is expensive, so the instance keeps the pipeline cached and protects
    the initialisation with a thread lock.  Inference is executed in a worker
    thread via :func:`asyncio.to_thread` so that multiple abstracts can be
    analysed concurrently from an async context without blocking the event loop.
    """

    def __init__(
        self,
        model_name: str = "sshleifer/distilbart-cnn-12-6",
        *,
        max_summary_tokens: int = 120,
    ) -> None:
        self.model_name = model_name
        self.max_summary_tokens = max_summary_tokens
        self._pipeline = None
        self._pipeline_lock = threading.Lock()

    def _ensure_pipeline(self):
        """Load the Hugging Face pipeline on the GPU the first time it is needed."""
        if self._pipeline is None:
            with self._pipeline_lock:
                if self._pipeline is None:
                    from transformers import pipeline  # Lazy import to keep start-up light.
                    import torch

                    device = 0 if torch.cuda.is_available() else -1
                    logging.info(
                        "Loading summarisation model '%s' on %s.",
                        self.model_name,
                        "GPU" if device >= 0 else "CPU",
                    )
                    self._pipeline = pipeline(
                        task="summarization",
                        model=self.model_name,
                        tokenizer=self.model_name,
                        device=device,
                    )
        return self._pipeline

    def _summarise_sync(self, abstract: str) -> str:
        if not abstract.strip():
            return "要旨が空のため、要約を生成できませんでした。"

        summariser = self._ensure_pipeline()
        result = summariser(
            abstract,
            max_length=self.max_summary_tokens,
            min_length=max(20, self.max_summary_tokens // 2),
            truncation=True,
        )[0]
        return result["summary_text"].strip()

    async def analyse(self, abstract: str) -> str:
        """Analyse the abstract asynchronously.

        The heavy lifting happens in a background thread so that coroutine callers
        can continue to await concurrently.
        """
        try:
            return await asyncio.to_thread(self._summarise_sync, abstract)
        except Exception as exc:  # pragma: no cover - defensive guard against runtime failures
            logging.error("GPU要約モデルの推論に失敗しました: %s", exc)
            return "GPU要約モデルでの分析中にエラーが発生しました。モデルと依存ライブラリを確認してください。"


async def fetch_arxiv_papers(
    search_query: str,
    max_results_per_query: int = 10,
    *,
    novelty_analyzer: Optional[NoveltyAnalyzer] = None,
) -> List[Dict[str, Any]]:
    """Query arXiv for a keyword string and analyse the novelty of each paper.

    Args:
        search_query: arXiv API query string (e.g. ``all:"diffusion" AND cat:cs.CV``).
        max_results_per_query: Maximum number of results to fetch from the API.
        novelty_analyzer: Optional ``NoveltyAnalyzer``.  Passing one makes unit
            testing easier and allows reusing a pre-loaded model.

    Returns:
        A list of dictionaries describing papers sorted by published date.
    """

    if not isinstance(search_query, str) or not search_query.strip():
        raise ValueError("search_query には空でない文字列を指定してください。")

    novelty_analyzer = novelty_analyzer or NoveltyAnalyzer()

    logging.info("'%s' に関する論文を検索中...", search_query)
    params = {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": str(max_results_per_query),
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logging.error("APIリクエストエラー (クエリ: %s): %s", search_query, exc)
        return []

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        logging.error("XML解析エラー (クエリ: %s): %s", search_query, exc)
        return []

    unique_papers: Dict[str, Dict[str, Any]] = {}

    for entry in root.findall("atom:entry", NAMESPACES):
        paper_id = _extract_text(entry, "atom:id")
        if not paper_id or paper_id in unique_papers:
            continue

        summary = _extract_text(entry, "atom:summary").replace("\n", " ")
        published_raw = _extract_text(entry, "atom:published")
        published_date = _normalise_published_date(published_raw)

        authors = [
            _extract_text(author, "atom:name")
            for author in entry.findall("atom:author", NAMESPACES)
            if _extract_text(author, "atom:name")
        ]

        unique_papers[paper_id] = {
            "title": _extract_text(entry, "atom:title"),
            "authors": authors,
            "summary": summary,
            "published_date": published_date,
            "url": paper_id,
            "novelty": "",
        }

    if not unique_papers:
        return []

    logging.info("%d 件のユニークな文献の新規性を分析します...", len(unique_papers))
    novelty_tasks = [novelty_analyzer.analyse(paper["summary"]) for paper in unique_papers.values()]
    novelty_results = await asyncio.gather(*novelty_tasks, return_exceptions=True)

    for paper, novelty_result in zip(unique_papers.values(), novelty_results):
        if isinstance(novelty_result, Exception):  # pragma: no cover - defensive guard
            logging.error("新規性分析のタスクが失敗しました: %s", novelty_result)
            paper["novelty"] = "新規性分析中に予期しないエラーが発生しました。ログを確認してください。"
        else:
            paper["novelty"] = novelty_result

    sorted_papers = sorted(unique_papers.values(), key=lambda p: p["published_date"], reverse=True)
    return sorted_papers


def _extract_text(element: ET.Element, path: str) -> str:
    text = element.findtext(path, default="", namespaces=NAMESPACES)
    return text.strip() if text else ""


def _normalise_published_date(published_raw: str) -> str:
    if not published_raw:
        return ""

    try:
        return datetime.strptime(published_raw, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
    except ValueError:
        return published_raw.split("T")[0]


if __name__ == "__main__":  # pragma: no cover - manual smoke test helper
    test_query = "all:\"happy\""

    async def _run_test():
        return await fetch_arxiv_papers(search_query=test_query, max_results_per_query=3)

    test_results = asyncio.run(_run_test())

    if test_results:
        for i, paper in enumerate(test_results, start=1):
            print("-" * 50)
            print(f"論文 {i}")
            print(f"タイトル: {paper['title']}")
            print(f"URL: {paper['url']}")
            print(f"novelty: {paper['novelty']}")
            print("-" * 50 + "\n")
    else:
        print("テスト実行で論文を取得できませんでした。")
