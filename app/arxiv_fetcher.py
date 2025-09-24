import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from threading import Lock
from typing import Iterable, List

import requests
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ARXIV_API_URL = "http://export.arxiv.org/api/query?"
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_LOCK = Lock()
_TOKENIZER = None
_MODEL = None


def _load_summarizer():
    """LLMのトークナイザーとモデルをGPU優先でロードする。"""
    global _TOKENIZER, _MODEL
    with _MODEL_LOCK:
        if _TOKENIZER is None or _MODEL is None:
            logging.info("LLMモデルをロードしています (device=%s)...", _DEVICE)
            _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
            dtype = torch.float16 if _DEVICE.type == "cuda" else torch.float32
            _MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
            _MODEL.to(_DEVICE)
            _MODEL.eval()
    return _TOKENIZER, _MODEL


def _summarize_abstract(abstract: str) -> str:
    """同期的に要旨を要約し、新規性の説明テキストを生成する。"""
    if not abstract.strip():
        return "要約対象の要旨が空でした。"

    tokenizer, model = _load_summarizer()
    inputs = tokenizer(
        abstract,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(_DEVICE)

    with torch.inference_mode():
        summary_ids = model.generate(
            **inputs,
            max_new_tokens=160,
            num_beams=4,
            early_stopping=True,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.strip()


async def analyze_novelty_with_llm(abstract: str) -> str:
    """GPU (利用可能な場合) で動作するLLMを使って要旨を要約する。"""
    try:
        return await asyncio.to_thread(_summarize_abstract, abstract)
    except Exception as exc:  # pragma: no cover - 例外時のフォールバック
        logging.error("LLMによる要約中にエラーが発生しました: %s", exc)
        return "ローカルLLMでの要約に失敗しました。モデルのダウンロード状況やGPUメモリを確認してください。"


async def fetch_arxiv_papers(search_queries: Iterable[str], max_results_per_query: int = 10) -> List[dict]:
    """
    複数のクエリで論文を検索し、GPU対応LLMで新規性を分析する。

    Args:
        search_queries (Iterable[str]): 検索クエリまたはクエリのリスト。
        max_results_per_query (int): 1クエリあたりの最大取得論文数。

    Returns:
        list: 論文情報の辞書を含むリスト。エラー時は空リストを返す。
    """

    if isinstance(search_queries, str):
        queries = [search_queries]
    else:
        queries = [q for q in search_queries if q]

    unique_papers = {}

    for query in queries:
        logging.info("'%s' に関する論文を検索中...", query)
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(max_results_per_query),
        }
        try:
            response = requests.get(ARXIV_API_URL, params=params, timeout=10)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            namespaces = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            for entry in root.findall("atom:entry", namespaces):
                paper_id = entry.findtext("atom:id", default="", namespaces=namespaces)
                if not paper_id or paper_id in unique_papers:
                    continue

                title = entry.findtext("atom:title", default="", namespaces=namespaces).strip()
                summary = (
                    entry.findtext("atom:summary", default="", namespaces=namespaces)
                    .strip()
                    .replace("\n", " ")
                )
                published_raw = entry.findtext("atom:published", default="", namespaces=namespaces)
                try:
                    published_date = datetime.strptime(published_raw, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
                except ValueError:
                    published_date = published_raw

                authors = [
                    author.findtext("atom:name", default="", namespaces=namespaces)
                    for author in entry.findall("atom:author", namespaces)
                ]
                authors = [author for author in authors if author]

                unique_papers[paper_id] = {
                    "title": title,
                    "authors": authors,
                    "summary": summary,
                    "published_date": published_date,
                    "url": paper_id,
                    "novelty": "",
                }
        except requests.exceptions.RequestException as exc:
            logging.error("APIリクエストエラー (クエリ: %s): %s", query, exc)
        except ET.ParseError as exc:
            logging.error("XML解析エラー (クエリ: %s): %s", query, exc)

    if not unique_papers:
        return []

    logging.info("%d件のユニークな論文の新規性を分析します...", len(unique_papers))
    tasks = [analyze_novelty_with_llm(paper["summary"]) for paper in unique_papers.values()]
    novelty_results = await asyncio.gather(*tasks)

    for paper, novelty_text in zip(unique_papers.values(), novelty_results):
        paper["novelty"] = novelty_text

    sorted_papers = sorted(unique_papers.values(), key=lambda p: p["published_date"], reverse=True)
    return sorted_papers


if __name__ == "__main__":
    test_query = "happy"
    results = asyncio.run(fetch_arxiv_papers(search_queries=test_query, max_results_per_query=3))

    if results:
        for i, paper in enumerate(results):
            print("-" * 50)
            print(f"論文 {i+1}")
            print(f"タイトル: {paper['title']}")
            print(f"URL: {paper['url']}")
            print(f"novelty: {paper['novelty']}")
            print("-" * 50 + "\n")
    else:
        print("テスト実行で論文を取得できませんでした。")
