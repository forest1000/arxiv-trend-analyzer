from __future__ import annotations
import time
from datetime import datetime
from typing import Iterator
import arxiv
from .models import Paper




def search_iter(
    query: str,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    max_results: int = 2000,
    sleep_sec: float = 0.2,
    ) -> Iterator[Paper]:
    # arXivの submittedDate フィルタをクエリに付与
    parts = [f"({query})"]
    if start or end:
        s = (start or datetime(1990, 1, 1)).strftime("%Y%m%d")
        e = (end or datetime(2100, 1, 1)).strftime("%Y%m%d")
        parts.append(f"submittedDate:[{s} TO {e}]")
    q = " AND ".join(parts)


    search = arxiv.Search(
    query=q,
    max_results=max_results,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    )


    for r in search.results(): # type: ignore[attr-defined]
        yield Paper(
            id=r.get_short_id(),
            title=r.title,
            summary=r.summary,
            published=r.published,
            updated=r.updated,
            authors=[a.name for a in r.authors],
            categories=r.categories,
            link_pdf=r.pdf_url if getattr(r, "pdf_url", None) else None,
        )
        time.sleep(sleep_sec)