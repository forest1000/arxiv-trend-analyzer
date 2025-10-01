from __future__ import annotations
import typer
from pathlib import Path
from datetime import datetime
import sqlite3
import pandas as pd


from .fetcher import search_iter
from .store import get_conn, upsert_papers
from .events import bus, FetchCompleted
from .handlers import register as register_handlers
from .analyzer import weekly_counts
from .viz import save_lineplot


app = typer.Typer(help="arXiv trend analyzer")
register_handlers() # イベント購読を有効化

@app.command()
def fetch(
    query: str = typer.Argument(..., help="arXiv query expression"),
    db: Path = typer.Option(Path("data/papers.db"), help="SQLite path"),
    date_from: str = typer.Option("1990-01-01", help="YYYY-MM-DD"),
    date_to: str = typer.Option("2100-01-01", help="YYYY-MM-DD"),
    max_results: int = typer.Option(1000, help="arXiv max_results"),
    ):
    d0, d1 = datetime.fromisoformat(date_from), datetime.fromisoformat(date_to)
    conn = get_conn(db)


    rows = []
    for p in search_iter(query, start=d0, end=d1, max_results=max_results):
        rows.append(
            (
                p.id,
                p.title,
                p.summary,
                p.published.isoformat(),
                p.updated.isoformat() if p.updated else None,
                "|".join(p.authors),
                "|".join(p.categories),
                str(p.link_pdf) if p.link_pdf else None,
            )
        )


    n = upsert_papers(conn, rows)


    # 取得完了イベントを発火（ハンドラが trend/plot を自動実行）
    bus.publish(FetchCompleted(query=query, count=n))


    typer.echo(f"fetched & upserted: {n}")




@app.command()
def trend(
        query: str = typer.Argument(..., help="keyword to filter title/summary"),
        db: Path = typer.Option(Path("data/papers.db"), help="SQLite path"),
        out: Path = typer.Option(Path("out"), help="output dir"),
        ):
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT published FROM papers WHERE title LIKE ? OR summary LIKE ?",
        conn,
        params=[f"%{query}%", f"%{query}%"],
    )
    wc = weekly_counts(df, date_col="published")
    out.mkdir(parents=True, exist_ok=True)
    wc.to_csv(out / f"trend_{query}.csv")
    save_lineplot(wc, out / f"trend_{query}.png")
    app()