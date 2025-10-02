from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
import sqlite3


from .events import bus, FetchCompleted
from .store import get_conn, log_fetch
from .analyzer import weekly_counts
from .viz import save_lineplot

# イベント・ハンドラ

def on_fetch_completed(ev: FetchCompleted) -> None:
    """取得完了後、自動で集計＆可視化を行う。"""
    conn = get_conn(Path("data/papers.db"))

    # ログ（再現性のため）
    log_fetch(conn, ev.query, datetime.utcnow().isoformat(), ev.count)


    # 該当クエリで DB から集計（タイトルか要約に含まれる単純フィルタ）
    df = pd.read_sql_query(
        "SELECT published FROM papers WHERE title LIKE ? OR summary LIKE ?",
        conn,
        params=[f"%{ev.query}%", f"%{ev.query}%"],
    )
    wc = weekly_counts(df, date_col="published")


    out = Path("out"); out.mkdir(parents=True, exist_ok=True)
    wc.to_csv(out / f"trend_{ev.query}.csv")
    save_lineplot(wc, out / f"trend_{ev.query}.png")



def register() -> None:
    bus.subscribe(FetchCompleted, on_fetch_completed)   