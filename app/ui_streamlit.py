from __future__ import annotations
import os
import sqlite3
from pathlib import Path
from datetime import datetime, date


import pandas as pd
import streamlit as st


from app.analyzer import weekly_counts
from app.fetcher import search_iter
from app.store import get_conn, upsert_papers, log_fetch


DB_PATH = Path(os.getenv("ARXIV_DB", "data/papers.db"))


st.set_page_config(page_title="arXiv Trend Analyzer", page_icon="📈", layout="wide")


@st.cache_resource
def _conn():
    return get_conn(DB_PATH)


@st.cache_data(show_spinner=False)
def _load_trend(query: str) -> tuple[pd.Series, pd.DataFrame, int]:
    with get_conn(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT id, title, summary, published, authors, link_pdf FROM papers "
            "WHERE title LIKE ? OR summary LIKE ?",
            conn, params=[f"%{query}%", f"%{query}%"]
        )
    if df.empty:
        return pd.Series(dtype="int64"), pd.DataFrame(), 0
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    trend = (
    df.set_index("published").resample("W").size().rename("count").sort_index()
    )
    recent = df.sort_values("published", ascending=False).head(50)
    return trend, recent, len(df)


@st.cache_data(show_spinner=False)
def _last_fetch_info(query: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
    "SELECT fetched_at, count FROM fetch_logs WHERE query=? ORDER BY id DESC LIMIT 1",
    (query,),
    ).fetchone()
    return {"fetched_at": row[0], "count": row[1]} if row else {}


# --- Sidebar ---
st.sidebar.header("Controls")
query = st.sidebar.text_input("Keyword / Query filter", value="medical")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("From", value=date(2024, 1, 1))
with col2:
    end_date = st.date_input("To", value=date.today())
max_results = st.sidebar.number_input("Max results (fetch)", 100, 5000, 500, step=100)


fetch_now = st.sidebar.button("Fetch from arXiv")
refresh = st.sidebar.button("Refresh view")


# --- Main ---
st.title("📈 arXiv Trend Analyzer")
st.caption("Search, fetch, and visualize weekly trends from arXiv")


# 取得実行（UIから）
if fetch_now and query.strip():
    with st.spinner("Fetching from arXiv..."):
        
        rows = []
        d0 = datetime.combine(start_date, datetime.min.time())
        d1 = datetime.combine(end_date, datetime.max.time())
        for p in search_iter(query, start=d0, end=d1, max_results=int(max_results)):
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
        with get_conn(DB_PATH) as conn:
           n = upsert_papers(conn, rows)
           log_fetch(conn, query, datetime.utcnow().isoformat(), n)
        st.success(f"Fetched & upserted: {n} records")
        _load_trend.clear() # キャッシュ更新
        _last_fetch_info.clear()


# 可視化
trend, recent, total = _load_trend(query)
meta = _last_fetch_info(query)


m1, m2, m3 = st.columns(3)
m1.metric("Total matched in DB", f"{total}")
m2.metric("Weeks with data", f"{len(trend)}")
m3.metric("Last fetch count", meta.get("count", 0))


st.subheader("Weekly trend")
if trend.empty:
    st.info("No data for this query yet. Try Fetch from the sidebar.")
else:
    st.line_chart(trend)


st.subheader("Recent papers (top 50)")
if recent.empty:
    st.write("-")
else:
    # 表示用整形
    view = recent[["published", "title", "authors", "link_pdf"]].copy()
    view.rename(columns={"link_pdf": "pdf"}, inplace=True)
    st.dataframe(view, use_container_width=True, hide_index=True)


# ダウンロード（CSV）
if not trend.empty:
    csv = trend.to_csv().encode("utf-8")
    st.download_button("Download trend CSV", csv, file_name=f"trend_{query}.csv", mime="text/csv")


if refresh:
    _load_trend.clear(); _last_fetch_info.clear()
    st.rerun()