from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
id TEXT PRIMARY KEY,
title TEXT,
summary TEXT,
published TEXT,
updated TEXT,
authors TEXT,
categories TEXT,
link_pdf TEXT
);


CREATE TABLE IF NOT EXISTS fetch_logs (
id INTEGER PRIMARY KEY AUTOINCREMENT,
query TEXT NOT NULL,
fetched_at TEXT NOT NULL,
count INTEGER NOT NULL
);
"""




def get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA)
    return conn




def upsert_papers(conn: sqlite3.Connection, rows: Iterable[Tuple]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
    "INSERT OR REPLACE INTO papers(id, title, summary, published, updated, authors, categories, link_pdf)"
    " VALUES (?,?,?,?,?,?,?,?)",
    rows,
    )
    conn.commit()
    return len(rows)




def log_fetch(conn: sqlite3.Connection, query: str, fetched_at: str, count: int) -> None:
    conn.execute(
    "INSERT INTO fetch_logs(query, fetched_at, count) VALUES (?,?,?)",
    (query, fetched_at, count),
    )
    conn.commit()