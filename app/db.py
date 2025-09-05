import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "jouster.db"

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            text TEXT NOT NULL,
            summary TEXT NOT NULL,
            topics TEXT NOT NULL,
            keywords TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

def insert_analysis(rec: dict) -> int:
    conn = _connect()
    cur = conn.execute(
        """
        INSERT INTO analyses (title, text, summary, topics, keywords, sentiment, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (
            rec.get("title"),
            rec["text"],
            rec["summary"],
            json.dumps(rec["topics"]),
            json.dumps(rec["keywords"]),
            rec["sentiment"],
            float(rec["confidence"]),
        ),
    )
    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    return last_id

def search(topic: str | None = None, keyword: str | None = None) -> list[dict]:
    conn = _connect()
    q = "SELECT * FROM analyses"
    params: list = []
    if topic or keyword:
        q += " WHERE 1=1"
        if topic:
            q += " AND lower(topics) LIKE ?"
            params.append(f"%{topic.lower()}%")
        if keyword:
            q += " OR lower(keywords) LIKE ?"
            params.append(f"%{keyword.lower()}%")
    q += " ORDER BY id DESC"
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    for r in rows:
        r["topics"] = json.loads(r["topics"])
        r["keywords"] = json.loads(r["keywords"])
    conn.close()
    return rows
