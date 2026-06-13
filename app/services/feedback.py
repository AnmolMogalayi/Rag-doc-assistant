"""Feedback persistence (POST /feedback) backed by SQLite.

Stores thumbs up/down + optional comment, linked to the answer's trace_id and question.
SQLite keeps the project zero-config; swap the DSN for Postgres in production.
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id           TEXT PRIMARY KEY,
    trace_id     TEXT,
    question     TEXT,
    rating       TEXT NOT NULL,          -- 'up' | 'down'
    comment      TEXT,
    orphan       INTEGER DEFAULT 0,      -- 1 if trace_id unknown
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_trace ON feedback(trace_id);
"""


def _connect() -> sqlite3.Connection:
    path = Path(get_settings().feedback_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.executescript(_SCHEMA)
    logger.info("Feedback DB ready at %s", get_settings().feedback_db)


def record_feedback(
    *, rating: str, trace_id: str | None = None, question: str | None = None,
    comment: str | None = None, known_trace: bool = True,
) -> dict:
    fb_id = uuid.uuid4().hex
    created = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO feedback (id, trace_id, question, rating, comment, orphan, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (fb_id, trace_id, question, rating, comment, 0 if known_trace else 1, created),
        )
        conn.commit()
    logger.info("Recorded feedback id=%s rating=%s trace=%s", fb_id, rating, trace_id)
    return {"id": fb_id, "trace_id": trace_id, "rating": rating,
            "orphan": not known_trace, "created_at": created}


def stats() -> dict:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT rating, COUNT(*) c FROM feedback GROUP BY rating"
        ).fetchall()
    counts = {r["rating"]: r["c"] for r in rows}
    return {"up": counts.get("up", 0), "down": counts.get("down", 0),
            "total": sum(counts.values())}
