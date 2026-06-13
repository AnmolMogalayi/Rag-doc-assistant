"""Document registry — durable record of what has been ingested (for GET /documents).

Backed by a small JSON file. The vector store is the source of truth for chunks; the
registry adds doc-level metadata (ingested_at, status, chunk counts) that Chroma does not
track natively. Reads merge registry entries with live chunk counts from the store.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> dict[str, dict]:
    path = Path(get_settings().registry_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Registry file corrupt; starting fresh")
        return {}


def _save(data: dict[str, dict]) -> None:
    path = Path(get_settings().registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def record_ingestion(indexed: list[dict]) -> None:
    """Upsert registry entries for newly indexed documents."""
    if not indexed:
        return
    with _lock:
        data = _load()
        for d in indexed:
            data[d["doc_id"]] = {
                "doc_id": d["doc_id"],
                "title": d.get("title"),
                "source": d.get("source"),
                "source_type": d.get("source_type"),
                "chunks": d.get("chunks", 0),
                "content_hash": d.get("content_hash"),
                "ingested_at": _now(),
                "status": "indexed",
            }
        _save(data)


def list_documents(live_counts: dict[str, int] | None = None) -> list[dict]:
    """Return registry entries, overlaying live per-doc chunk counts when provided."""
    with _lock:
        data = _load()
    out = []
    for entry in data.values():
        e = dict(entry)
        if live_counts is not None:
            e["chunks"] = live_counts.get(e["doc_id"], e.get("chunks", 0))
        out.append(e)
    return sorted(out, key=lambda e: e.get("ingested_at", ""), reverse=True)
