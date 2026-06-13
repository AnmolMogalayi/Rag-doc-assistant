"""Shared test fixtures.

Tests run fully offline: no LLM keys, no network, no model downloads. We point all
filesystem paths at a throwaway temp dir and inject fake LLM/vector-store behavior.
"""
from __future__ import annotations

import os
import tempfile

import pytest

# Configure environment BEFORE importing the app so cached settings pick it up.
_TMP = tempfile.mkdtemp(prefix="rag-test-")
os.environ.update({
    "ENVIRONMENT": "test",
    "LOG_JSON": "false",
    "CHROMA_DIR": os.path.join(_TMP, "chroma"),
    "FEEDBACK_DB": os.path.join(_TMP, "feedback.db"),
    "REGISTRY_PATH": os.path.join(_TMP, "registry.json"),
    "CORPUS_DIR": os.path.join(_TMP, "corpus"),
    "USE_RERANKER": "false",
    "USE_HYBRID_SEARCH": "false",
    "ENABLE_WEB_SEARCH": "false",
    "ENABLE_SELF_CHECK": "true",
    "MAX_RETRIES": "2",
})

from langchain_core.documents import Document  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()
get_settings().ensure_dirs()


class FakeVectorStore:
    """In-memory stand-in for VectorStoreManager (no embeddings / Chroma)."""

    def __init__(self, docs: list[Document] | None = None):
        self._docs = docs or []

    def is_empty(self) -> bool:
        return not self._docs

    def count(self) -> int:
        return len(self._docs)

    def retrieve(self, query: str, k: int | None = None):
        return list(self._docs)[: (k or 5)]

    def list_documents(self):
        agg: dict[str, dict] = {}
        for d in self._docs:
            did = d.metadata.get("doc_id", "doc")
            e = agg.setdefault(did, {"doc_id": did, "title": d.metadata.get("title"),
                                     "source": d.metadata.get("source"),
                                     "source_type": d.metadata.get("source_type"), "chunks": 0})
            e["chunks"] += 1
        return list(agg.values())

    def index_documents(self, loaded):
        return {"indexed": [], "skipped": [], "chunks": 0}


def make_doc(text: str, doc_id="doc1", source="doc1.md", section="Intro", n=0) -> Document:
    return Document(page_content=text, metadata={
        "doc_id": doc_id, "chunk_id": f"{doc_id}::{n}", "title": doc_id,
        "source": source, "section": section, "source_type": "file",
    })


@pytest.fixture
def fake_docs() -> list[Document]:
    return [
        make_doc("LangGraph uses add_conditional_edges to route on state.", "lg", "langgraph.md",
                 "Conditional Edges", 0),
        make_doc("FastAPI returns 422 on validation errors automatically.", "fa", "fastapi.md",
                 "Validation", 1),
    ]
