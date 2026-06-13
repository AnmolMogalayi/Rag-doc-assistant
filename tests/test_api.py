"""API contract tests via FastAPI TestClient (LLM + vector store faked)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core import vectorstore
from app.main import create_app
from app.services import rag_service
from tests.conftest import FakeVectorStore, make_doc


@pytest.fixture
def client(monkeypatch):
    # Inject a populated fake store so startup auto-ingest is skipped.
    fake = FakeVectorStore([make_doc("LangGraph routing via add_conditional_edges",
                                     doc_id="lg", source="langgraph.md", section="Edges")])
    vectorstore.set_vectorstore(fake)

    def fake_answer(question, **kwargs):
        return {
            "trace_id": "test-trace-1", "question": question,
            "answer": "Use add_conditional_edges [1].",
            "sources": [{"title": "lg", "source": "langgraph.md",
                         "section": "Edges", "source_type": "file"}],
            "citations": [{"n": 1, "doc_id": "lg", "title": "lg", "source": "langgraph.md",
                           "section": "Edges", "source_type": "file"}],
            "query_type": "how_to", "confidence": 0.9, "retries": 0,
            "web_search_used": False, "grounded": "grounded", "answer_quality": "useful",
            "needs_ingestion": False, "latency_ms": 12, "steps": ["analyze_query", "generate"],
            "error": None,
        }

    monkeypatch.setattr(rag_service, "answer_question", fake_answer)
    monkeypatch.setattr(rag_service, "is_known_trace", lambda t: t == "test-trace-1")

    app = create_app()
    with TestClient(app) as c:
        yield c
    vectorstore.set_vectorstore(None)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "llm_provider" in body


def test_query_returns_answer_and_sources(client):
    r = client.post("/query", json={"question": "How do conditional edges work?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Use add_conditional_edges")
    assert body["sources"][0]["source"] == "langgraph.md"
    assert body["citations"][0]["n"] == 1
    assert body["trace_id"] == "test-trace-1"
    assert r.headers.get("x-trace-id")


def test_query_validation_rejects_blank(client):
    r = client.post("/query", json={"question": "  "})
    assert r.status_code == 422


def test_query_validation_rejects_too_short(client):
    r = client.post("/query", json={"question": "a"})
    assert r.status_code == 422


def test_documents_listing(client):
    r = client.get("/documents")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert body["total_chunks"] >= 1


def test_feedback_known_trace(client):
    r = client.post("/feedback", json={"trace_id": "test-trace-1", "rating": "up",
                                       "comment": "great"})
    assert r.status_code == 201
    body = r.json()
    assert body["rating"] == "up"
    assert body["orphan"] is False


def test_feedback_orphan_trace(client):
    r = client.post("/feedback", json={"trace_id": "unknown", "rating": "down"})
    assert r.status_code == 201
    assert r.json()["orphan"] is True


def test_feedback_invalid_rating(client):
    r = client.post("/feedback", json={"trace_id": "x", "rating": "maybe"})
    assert r.status_code == 422


def test_ingest_requires_input(client):
    r = client.post("/ingest")
    assert r.status_code == 400
