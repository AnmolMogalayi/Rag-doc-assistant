"""Node-level tests with the LLM monkeypatched (offline, deterministic)."""
from __future__ import annotations

from types import SimpleNamespace

from app.graph import graders, nodes
from app.graph.state import initial_state
from tests.conftest import FakeVectorStore, make_doc


def _state(**over):
    s = initial_state("How do conditional edges work?")
    s.update(over)
    return s


def test_grade_documents_filters_irrelevant(monkeypatch):
    docs = [make_doc("relevant about edges", n=0), make_doc("totally unrelated", n=1)]

    def fake_grade(question, chunk, source="", section=""):
        score = "yes" if "edges" in chunk else "no"
        return SimpleNamespace(binary_score=score, reason="t")

    monkeypatch.setattr(graders, "grade_document", fake_grade)
    out = nodes.grade_documents(_state(raw_documents=docs))
    assert len(out["documents"]) == 1
    assert "edges" in out["documents"][0].page_content
    assert len(out["relevance_grades"]) == 2


def test_grade_documents_fails_open(monkeypatch):
    # If the grader raises, the chunk should be kept rather than silently dropped.
    def boom(*a, **k):
        raise RuntimeError("grader down")

    monkeypatch.setattr(graders, "grade_document", boom)
    out = nodes.grade_documents(_state(raw_documents=[make_doc("x")]))
    assert len(out["documents"]) == 1


def test_generate_builds_numbered_citations(monkeypatch):
    monkeypatch.setattr(nodes, "_invoke_generator", lambda s, h: "Use add_conditional_edges [1].")
    docs = [make_doc("edges doc", doc_id="lg", source="langgraph.md", section="Edges")]
    out = nodes.generate(_state(documents=docs))
    assert out["citations"][0]["n"] == 1
    assert out["citations"][0]["source"] == "langgraph.md"
    assert "[1]" in out["generation"]
    assert out["generation_attempts"] == 1


def test_generate_without_docs_is_grounded_idk(monkeypatch):
    monkeypatch.setattr(nodes, "_invoke_generator",
                        lambda s, h: "I don't have that in the docs.")
    out = nodes.generate(_state(documents=[]))
    assert out["citations"] == []
    assert out["confidence"] == 0.0


def test_transform_query_increments_retry(monkeypatch):
    monkeypatch.setattr(graders, "rewrite_query", lambda q, p, a: "improved query")
    out = nodes.transform_query(_state(retry_count=0))
    assert out["retry_count"] == 1
    assert out["rewritten_query"] == "improved query"


def test_retrieve_on_empty_index_flags_ingestion(monkeypatch):
    monkeypatch.setattr(nodes, "get_vectorstore", lambda: FakeVectorStore([]))
    out = nodes.retrieve(_state(search_queries=["q"]))
    assert out["needs_ingestion"] is True
    assert out["raw_documents"] == []


def test_retrieve_returns_candidates(monkeypatch):
    vs = FakeVectorStore([make_doc("a", n=0), make_doc("b", n=1)])
    monkeypatch.setattr(nodes, "get_vectorstore", lambda: vs)
    out = nodes.retrieve(_state(search_queries=["q"]))
    assert out["needs_ingestion"] is False
    assert len(out["raw_documents"]) == 2
