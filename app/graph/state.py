"""The graph state schema — the core evaluation criterion called out in the PDF.

Design notes
------------
* We use a TypedDict (not MessagesState) because this is a *structured* RAG pipeline,
  not a chat-tool agent: we want explicit, inspectable fields for documents, grades,
  retries and the per-node trace, which makes routing logic and debugging clear.
* Retry tracking is first-class: `retry_count`/`max_retries` bound the rewrite→retrieve
  loop, and `generation_attempts`/`max_generation_attempts` bound regeneration after a
  failed groundedness check. Both guarantee termination (no infinite loops).
* `steps` accumulates a human-readable trace returned in the API response and logs, so
  every decision the graph made is observable.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.documents import Document

QueryType = Literal["conceptual", "how_to", "troubleshooting", "api_reference", "unknown"]
RouteDecision = Literal["generate", "transform_query", "web_search", "give_up"]


def _append(left: list, right: list) -> list:
    """Reducer: concatenate trace lists across node updates."""
    return (left or []) + (right or [])


class RAGState(TypedDict, total=False):
    # ---- Inputs ----
    question: str                      # original user question (immutable reference)
    session_id: str                    # for conversation memory (bonus)
    chat_history: list[dict]           # prior turns [{role, content}]

    # ---- Query analysis (Node 1) ----
    query_type: QueryType
    rewritten_query: str               # current query used for retrieval
    search_queries: list[str]          # multi-query expansion

    # ---- Retrieval (Node 2) ----
    raw_documents: list[Document]      # everything retrieved this round
    documents: list[Document]          # relevant docs kept after grading

    # ---- Grading (Node 3) ----
    relevance_grades: list[dict]       # [{chunk_id, score, reason}]

    # ---- Control / routing ----
    retry_count: int                   # rewrite→retrieve loop counter
    max_retries: int
    generation_attempts: int           # regeneration counter (Self-RAG)
    max_generation_attempts: int
    route: RouteDecision
    web_search_used: bool
    needs_ingestion: bool              # true when the index is empty

    # ---- Generation (Node 4) ----
    generation: str
    citations: list[dict]              # [{n, doc_id, title, source, section}]

    # ---- Self-check (bonus: Self-RAG) ----
    hallucination_grade: str           # "grounded" | "ungrounded" | "skipped"
    answer_grade: str                  # "useful" | "not_useful" | "skipped"
    confidence: float                  # 0..1 heuristic

    # ---- Meta ----
    steps: Annotated[list[str], _append]  # accumulated trace across nodes
    error: Optional[str]
    extra: dict[str, Any]


def initial_state(
    question: str,
    *,
    session_id: str = "default",
    chat_history: Optional[list[dict]] = None,
    max_retries: int = 2,
    max_generation_attempts: int = 1,
) -> RAGState:
    return RAGState(
        question=question.strip(),
        session_id=session_id,
        chat_history=chat_history or [],
        query_type="unknown",
        rewritten_query=question.strip(),
        search_queries=[],
        raw_documents=[],
        documents=[],
        relevance_grades=[],
        retry_count=0,
        max_retries=max_retries,
        generation_attempts=0,
        max_generation_attempts=max_generation_attempts,
        route="generate",
        web_search_used=False,
        needs_ingestion=False,
        generation="",
        citations=[],
        hallucination_grade="skipped",
        answer_grade="skipped",
        confidence=0.0,
        steps=[],
        error=None,
        extra={},
    )
