"""RAG orchestration: run the compiled graph for a question and shape the API response.

Owns trace-id assignment, recursion limits, session/memory config, latency timing, and a
small bounded record of recent traces so /feedback can mark whether a trace_id is known.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock

from app.config import get_settings
from app.graph.state import initial_state
from app.graph.workflow import get_graph
from app.logging_config import get_logger, get_trace_id, set_trace_id

logger = get_logger(__name__)

# Bounded LRU of recent trace_ids (for feedback validation), no external store needed.
_RECENT_MAX = 1000
_recent_traces: "OrderedDict[str, str]" = OrderedDict()
_recent_lock = Lock()


def _remember_trace(trace_id: str, question: str) -> None:
    with _recent_lock:
        _recent_traces[trace_id] = question
        while len(_recent_traces) > _RECENT_MAX:
            _recent_traces.popitem(last=False)


def is_known_trace(trace_id: str) -> bool:
    with _recent_lock:
        return trace_id in _recent_traces


def answer_question(
    question: str,
    *,
    session_id: str = "default",
    chat_history: list[dict] | None = None,
    top_k: int | None = None,
    max_retries: int | None = None,
    trace_id: str | None = None,
) -> dict:
    """Run the graph end-to-end and return a structured, API-ready result."""
    settings = get_settings()
    trace_id = set_trace_id(trace_id)
    started = time.perf_counter()

    state = initial_state(
        question,
        session_id=session_id,
        chat_history=chat_history,
        max_retries=max_retries if max_retries is not None else settings.max_retries,
        max_generation_attempts=settings.max_generation_attempts,
    )

    graph = get_graph()
    config = {
        "recursion_limit": settings.recursion_limit,
        "configurable": {"thread_id": session_id},
    }

    logger.info("Running RAG graph", extra={"ctx_question": question, "ctx_session": session_id})
    try:
        final = graph.invoke(state, config=config)
    except Exception as exc:
        logger.exception("Graph execution failed")
        elapsed = round((time.perf_counter() - started) * 1000)
        return {
            "trace_id": trace_id, "question": question,
            "answer": "An internal error occurred while processing your question.",
            "sources": [], "citations": [], "query_type": "unknown",
            "confidence": 0.0, "retries": 0, "web_search_used": False,
            "grounded": "error", "answer_quality": "error",
            "latency_ms": elapsed, "steps": [], "error": str(exc),
        }

    elapsed = round((time.perf_counter() - started) * 1000)
    _remember_trace(trace_id, question)

    citations = final.get("citations", [])
    # De-duplicate sources for a clean "sources" list while keeping per-citation detail.
    seen, sources = set(), []
    for c in citations:
        key = c.get("source")
        if key and key not in seen:
            seen.add(key)
            sources.append({"title": c.get("title"), "source": key,
                            "section": c.get("section"), "source_type": c.get("source_type")})

    result = {
        "trace_id": trace_id,
        "question": question,
        "answer": final.get("generation", ""),
        "sources": sources,
        "citations": citations,
        "query_type": final.get("query_type", "unknown"),
        "confidence": final.get("confidence", 0.0),
        "retries": final.get("retry_count", 0),
        "web_search_used": final.get("web_search_used", False),
        "grounded": final.get("hallucination_grade", "skipped"),
        "answer_quality": final.get("answer_grade", "skipped"),
        "needs_ingestion": final.get("needs_ingestion", False),
        "latency_ms": elapsed,
        "steps": final.get("steps", []),
        "error": final.get("error"),
    }
    logger.info("RAG complete in %dms (retries=%d, sources=%d, grounded=%s)",
                elapsed, result["retries"], len(sources), result["grounded"])
    return result
