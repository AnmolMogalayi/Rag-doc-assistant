"""Graph node functions. Each takes the state and returns a partial-state update dict.

Nodes are deliberately small and single-purpose; all LLM calls go through `graders.py`
(structured + retried). Graders fail *open* so an LLM hiccup degrades quality but never
takes down a request.
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.retries import llm_retry
from app.core.vectorstore import get_vectorstore
from app.graph import graders, prompts
from app.graph.state import RAGState
from app.graph.web_search import web_search as do_web_search
from app.logging_config import get_logger

logger = get_logger(__name__)


def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = [f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-6:]]
    return "\n".join(lines)


def _format_context(docs: list[Document]) -> tuple[str, list[dict]]:
    """Build numbered context block + the citation list referencing the same numbers."""
    blocks, citations = [], []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        blocks.append(f"[{i}] (source: {m.get('source')} — {m.get('section')})\n{d.page_content}")
        citations.append({
            "n": i,
            "doc_id": m.get("doc_id"),
            "title": m.get("title"),
            "source": m.get("source"),
            "section": m.get("section"),
            "source_type": m.get("source_type"),
        })
    return "\n\n".join(blocks), citations


# --------------------------------------------------------------------------- #
# Node 1: Query Analysis
# --------------------------------------------------------------------------- #
def analyze_query(state: RAGState) -> dict:
    question = state["question"]
    history = _format_history(state.get("chat_history", []))
    try:
        analysis = graders.analyze_query(question, history)
        rewritten = analysis.rewritten_query.strip() or question
        queries = [rewritten] + [q for q in analysis.alternative_queries if q.strip()]
        # de-dupe while preserving order
        seen, uniq = set(), []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                uniq.append(q)
        return {
            "query_type": analysis.query_type,
            "rewritten_query": rewritten,
            "search_queries": uniq[:4],
            "steps": [f"analyze_query: type={analysis.query_type}, rewrote -> '{rewritten}'"],
        }
    except Exception as exc:
        logger.exception("Query analysis failed; using raw question")
        return {
            "query_type": "unknown",
            "rewritten_query": question,
            "search_queries": [question],
            "steps": [f"analyze_query: FAILED ({exc}); using raw question"],
        }


# --------------------------------------------------------------------------- #
# Node 2: Retrieval
# --------------------------------------------------------------------------- #
def retrieve(state: RAGState) -> dict:
    vs = get_vectorstore()
    if vs.is_empty():
        logger.warning("Retrieval against empty index")
        return {
            "raw_documents": [],
            "needs_ingestion": True,
            "steps": ["retrieve: index is EMPTY — no documents to search"],
        }

    queries = state.get("search_queries") or [state.get("rewritten_query") or state["question"]]
    settings = get_settings()
    seen: set[str] = set()
    merged: list[Document] = []
    # Multi-query retrieval: union results from each search query, de-duped by chunk_id.
    for q in queries[:3]:
        for d in vs.retrieve(q, k=settings.top_k):
            key = d.metadata.get("chunk_id") or d.page_content[:120]
            if key not in seen:
                seen.add(key)
                merged.append(d)
    merged = merged[: settings.retrieval_fetch_k]
    sources = sorted({d.metadata.get("source", "?") for d in merged})
    return {
        "raw_documents": merged,
        "needs_ingestion": False,
        "steps": [f"retrieve: {len(merged)} candidate chunk(s) from {len(sources)} source(s)"],
    }


# --------------------------------------------------------------------------- #
# Node 3: Document Grading (self-corrective core)
# --------------------------------------------------------------------------- #
def grade_documents(state: RAGState) -> dict:
    question = state["question"]
    raw = state.get("raw_documents", [])
    kept: list[Document] = []
    grades: list[dict] = []
    for d in raw:
        try:
            result = graders.grade_document(
                question, d.page_content,
                source=d.metadata.get("source", ""), section=d.metadata.get("section", ""),
            )
            relevant = result.binary_score == "yes"
            grades.append({
                "chunk_id": d.metadata.get("chunk_id"),
                "score": result.binary_score,
                "reason": result.reason,
            })
            if relevant:
                kept.append(d)
        except Exception:
            # Fail open: keep the doc rather than silently dropping context.
            logger.exception("Grading failed for a chunk; keeping it (fail-open)")
            kept.append(d)
            grades.append({"chunk_id": d.metadata.get("chunk_id"), "score": "error"})
    return {
        "documents": kept,
        "relevance_grades": grades,
        "steps": [f"grade_documents: kept {len(kept)}/{len(raw)} chunk(s) as relevant"],
    }


# --------------------------------------------------------------------------- #
# Corrective node: Transform Query (rewrite + re-retrieve loop)
# --------------------------------------------------------------------------- #
def transform_query(state: RAGState) -> dict:
    attempt = state.get("retry_count", 0) + 1
    previous = state.get("rewritten_query") or state["question"]
    try:
        improved = graders.rewrite_query(state["question"], previous, attempt)
    except Exception:
        logger.exception("Query rewrite failed; reusing original question")
        improved = state["question"]
    return {
        "rewritten_query": improved,
        "search_queries": [improved],
        "retry_count": attempt,
        "steps": [f"transform_query: attempt {attempt} -> '{improved}'"],
    }


# --------------------------------------------------------------------------- #
# Bonus node: Web Search fallback
# --------------------------------------------------------------------------- #
def web_search_node(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["question"]
    docs = do_web_search(query)
    return {
        "documents": docs,
        "web_search_used": True,
        "steps": [f"web_search: {len(docs)} web result(s)"],
    }


# --------------------------------------------------------------------------- #
# Node 4: Generation
# --------------------------------------------------------------------------- #
@llm_retry(max_attempts=3)
def _invoke_generator(system: str, human: str) -> str:
    model = get_chat_model("generator")
    resp = model.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": human},
    ])
    return (getattr(resp, "content", "") or "").strip()


def generate(state: RAGState) -> dict:
    docs = state.get("documents", [])
    question = state["question"]

    if not docs:
        # Grounded "I don't know" — never fabricate.
        try:
            text = _invoke_generator(prompts.GENERATE_IDK,
                                     f"Question: {question}\n\nNo relevant documentation was found.")
        except Exception:
            text = ("I don't have information about this in the available documentation. "
                    "Try rephrasing your question or ingesting more documents.")
        return {
            "generation": text, "citations": [], "confidence": 0.0,
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "steps": ["generate: no relevant context -> grounded 'I don't know'"],
        }

    context, citations = _format_context(docs)
    human = prompts.GENERATE_HUMAN.format(question=question, context=context)
    try:
        text = _invoke_generator(prompts.GENERATE_SYSTEM, human)
    except Exception as exc:
        logger.exception("Generation failed")
        return {
            "generation": "", "citations": citations, "error": str(exc),
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "steps": [f"generate: FAILED ({exc})"],
        }

    # Heuristic confidence: blend of #relevant docs and presence of citations.
    confidence = min(1.0, 0.4 + 0.15 * len(docs)) if "[" in text else 0.5
    return {
        "generation": text,
        "citations": citations,
        "confidence": round(confidence, 2),
        "generation_attempts": state.get("generation_attempts", 0) + 1,
        "steps": [f"generate: produced answer with {len(citations)} citation(s)"],
    }


# --------------------------------------------------------------------------- #
# Bonus node: Self-RAG groundedness + answer-quality check
# --------------------------------------------------------------------------- #
def grade_generation(state: RAGState) -> dict:
    settings = get_settings()
    if not settings.enable_self_check or not state.get("documents") or not state.get("generation"):
        return {"hallucination_grade": "skipped", "answer_grade": "skipped",
                "steps": ["grade_generation: skipped"]}

    context, _ = _format_context(state["documents"])
    grounded = "grounded"
    useful = "useful"
    try:
        g = graders.grade_groundedness(state["generation"], context)
        grounded = "grounded" if g.binary_score == "yes" else "ungrounded"
    except Exception:
        logger.exception("Groundedness check failed (fail-open=grounded)")
    try:
        a = graders.grade_answer(state["question"], state["generation"])
        useful = "useful" if a.binary_score == "yes" else "not_useful"
    except Exception:
        logger.exception("Answer-quality check failed (fail-open=useful)")

    conf = state.get("confidence", 0.5)
    if grounded == "grounded" and useful == "useful":
        conf = max(conf, 0.85)
    elif grounded == "ungrounded":
        conf = min(conf, 0.3)
    return {
        "hallucination_grade": grounded,
        "answer_grade": useful,
        "confidence": round(conf, 2),
        "steps": [f"grade_generation: groundedness={grounded}, answer={useful}"],
    }
