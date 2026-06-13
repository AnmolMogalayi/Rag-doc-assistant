"""Assemble the self-corrective RAG StateGraph.

Topology
--------
    START -> analyze_query -> retrieve -> grade_documents
        --(decide_to_generate)--> generate            [relevant docs found]
                              --> transform_query -> retrieve   [retry within limit]
                              --> web_search -> generate        [retries gone, web enabled]
                              --> generate                      [give up -> grounded "I don't know"]
    generate --(decide_after_generation: Self-RAG)-->
                              --> END                            [grounded & useful, or checks off]
                              --> generate                       [ungrounded, regen budget left]
                              --> transform_query                [not useful, retry budget left]

Both loops are bounded by counters in the state schema, guaranteeing termination.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph import nodes
from app.graph.state import RAGState
from app.logging_config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Conditional edge routers
# --------------------------------------------------------------------------- #
def decide_to_generate(state: RAGState) -> str:
    """Route after document grading (the required conditional edge on grading outcome)."""
    settings = get_settings()
    has_relevant = bool(state.get("documents"))

    if has_relevant:
        return "generate"

    # No relevant docs. If the index was empty, retrying is pointless.
    if state.get("needs_ingestion"):
        return "give_up"

    if state.get("retry_count", 0) < state.get("max_retries", settings.max_retries):
        return "transform_query"

    # Retries exhausted -> try the web (if enabled) before giving up.
    if settings.enable_web_search and not state.get("web_search_used"):
        return "web_search"

    return "give_up"


def decide_after_generation(state: RAGState) -> str:
    """Route after generation using the Self-RAG groundedness + answer-quality grades."""
    settings = get_settings()
    if not settings.enable_self_check:
        return "end"

    grounded = state.get("hallucination_grade", "skipped")
    useful = state.get("answer_grade", "skipped")
    gen_attempts = state.get("generation_attempts", 0)
    max_gen = state.get("max_generation_attempts", settings.max_generation_attempts)

    # Ungrounded answer (hallucination) and we still have a regeneration budget -> retry generate.
    if grounded == "ungrounded" and gen_attempts <= max_gen and state.get("documents"):
        return "regenerate"

    # Grounded but doesn't answer the question, and we can still rewrite/re-retrieve.
    if useful == "not_useful" and state.get("retry_count", 0) < state.get("max_retries",
                                                                          settings.max_retries):
        return "transform_query"

    return "end"


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_graph(checkpointer=None):
    """Build and compile the StateGraph. Pass a checkpointer to enable session memory."""
    workflow = StateGraph(RAGState)

    workflow.add_node("analyze_query", nodes.analyze_query)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("transform_query", nodes.transform_query)
    workflow.add_node("web_search", nodes.web_search_node)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("grade_generation", nodes.grade_generation)

    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Required conditional edge: routes on the document-grading outcome.
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "web_search",
            "give_up": "generate",  # generate() emits a grounded "I don't know"
        },
    )
    workflow.add_edge("transform_query", "retrieve")  # the corrective loop
    workflow.add_edge("web_search", "generate")

    # After generation, run the Self-RAG self-check, then route on it.
    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges(
        "grade_generation",
        decide_after_generation,
        {
            "regenerate": "generate",
            "transform_query": "transform_query",
            "end": END,
        },
    )

    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info("RAG StateGraph compiled (memory=%s)", checkpointer is not None)
    return compiled


@lru_cache
def get_graph():
    """Process-wide compiled graph with an in-memory checkpointer when memory is enabled."""
    settings = get_settings()
    checkpointer = None
    if settings.enable_memory:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
    return build_graph(checkpointer)


def reset_graph_cache() -> None:
    get_graph.cache_clear()
