"""Structured LLM graders (relevance, groundedness, answer quality) and the query analyzer.

All use `with_structured_output` so the model is forced to return a typed object; calls are
wrapped in tenacity retries and fail *open* (default to keeping/accepting) so a grader outage
degrades quality but never crashes a query.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.retries import llm_retry
from app.graph import prompts
from app.logging_config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Structured output schemas
# --------------------------------------------------------------------------- #
class GradeDocument(BaseModel):
    """Binary relevance score for a single retrieved chunk."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the chunk is relevant to the question, else 'no'."
    )
    reason: str = Field(default="", description="One short clause explaining the decision.")


class GradeGrounded(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer is fully supported by the context, else 'no'."
    )


class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer resolves the question, else 'no'."
    )


class QueryAnalysis(BaseModel):
    query_type: Literal["conceptual", "how_to", "troubleshooting", "api_reference"] = Field(
        description="The type of the user's question."
    )
    rewritten_query: str = Field(description="A single search-optimized rewrite of the question.")
    alternative_queries: list[str] = Field(
        default_factory=list,
        description="Up to 3 alternative phrasings of the information need.",
    )


# --------------------------------------------------------------------------- #
# Grader functions
# --------------------------------------------------------------------------- #
@llm_retry(max_attempts=3)
def analyze_query(question: str, history: str = "") -> QueryAnalysis:
    model = get_chat_model("rewriter").with_structured_output(QueryAnalysis)
    msgs = [
        {"role": "system", "content": prompts.QUERY_ANALYSIS_SYSTEM},
        {"role": "user", "content": prompts.QUERY_ANALYSIS_HUMAN.format(
            history=history or "(none)", question=question)},
    ]
    return model.invoke(msgs)


@llm_retry(max_attempts=3)
def grade_document(question: str, chunk: str, source: str = "", section: str = "") -> GradeDocument:
    model = get_chat_model("grader").with_structured_output(GradeDocument)
    msgs = [
        {"role": "system", "content": prompts.GRADE_DOCUMENTS_SYSTEM},
        {"role": "user", "content": prompts.GRADE_DOCUMENTS_HUMAN.format(
            question=question, chunk=chunk, source=source, section=section)},
    ]
    return model.invoke(msgs)


@llm_retry(max_attempts=3)
def rewrite_query(question: str, previous_query: str, attempt: int) -> str:
    model = get_chat_model("rewriter")
    msgs = [
        {"role": "system", "content": prompts.REWRITE_SYSTEM},
        {"role": "user", "content": prompts.REWRITE_HUMAN.format(
            question=question, previous_query=previous_query, attempt=attempt)},
    ]
    resp = model.invoke(msgs)
    return (getattr(resp, "content", "") or "").strip() or question


@llm_retry(max_attempts=3)
def grade_groundedness(answer: str, context: str) -> GradeGrounded:
    model = get_chat_model("grader").with_structured_output(GradeGrounded)
    msgs = [
        {"role": "system", "content": prompts.HALLUCINATION_SYSTEM},
        {"role": "user", "content": prompts.HALLUCINATION_HUMAN.format(
            answer=answer, context=context)},
    ]
    return model.invoke(msgs)


@llm_retry(max_attempts=3)
def grade_answer(question: str, answer: str) -> GradeAnswer:
    model = get_chat_model("grader").with_structured_output(GradeAnswer)
    msgs = [
        {"role": "system", "content": prompts.ANSWER_QUALITY_SYSTEM},
        {"role": "user", "content": prompts.ANSWER_QUALITY_HUMAN.format(
            question=question, answer=answer)},
    ]
    return model.invoke(msgs)
