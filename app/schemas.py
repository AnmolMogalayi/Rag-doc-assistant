"""Pydantic v2 request/response models — the API contract with validation."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# /query
# --------------------------------------------------------------------------- #
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, description="Natural-language question.")
    session_id: str = Field(default="default", max_length=128)
    chat_history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    max_retries: Optional[int] = Field(default=None, ge=0, le=5)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class SourceRef(BaseModel):
    title: Optional[str] = None
    source: Optional[str] = None
    section: Optional[str] = None
    source_type: Optional[str] = None


class Citation(SourceRef):
    n: int
    doc_id: Optional[str] = None


class QueryResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    sources: list[SourceRef]
    citations: list[Citation]
    query_type: str
    confidence: float
    retries: int
    web_search_used: bool
    grounded: str
    answer_quality: str
    needs_ingestion: bool = False
    latency_ms: int
    steps: list[str]
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# /ingest
# --------------------------------------------------------------------------- #
class IngestUrlsRequest(BaseModel):
    urls: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("urls")
    @classmethod
    def _valid_urls(cls, v: list[str]) -> list[str]:
        cleaned = []
        for u in v:
            u = u.strip()
            if not (u.startswith("http://") or u.startswith("https://")):
                raise ValueError(f"Invalid URL (must start with http/https): {u}")
            cleaned.append(u)
        return cleaned


class IngestedDoc(BaseModel):
    doc_id: str
    title: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = None
    chunks: int = 0


class IngestResponse(BaseModel):
    indexed: list[IngestedDoc]
    skipped: list[dict]
    failed: list[dict]
    total_chunks: int
    total_documents_in_index: int


# --------------------------------------------------------------------------- #
# /documents
# --------------------------------------------------------------------------- #
class DocumentInfo(BaseModel):
    doc_id: str
    title: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = None
    chunks: int = 0
    ingested_at: Optional[str] = None
    status: Optional[str] = None


class DocumentsResponse(BaseModel):
    count: int
    total_chunks: int
    documents: list[DocumentInfo]


# --------------------------------------------------------------------------- #
# /feedback
# --------------------------------------------------------------------------- #
class FeedbackRequest(BaseModel):
    trace_id: Optional[str] = Field(default=None, description="trace_id from a /query response.")
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(default=None, max_length=2000)
    question: Optional[str] = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    id: str
    trace_id: Optional[str]
    rating: str
    orphan: bool
    created_at: str


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    index_documents: int
    index_chunks: int
    llm_provider: str
    embedding_provider: str
    web_search_enabled: bool


class ErrorResponse(BaseModel):
    detail: str
    trace_id: Optional[str] = None
