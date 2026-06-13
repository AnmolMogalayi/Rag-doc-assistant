"""API routes: /query, /ingest, /documents, /feedback (+ /healthz, /metrics).

Input validation via Pydantic; meaningful HTTP status codes; errors are caught and
returned as clean JSON without leaking stack traces.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status

from app import __version__, metrics
from app.config import get_settings
from app.core import ingestion
from app.core.vectorstore import get_vectorstore
from app.logging_config import get_logger, get_trace_id
from app.schemas import (
    DocumentInfo,
    DocumentsResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IngestResponse,
    IngestUrlsRequest,
    QueryRequest,
    QueryResponse,
)
from app.services import feedback as feedback_service
from app.services import rag_service
from app.services import registry

logger = get_logger(__name__)
router = APIRouter()


# --------------------------------------------------------------------------- #
# POST /query
# --------------------------------------------------------------------------- #
@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest) -> QueryResponse:
    """Answer a question against the indexed corpus via the self-corrective RAG graph."""
    metrics.inc("queries_total")
    result = rag_service.answer_question(
        req.question,
        session_id=req.session_id,
        chat_history=[t.model_dump() for t in req.chat_history],
        top_k=req.top_k,
        max_retries=req.max_retries,
    )
    metrics.observe_latency("query", result["latency_ms"])
    metrics.inc("queries_grounded", grounded=result.get("grounded", "skipped"))
    if result.get("error"):
        metrics.inc("queries_error")
    return QueryResponse(**result)


# --------------------------------------------------------------------------- #
# POST /ingest  (multipart files and/or JSON urls)
# --------------------------------------------------------------------------- #
@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED,
             tags=["ingestion"])
async def ingest(
    files: list[UploadFile] | None = File(default=None),
    urls: str | None = Form(default=None, description="JSON array or comma-separated URLs"),
) -> IngestResponse:
    """Ingest documents from uploaded files and/or URLs. Idempotent (content-hash dedup)."""
    settings = get_settings()
    vs = get_vectorstore()
    loaded: list[ingestion.LoadedDoc] = []
    failed: list[dict] = []

    # ---- Files ----
    for uf in files or []:
        try:
            ext = Path(uf.filename or "").suffix.lower()
            if ext not in settings.allowed_extensions:
                failed.append({"source": uf.filename, "error": f"unsupported extension '{ext}'"})
                continue
            raw = await uf.read()
            if len(raw) > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                    f"{uf.filename} exceeds {settings.max_upload_mb}MB")
            with tempfile.NamedTemporaryFile("wb", suffix=ext, delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                loaded.append(ingestion.load_file(tmp_path))
                # preserve original filename as the source
                loaded[-1].source = uf.filename or loaded[-1].source
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to read upload %s", uf.filename)
            failed.append({"source": uf.filename, "error": str(exc)})

    # ---- URLs (accept JSON array or comma-separated) ----
    parsed_urls: list[str] = []
    if urls:
        import json as _json

        try:
            val = _json.loads(urls)
            parsed_urls = val if isinstance(val, list) else [str(val)]
        except Exception:
            parsed_urls = [u.strip() for u in urls.split(",") if u.strip()]
        # validate
        parsed_urls = IngestUrlsRequest(urls=parsed_urls).urls

    for url in parsed_urls:
        try:
            loaded.append(ingestion.load_url(url))
        except Exception as exc:
            logger.exception("Failed to fetch %s", url)
            failed.append({"source": url, "error": str(exc)})

    if not loaded and not failed:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Provide at least one file or URL to ingest.")

    summary = vs.index_documents(loaded) if loaded else {"indexed": [], "skipped": [], "chunks": 0}
    registry.record_ingestion(summary["indexed"])
    metrics.inc("ingested_documents", float(len(summary["indexed"])))
    metrics.inc("ingested_chunks", float(summary["chunks"]))

    return IngestResponse(
        indexed=[
            DocumentInfoToIngested(d) for d in summary["indexed"]
        ],
        skipped=summary["skipped"],
        failed=failed,
        total_chunks=summary["chunks"],
        total_documents_in_index=len(vs.list_documents()),
    )


def DocumentInfoToIngested(d: dict):  # noqa: N802 - small adapter
    from app.schemas import IngestedDoc

    return IngestedDoc(
        doc_id=d["doc_id"], title=d.get("title"), source=d.get("source"),
        source_type=d.get("source_type"), chunks=d.get("chunks", 0),
    )


# --------------------------------------------------------------------------- #
# GET /documents
# --------------------------------------------------------------------------- #
@router.get("/documents", response_model=DocumentsResponse, tags=["ingestion"])
def list_documents() -> DocumentsResponse:
    """List documents currently in the corpus/index."""
    vs = get_vectorstore()
    live = vs.list_documents()
    live_counts = {d["doc_id"]: d["chunks"] for d in live}
    reg = registry.list_documents(live_counts=live_counts)

    # Merge: prefer registry entries, add any live docs missing from the registry.
    by_id = {d["doc_id"]: d for d in reg}
    for d in live:
        by_id.setdefault(d["doc_id"], d)

    docs = [DocumentInfo(**{k: v for k, v in d.items() if k in DocumentInfo.model_fields})
            for d in by_id.values()]
    total_chunks = sum(d.chunks for d in docs)
    return DocumentsResponse(count=len(docs), total_chunks=total_chunks, documents=docs)


# --------------------------------------------------------------------------- #
# POST /feedback
# --------------------------------------------------------------------------- #
@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED,
             tags=["feedback"])
def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Record thumbs up/down + optional comment on an answer."""
    known = bool(req.trace_id) and rag_service.is_known_trace(req.trace_id)
    rec = feedback_service.record_feedback(
        rating=req.rating, trace_id=req.trace_id, question=req.question,
        comment=req.comment, known_trace=known if req.trace_id else False,
    )
    metrics.inc("feedback_total", rating=req.rating)
    return FeedbackResponse(**rec)


# --------------------------------------------------------------------------- #
# Health & metrics
# --------------------------------------------------------------------------- #
@router.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz() -> HealthResponse:
    settings = get_settings()
    vs = get_vectorstore()
    docs = vs.list_documents()
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        index_documents=len(docs),
        index_chunks=vs.count(),
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        web_search_enabled=settings.enable_web_search,
    )


@router.get("/metrics", tags=["ops"])
def metrics_endpoint() -> Response:
    return Response(content=metrics.render(), media_type="text/plain")
