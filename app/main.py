"""FastAPI application factory: middleware, exception handlers, lifespan, routers.

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__, metrics
from app.config import get_settings
from app.logging_config import configure_logging, get_logger, get_trace_id, set_trace_id

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    settings.ensure_dirs()
    logger.info("Starting %s v%s (env=%s)", settings.app_name, __version__, settings.environment)

    # Initialize feedback DB.
    from app.services import feedback as feedback_service

    feedback_service.init_db()

    # Warm the vector store; auto-ingest the bundled corpus if the index is empty.
    from app.core import ingestion
    from app.core.vectorstore import get_vectorstore
    from app.services import registry

    vs = get_vectorstore()
    if vs.is_empty():
        logger.info("Index empty — auto-ingesting bundled corpus from %s", settings.corpus_dir)
        try:
            loaded = ingestion.load_corpus_dir(settings.corpus_dir, settings)
            if loaded:
                summary = vs.index_documents(loaded)
                registry.record_ingestion(summary["indexed"])
                logger.info("Auto-ingest indexed %d doc(s), %d chunk(s)",
                            len(summary["indexed"]), summary["chunks"])
            else:
                logger.warning("No documents found in corpus dir; index remains empty")
        except Exception:
            logger.exception("Auto-ingest failed; start the app and POST /ingest manually")
    else:
        logger.info("Index already populated with %d chunk(s)", vs.count())

    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="RAG-Based Technical Documentation Assistant",
        description=(
            "Self-corrective (CRAG/Self-RAG) Retrieval-Augmented Generation over technical docs, "
            "built as a LangGraph StateGraph and served via FastAPI."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Request middleware: trace id + access log + latency metric ----
    @app.middleware("http")
    async def trace_and_log(request: Request, call_next):
        incoming = request.headers.get("x-trace-id")
        tid = set_trace_id(incoming)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            metrics.inc("http_errors_total")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error", "trace_id": tid},
            )
        elapsed = round((time.perf_counter() - start) * 1000)
        response.headers["x-trace-id"] = tid
        metrics.observe_latency("http", elapsed, path=request.url.path)
        logger.info("%s %s -> %d (%dms)", request.method, request.url.path,
                    response.status_code, elapsed)
        return response

    # ---- Exception handlers (clean JSON, correct codes) ----
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "trace_id": get_trace_id()},
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "trace_id": get_trace_id()},
        )

    from app.api.routes import router

    app.include_router(router)

    @app.get("/", tags=["ops"])
    def root():
        return {
            "name": "RAG-Based Technical Documentation Assistant",
            "version": __version__,
            "docs": "/docs",
            "endpoints": ["/query", "/ingest", "/documents", "/feedback", "/healthz", "/metrics"],
        }

    return app


app = create_app()
