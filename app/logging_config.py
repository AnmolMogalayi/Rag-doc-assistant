"""Structured, request-scoped logging.

Emits JSON logs (production) or human-readable logs (dev) and binds a per-request
`trace_id` via a contextvar so every log line for a query can be correlated.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

# Correlation id available to any log record within a request/graph run.
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def set_trace_id(trace_id: str | None = None) -> str:
    tid = trace_id or new_trace_id()
    trace_id_var.set(tid)
    return tid


def get_trace_id() -> str:
    return trace_id_var.get()


class _TraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.trace_id = trace_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": getattr(record, "trace_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any structured extras
        for key, val in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = val
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} "
            f"[{record.levelname:<7}] [{getattr(record, 'trace_id', '-')}] "
            f"{record.name}: {record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    # Clear pre-existing handlers (e.g. uvicorn reload double-config)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_TraceFilter())
    handler.setFormatter(_JsonFormatter() if json_logs else _TextFormatter())
    root.addHandler(handler)

    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "urllib3", "chromadb", "sentence_transformers", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
