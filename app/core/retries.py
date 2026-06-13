"""Resilience helpers: a single decorator for retrying flaky LLM / network calls.

Uses tenacity with exponential backoff + jitter. Applied to every external LLM and
embedding call so transient 429/5xx/timeout errors do not surface as request failures.
"""
from __future__ import annotations

from typing import Callable, TypeVar

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Clearly non-transient errors — do NOT retry these (they will never succeed).
NON_RETRYABLE = (ValueError, TypeError, KeyError, NotImplementedError)


def _log_retry(retry_state) -> None:  # noqa: ANN001
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retrying %s after error: %r (attempt %d)",
        getattr(retry_state.fn, "__name__", "call"),
        exc,
        retry_state.attempt_number,
    )


def llm_retry(max_attempts: int = 3):
    """Decorator factory: retry a callable with exponential backoff + jitter.

    Retries on any exception except the obviously non-transient ones in NON_RETRYABLE.
    """

    def _wrap(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=1, max=20),
            retry=retry_if_not_exception_type(NON_RETRYABLE),
            before_sleep=_log_retry,
        )
        def _inner(*args, **kwargs) -> T:
            return func(*args, **kwargs)

        return _inner

    return _wrap
