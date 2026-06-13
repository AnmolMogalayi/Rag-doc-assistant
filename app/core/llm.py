"""Provider-agnostic LLM and embedding factories.

The whole pipeline talks to *roles* ("grader", "rewriter", "generator") rather than a
specific vendor. `init_chat_model` lets us swap OpenAI / Anthropic / Groq / Google / Ollama
purely via config, and embeddings default to a free local model so the app runs with no
paid keys.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def _export_provider_keys(settings: Settings) -> None:
    """Make the relevant key available under the env var the provider SDK expects."""
    mapping = {
        "openai": ("OPENAI_API_KEY", settings.openai_api_key),
        "anthropic": ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
        "groq": ("GROQ_API_KEY", settings.groq_api_key),
        "google_genai": ("GOOGLE_API_KEY", settings.google_api_key),
    }
    if settings.llm_provider in mapping:
        env_name, value = mapping[settings.llm_provider]
        if value and not os.environ.get(env_name):
            os.environ[env_name] = value


def _configure_langsmith(settings: Settings) -> None:
    if settings.enable_langsmith and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info("LangSmith tracing enabled (project=%s)", settings.langchain_project)


@lru_cache
def get_chat_model(role: str = "generator") -> Any:
    """Return a chat model for a role. Cached per (role) so we reuse clients.

    Roles use different temperatures and may use the cheaper `fast_model`:
      - grader / rewriter / router  -> fast_model, temperature 0 (deterministic)
      - generator                   -> main model, slightly warmer
    """
    from langchain.chat_models import init_chat_model

    settings = get_settings()
    _export_provider_keys(settings)
    _configure_langsmith(settings)

    is_generator = role == "generator"
    model = settings.llm_model if is_generator else settings.fast_model
    temperature = settings.generation_temperature if is_generator else settings.llm_temperature

    kwargs: dict[str, Any] = {
        "model": model,
        "model_provider": settings.llm_provider,
        "temperature": temperature,
        "timeout": settings.llm_timeout,
        "max_retries": 0,  # we handle retries via tenacity at call sites
    }
    if settings.llm_provider == "ollama":
        kwargs["base_url"] = settings.ollama_base_url
        kwargs.pop("timeout", None)
    if is_generator:
        kwargs["max_tokens"] = settings.llm_max_tokens

    logger.info("Initializing chat model role=%s provider=%s model=%s",
                role, settings.llm_provider, model)
    try:
        return init_chat_model(**kwargs)
    except Exception:  # pragma: no cover - surfaces a clear startup error
        logger.exception("Failed to initialize chat model for role=%s", role)
        raise


@lru_cache
def get_embeddings() -> Any:
    """Return an embeddings client (local HuggingFace by default, OpenAI optional)."""
    settings = get_settings()

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        _export_provider_keys(settings)
        logger.info("Using OpenAI embeddings: %s", settings.openai_embedding_model)
        return OpenAIEmbeddings(model=settings.openai_embedding_model)

    # Default: local sentence-transformers (free, no key).
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Using local HuggingFace embeddings: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def reset_model_cache() -> None:
    """Clear cached clients (used by tests that inject fakes)."""
    get_chat_model.cache_clear()
    get_embeddings.cache_clear()
