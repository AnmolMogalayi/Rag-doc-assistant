"""Centralized, environment-driven configuration (12-factor).

Everything tunable lives here so the rest of the codebase reads config, never os.environ.
Values are validated by pydantic-settings and can be overridden via a `.env` file or env vars.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (app/config.py -> app -> root)
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- App -----
    app_name: str = "rag-doc-assistant"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: str = "*"  # comma-separated

    # ----- Paths -----
    corpus_dir: Path = ROOT_DIR / "data" / "corpus"
    chroma_dir: Path = ROOT_DIR / "data" / "chroma"
    feedback_db: Path = ROOT_DIR / "data" / "feedback.db"
    registry_path: Path = ROOT_DIR / "data" / "registry.json"

    # ----- LLM (provider-agnostic via langchain.init_chat_model) -----
    # Provider one of: openai | anthropic | groq | google_genai | ollama
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    # Optional separate, cheaper model for grading/rewriting (defaults to llm_model)
    fast_llm_model: str | None = None
    llm_temperature: float = 0.0
    generation_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout: int = 60
    llm_max_retries: int = 3

    # API keys (only the one matching llm_provider is required)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None
    google_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # ----- Embeddings -----
    # "huggingface" (local, free, default) or "openai"
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # ----- Chunking -----
    chunk_size: int = 800          # tokens (tiktoken)
    chunk_overlap: int = 120       # ~15%
    markdown_header_split: bool = True

    # ----- Retrieval -----
    collection_name: str = "tech_docs"
    top_k: int = 5                 # final chunks passed to grader/generator
    retrieval_fetch_k: int = 20    # candidates before rerank/fusion
    use_hybrid_search: bool = True # dense + BM25 (RRF)
    use_reranker: bool = True      # cross-encoder rerank of candidates
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"  # FlashRank model id

    # ----- Graph control -----
    max_retries: int = 2           # query-rewrite/re-retrieve loop limit
    max_generation_attempts: int = 1  # bounded regeneration on hallucination
    recursion_limit: int = 25      # LangGraph hard stop
    enable_self_check: bool = True # Self-RAG groundedness + answer grading

    # ----- Web search fallback (bonus) -----
    enable_web_search: bool = False
    tavily_api_key: str | None = None
    web_search_max_results: int = 3

    # ----- Conversation memory (bonus) -----
    enable_memory: bool = True

    # ----- Ingestion guards -----
    allowed_upload_extensions: str = ".md,.markdown,.txt,.html,.htm,.rst"
    max_upload_mb: int = 10

    # ----- Observability -----
    enable_langsmith: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "rag-doc-assistant"

    @field_validator("fast_llm_model", mode="before")
    @classmethod
    def _default_fast(cls, v, info):  # noqa: ANN001
        return v or None

    @property
    def fast_model(self) -> str:
        return self.fast_llm_model or self.llm_model

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extensions(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_upload_extensions.split(",") if e.strip()}

    def ensure_dirs(self) -> None:
        for p in (self.corpus_dir, self.chroma_dir, self.feedback_db.parent, self.registry_path.parent):
            Path(p).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton settings accessor."""
    s = Settings()
    s.ensure_dirs()
    return s
