"""Web-search fallback (bonus). Uses Tavily when enabled and configured.

Returns LangChain Documents so the rest of the pipeline (generation, citations) is agnostic
to whether context came from the vector store or the web. Fails closed (empty list) on any error.
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def web_search(query: str) -> list[Document]:
    settings = get_settings()
    if not settings.enable_web_search or not settings.tavily_api_key:
        logger.info("Web search requested but disabled/unconfigured; skipping")
        return []
    try:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(
            max_results=settings.web_search_max_results,
            tavily_api_key=settings.tavily_api_key,
        )
        result = tool.invoke({"query": query})
        items = result.get("results", result) if isinstance(result, dict) else result
        docs: list[Document] = []
        for r in items or []:
            content = r.get("content") or r.get("snippet") or ""
            url = r.get("url", "web")
            title = r.get("title", url)
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "doc_id": f"web::{url}", "chunk_id": f"web::{url}",
                        "title": title, "source": url, "source_type": "web",
                        "section": "Web result",
                    },
                ))
        logger.info("Web search returned %d result(s)", len(docs))
        return docs
    except Exception:  # pragma: no cover
        logger.exception("Web search failed")
        return []
