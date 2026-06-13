"""Optional cross-encoder reranking of retrieval candidates.

Uses FlashRank (small, fast, local ONNX cross-encoder) to reorder candidates by true
query relevance — the current best-practice step after hybrid retrieval. Degrades
gracefully: if FlashRank is unavailable, the original order is preserved.
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_ranker = None
_unavailable = False


def _get_ranker():
    global _ranker, _unavailable
    if _ranker is not None or _unavailable:
        return _ranker
    try:
        from flashrank import Ranker

        settings = get_settings()
        _ranker = Ranker(model_name=settings.reranker_model)
        logger.info("FlashRank reranker loaded: %s", settings.reranker_model)
    except Exception as exc:  # pragma: no cover
        _unavailable = True
        logger.warning("Reranker unavailable (%s); skipping rerank step", exc)
    return _ranker


def rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """Return the top_k docs reordered by cross-encoder relevance to `query`."""
    if not docs:
        return docs
    ranker = _get_ranker()
    if ranker is None:
        return docs[:top_k]

    try:
        from flashrank import RerankRequest

        passages = [
            {"id": i, "text": d.page_content, "meta": d.metadata}
            for i, d in enumerate(docs)
        ]
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        ordered = []
        for item in ranked[:top_k]:
            doc = docs[item["id"]]
            doc.metadata = {**doc.metadata, "rerank_score": float(item.get("score", 0.0))}
            ordered.append(doc)
        return ordered
    except Exception:  # pragma: no cover
        logger.exception("Rerank failed; returning original order")
        return docs[:top_k]
