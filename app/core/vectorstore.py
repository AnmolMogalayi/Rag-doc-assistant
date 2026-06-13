"""Vector store + hybrid retrieval.

Wraps a persistent ChromaDB collection and adds:
  - dense similarity search (embeddings)
  - sparse BM25 lexical search (great for exact API/symbol names)
  - Reciprocal Rank Fusion (RRF) to combine both
  - optional cross-encoder rerank of the fused candidate set

The store is the single source of truth for indexed chunks; the document registry
(app/services/registry.py) tracks doc-level metadata for GET /documents.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Optional

from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.core import reranker
from app.core.ingestion import LoadedDoc, chunk_documents
from app.core.llm import get_embeddings
from app.core.retries import llm_retry
from app.logging_config import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Owns the Chroma collection and an in-memory BM25 index over the same chunks."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._lock = threading.RLock()
        self._chroma = None
        self._bm25 = None          # rank_bm25 retriever (rebuilt on writes)
        self._bm25_docs: list[Document] = []

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def chroma(self):
        if self._chroma is None:
            from langchain_chroma import Chroma

            self._chroma = Chroma(
                collection_name=self.settings.collection_name,
                embedding_function=get_embeddings(),
                persist_directory=str(self.settings.chroma_dir),
            )
            logger.info("Chroma collection '%s' ready at %s",
                        self.settings.collection_name, self.settings.chroma_dir)
        return self._chroma

    def _all_chunks(self) -> list[Document]:
        """Pull every stored chunk back out of Chroma (used to (re)build BM25)."""
        try:
            data = self.chroma.get(include=["documents", "metadatas"])
        except Exception:
            logger.exception("Failed to read chunks from Chroma")
            return []
        docs = []
        for text, meta in zip(data.get("documents", []), data.get("metadatas", [])):
            docs.append(Document(page_content=text, metadata=meta or {}))
        return docs

    def _rebuild_bm25(self) -> None:
        if not self.settings.use_hybrid_search:
            return
        try:
            from langchain_community.retrievers import BM25Retriever

            docs = self._all_chunks()
            self._bm25_docs = docs
            if docs:
                self._bm25 = BM25Retriever.from_documents(docs)
                self._bm25.k = self.settings.retrieval_fetch_k
            else:
                self._bm25 = None
            logger.info("BM25 index rebuilt over %d chunk(s)", len(docs))
        except Exception:  # pragma: no cover
            logger.exception("Failed to build BM25 index; hybrid search disabled this session")
            self._bm25 = None

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    @llm_retry(max_attempts=3)
    def _add(self, chunks: list[Document], ids: list[str]) -> None:
        self.chroma.add_documents(chunks, ids=ids)

    def index_documents(self, loaded: list[LoadedDoc]) -> dict:
        """Chunk + embed + upsert documents. Returns a per-doc summary.

        Idempotent: a doc whose content_hash already exists is skipped (re-ingest safe).
        """
        with self._lock:
            existing_hashes = self.existing_content_hashes()
            summary = {"indexed": [], "skipped": [], "chunks": 0}
            to_index: list[LoadedDoc] = []
            for d in loaded:
                if d.content_hash in existing_hashes:
                    summary["skipped"].append({"doc_id": d.doc_id, "title": d.title,
                                               "reason": "duplicate_content"})
                else:
                    to_index.append(d)

            if to_index:
                chunks = chunk_documents(to_index, self.settings)
                ids = [c.metadata["chunk_id"] for c in chunks]
                self._add(chunks, ids)
                summary["chunks"] = len(chunks)
                per_doc = defaultdict(int)
                for c in chunks:
                    per_doc[c.metadata["doc_id"]] += 1
                for d in to_index:
                    summary["indexed"].append({
                        "doc_id": d.doc_id, "title": d.title, "source": d.source,
                        "source_type": d.source_type, "chunks": per_doc[d.doc_id],
                        "content_hash": d.content_hash,
                    })
                self._rebuild_bm25()
            logger.info("Indexed %d new doc(s), skipped %d, %d chunk(s)",
                        len(summary["indexed"]), len(summary["skipped"]), summary["chunks"])
            return summary

    # ------------------------------------------------------------------ #
    # Reads / stats
    # ------------------------------------------------------------------ #
    def existing_content_hashes(self) -> set[str]:
        try:
            data = self.chroma.get(include=["metadatas"])
        except Exception:
            return set()
        return {m.get("content_hash") for m in data.get("metadatas", []) if m}

    def count(self) -> int:
        try:
            return self.chroma._collection.count()
        except Exception:
            return len(self._all_chunks())

    def is_empty(self) -> bool:
        return self.count() == 0

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    @llm_retry(max_attempts=3)
    def _dense(self, query: str, k: int) -> list[tuple[Document, float]]:
        # Chroma returns (doc, distance); smaller distance = more similar.
        return self.chroma.similarity_search_with_score(query, k=k)

    def retrieve(self, query: str, k: Optional[int] = None) -> list[Document]:
        """Hybrid retrieve: dense + BM25 fused by RRF, then optional rerank."""
        k = k or self.settings.top_k
        fetch_k = self.settings.retrieval_fetch_k

        dense_hits = self._dense(query, fetch_k)
        dense_docs = [d for d, _ in dense_hits]
        for rank, (d, dist) in enumerate(dense_hits):
            d.metadata = {**d.metadata, "dense_rank": rank, "dense_distance": float(dist)}

        if self.settings.use_hybrid_search:
            if self._bm25 is None:
                self._rebuild_bm25()
            sparse_docs = []
            if self._bm25 is not None:
                try:
                    sparse_docs = self._bm25.invoke(query)
                except Exception:
                    logger.exception("BM25 query failed; using dense only")
            fused = _reciprocal_rank_fusion([dense_docs, sparse_docs])
        else:
            fused = dense_docs

        candidates = fused[: max(fetch_k, k)]

        if self.settings.use_reranker and candidates:
            return reranker.rerank(query, candidates, k)
        return candidates[:k]

    def list_documents(self) -> list[dict]:
        """Aggregate stored chunks into a per-document view."""
        docs = self._all_chunks()
        agg: dict[str, dict] = {}
        for c in docs:
            did = c.metadata.get("doc_id", "unknown")
            entry = agg.setdefault(did, {
                "doc_id": did,
                "title": c.metadata.get("title"),
                "source": c.metadata.get("source"),
                "source_type": c.metadata.get("source_type"),
                "chunks": 0,
            })
            entry["chunks"] += 1
        return sorted(agg.values(), key=lambda e: (e.get("title") or ""))


def _reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Combine ranked lists via RRF. score = sum 1/(k + rank). De-dups by chunk_id."""
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc.metadata.get("chunk_id") or doc.page_content[:120]
            scores[key] += 1.0 / (k + rank)
            by_id.setdefault(key, doc)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for key, score in ordered:
        doc = by_id[key]
        doc.metadata = {**doc.metadata, "rrf_score": round(score, 6)}
        out.append(doc)
    return out


# Process-wide singleton (the API and graph share one manager).
_manager: Optional[VectorStoreManager] = None
_manager_lock = threading.Lock()


def get_vectorstore() -> VectorStoreManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = VectorStoreManager()
    return _manager


def set_vectorstore(manager: VectorStoreManager) -> None:
    """Inject a manager (used by tests)."""
    global _manager
    _manager = manager
