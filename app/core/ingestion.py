"""Document ingestion: load -> clean -> structure-aware chunk -> (embed/store handled by store).

Chunking strategy (see docs/PHASE1_ANALYSIS.md §6):
  1. Markdown-header split to keep chunks within one logical section and capture the
     header path as metadata (used for human-friendly citations).
  2. Recursive, token-aware character split *within* each section, tuned for prose + code.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


@dataclass
class LoadedDoc:
    """A raw source document before chunking."""

    doc_id: str
    title: str
    source: str          # path or URL
    source_type: str     # file | url
    text: str
    content_hash: str
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "doc"


def _strip_html(html: str) -> str:
    # Lightweight HTML->text. For richer extraction, swap in trafilatura/bs4.
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    return text


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("# ").strip()
        if line:
            return line[:80]
    return fallback


def load_file(path: str | Path) -> LoadedDoc:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() in {".html", ".htm"}:
        text = _strip_html(raw)
    else:
        text = raw
    text = _MULTI_BLANK_RE.sub("\n\n", text).strip()
    title = _title_from_text(text, p.stem.replace("_", " ").title())
    return LoadedDoc(
        doc_id=f"{_slug(title)}-{_hash(text)[:6]}",
        title=title,
        source=str(p.name),
        source_type="file",
        text=text,
        content_hash=_hash(text),
    )


def load_url(url: str, *, timeout: int = 20) -> LoadedDoc:
    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                     headers={"User-Agent": "rag-doc-assistant/1.0"})
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    body = resp.text
    text = _strip_html(body) if ("html" in ctype or body.lstrip().startswith("<")) else body
    text = _MULTI_BLANK_RE.sub("\n\n", text).strip()
    title = _title_from_text(text, url.rstrip("/").split("/")[-1] or url)
    return LoadedDoc(
        doc_id=f"{_slug(title)}-{_hash(text)[:6]}",
        title=title,
        source=url,
        source_type="url",
        text=text,
        content_hash=_hash(text),
    )


def load_corpus_dir(corpus_dir: str | Path, settings: Settings | None = None) -> list[LoadedDoc]:
    settings = settings or get_settings()
    corpus_dir = Path(corpus_dir)
    docs: list[LoadedDoc] = []
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in settings.allowed_extensions:
            try:
                docs.append(load_file(p))
            except Exception:
                logger.exception("Failed to load %s", p)
    logger.info("Loaded %d document(s) from %s", len(docs), corpus_dir)
    return docs


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _length_fn(settings: Settings):
    """Token-aware length using tiktoken when available, else char/4 approximation."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda t: len(enc.encode(t))
    except Exception:  # pragma: no cover
        logger.warning("tiktoken unavailable; approximating token length by chars/4")
        return lambda t: max(1, len(t) // 4)


def chunk_document(doc: LoadedDoc, settings: Settings | None = None) -> list[Document]:
    """Split one LoadedDoc into LangChain Documents with rich metadata."""
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    settings = settings or get_settings()
    length_fn = _length_fn(settings)

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=length_fn,
        # Prefer splitting on structural boundaries before sentences/words.
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
    )

    section_docs: list[Document] = []
    looks_markdown = doc.source.endswith((".md", ".markdown")) or doc.text.lstrip().startswith("#")

    if settings.markdown_header_split and looks_markdown:
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        try:
            section_docs = header_splitter.split_text(doc.text)
        except Exception:
            logger.warning("Markdown header split failed for %s; falling back", doc.doc_id)
            section_docs = [Document(page_content=doc.text)]
    else:
        section_docs = [Document(page_content=doc.text)]

    chunks: list[Document] = []
    for sec in section_docs:
        header_path = " › ".join(
            sec.metadata.get(h) for h in ("h1", "h2", "h3") if sec.metadata.get(h)
        )
        for piece in recursive.split_text(sec.page_content):
            piece = piece.strip()
            if not piece:
                continue
            idx = len(chunks)
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "doc_id": doc.doc_id,
                        "chunk_id": f"{doc.doc_id}::{idx}",
                        "title": doc.title,
                        "source": doc.source,
                        "source_type": doc.source_type,
                        "section": header_path or doc.title,
                        "content_hash": doc.content_hash,
                        "tokens": length_fn(piece),
                    },
                )
            )
    logger.info("Chunked %s -> %d chunk(s)", doc.doc_id, len(chunks))
    return chunks


def chunk_documents(docs: Iterable[LoadedDoc], settings: Settings | None = None) -> list[Document]:
    out: list[Document] = []
    for d in docs:
        out.extend(chunk_document(d, settings))
    return out
