"""Ingestion / chunking tests (pure — no LLM, no network)."""
from __future__ import annotations

from app.core import ingestion
from app.core.ingestion import LoadedDoc

MD = """# Title

Intro paragraph about the system.

## Section A

Content of section A with the symbol add_conditional_edges referenced.

## Section B

Content of section B. More text to ensure a reasonable chunk size and that the
recursive splitter has material to work with across the section boundary.
"""


def _doc() -> LoadedDoc:
    return LoadedDoc(doc_id="d-abc123", title="Title", source="d.md",
                     source_type="file", text=MD, content_hash="hash123")


def test_chunk_metadata_is_complete():
    chunks = ingestion.chunk_document(_doc())
    assert chunks, "expected at least one chunk"
    for i, c in enumerate(chunks):
        m = c.metadata
        assert m["doc_id"] == "d-abc123"
        assert m["chunk_id"] == f"d-abc123::{i}"
        assert m["source"] == "d.md"
        assert m["title"] == "Title"
        assert "section" in m and m["section"]
        assert isinstance(m["tokens"], int) and m["tokens"] > 0


def test_header_path_captured_in_section():
    chunks = ingestion.chunk_document(_doc())
    sections = {c.metadata["section"] for c in chunks}
    # The markdown header splitter should surface the section headers.
    assert any("Section A" in s for s in sections)
    assert any("Section B" in s for s in sections)


def test_chunk_ids_unique():
    chunks = ingestion.chunk_document(_doc())
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_html_is_stripped_on_load(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<html><body><h1>Hello</h1><p>World &amp; more</p></body></html>",
                 encoding="utf-8")
    loaded = ingestion.load_file(p)
    assert "<" not in loaded.text
    assert "Hello" in loaded.text and "World" in loaded.text
