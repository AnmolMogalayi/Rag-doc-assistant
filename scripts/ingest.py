"""Standalone ingestion CLI: load -> chunk -> embed -> store into the vector store.

Usage:
    python -m scripts.ingest --corpus-dir data/corpus
    python -m scripts.ingest --url https://example.com/docs/page
    python -m scripts.ingest --file ./notes.md --file ./api.html
    python -m scripts.ingest --reset            # wipe + reindex the bundled corpus
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from app.config import get_settings
from app.core import ingestion
from app.core.vectorstore import get_vectorstore
from app.logging_config import configure_logging
from app.services import registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--corpus-dir", help="Directory of docs to ingest")
    parser.add_argument("--file", action="append", default=[], help="File(s) to ingest")
    parser.add_argument("--url", action="append", default=[], help="URL(s) to ingest")
    parser.add_argument("--reset", action="store_true", help="Delete existing index first")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=False)

    if args.reset:
        print(f"Resetting index at {settings.chroma_dir}")
        shutil.rmtree(settings.chroma_dir, ignore_errors=True)
        Path(settings.registry_path).unlink(missing_ok=True)
        settings.ensure_dirs()

    loaded: list[ingestion.LoadedDoc] = []
    corpus_dir = args.corpus_dir or (str(settings.corpus_dir)
                                     if not (args.file or args.url) else None)
    if corpus_dir:
        loaded += ingestion.load_corpus_dir(corpus_dir, settings)
    for f in args.file:
        try:
            loaded.append(ingestion.load_file(f))
        except Exception as exc:
            print(f"  ! failed to load {f}: {exc}", file=sys.stderr)
    for u in args.url:
        try:
            loaded.append(ingestion.load_url(u))
        except Exception as exc:
            print(f"  ! failed to fetch {u}: {exc}", file=sys.stderr)

    if not loaded:
        print("No documents to ingest.", file=sys.stderr)
        return 1

    vs = get_vectorstore()
    summary = vs.index_documents(loaded)
    registry.record_ingestion(summary["indexed"])

    print(f"\nIndexed {len(summary['indexed'])} document(s), "
          f"{summary['chunks']} chunk(s); skipped {len(summary['skipped'])} duplicate(s).")
    for d in summary["indexed"]:
        print(f"  - {d['title']} [{d['doc_id']}] ({d['chunks']} chunks)")
    print(f"Total chunks in index: {vs.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
