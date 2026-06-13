"""Fetch a small technical-documentation corpus from official URLs into data/corpus/.

The repo already ships a curated Markdown corpus (so the project runs offline). This script
lets a reviewer refresh / extend it from upstream sources. Each page is saved as Markdown.

Usage:
    python -m scripts.fetch_corpus
    python -m scripts.fetch_corpus --url https://example.com/docs/page --name custom_page
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import httpx

from app.config import get_settings

# Curated, stable raw-Markdown sources (README / docs) for the default corpus.
DEFAULT_SOURCES = [
    ("fastapi_readme", "https://raw.githubusercontent.com/fastapi/fastapi/master/README.md"),
    ("pydantic_readme", "https://raw.githubusercontent.com/pydantic/pydantic/main/README.md"),
    ("langgraph_readme", "https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md"),
    ("langchain_readme", "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"),
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _to_markdownish(text: str, content_type: str) -> str:
    if "html" in content_type or text.lstrip().startswith("<"):
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
        text = _HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch_one(name: str, url: str, out_dir: Path) -> bool:
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True,
                         headers={"User-Agent": "rag-doc-assistant/1.0"})
        resp.raise_for_status()
        text = _to_markdownish(resp.text, resp.headers.get("content-type", ""))
        if not text:
            print(f"  ! {name}: empty content", file=sys.stderr)
            return False
        path = out_dir / f"{name}.md"
        path.write_text(text, encoding="utf-8")
        print(f"  ✓ {name} -> {path} ({len(text)} chars)")
        return True
    except Exception as exc:
        print(f"  ! {name}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch technical docs corpus.")
    parser.add_argument("--url", help="Fetch a single custom URL")
    parser.add_argument("--name", help="Filename stem for --url")
    parser.add_argument("--out", help="Output dir (default: configured corpus dir)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else get_settings().corpus_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching corpus into {out_dir}")
    ok = 0
    if args.url:
        name = args.name or re.sub(r"\W+", "_", args.url.split("//")[-1])[:40]
        ok += fetch_one(name, args.url, out_dir)
    else:
        for name, url in DEFAULT_SOURCES:
            ok += fetch_one(name, url, out_dir)
    print(f"Done. {ok} document(s) fetched.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
