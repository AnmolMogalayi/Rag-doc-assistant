# Phase 1 — Analysis & Architecture

> RAG-Based Technical Documentation Assistant with a self-corrective LangGraph workflow.
> This document is the result of reading the assignment PDF end-to-end and treating it as
> the source of truth, then layering current (2025–2026) RAG engineering best practices on top.

---

## 1. Assignment Summary

Build a **Retrieval-Augmented Generation (RAG)** system that answers natural-language questions
about a small corpus (3–5 docs) of **technical documentation**. The retrieval/answering pipeline
must be implemented as a **LangGraph `StateGraph`** with — at minimum — these nodes:

1. **Query Analysis** — rewrite/expand the query; optionally classify query type.
2. **Retrieval** — vector similarity search (ChromaDB/FAISS), top-k chunks + source metadata.
3. **Document Grading** — LLM grades each chunk relevant/irrelevant (the *self-corrective* core).
4. **Generation** — grounded answer **with citations**.

The graph must contain **at least one conditional edge** that routes on the grading outcome:
relevant → generate; none relevant → rewrite query and re-retrieve **with a retry limit**.

It must be served through **FastAPI** with `POST /query`, `POST /ingest`, `GET /documents`,
`POST /feedback`. A document **ingestion pipeline** (load → chunk → embed → store) is required,
and the **chunking strategy must be documented and justified**.

The evaluators explicitly state: *"There is no single correct architecture — we want to see how
you think,"* and *"We value clear thinking and honest documentation of tradeoffs over feature
completeness."* The **state schema and retry tracking are called out as a core evaluation criterion.**

**Bonus (optional):** hallucination check (Self-RAG), web-search fallback (Tavily/Serper/Exa),
conversation memory, and a simple UI (Streamlit/Gradio).

---

## 2. Requirements Breakdown (verbatim → obligation)

| # | PDF source | Obligation | Type | Status in this build |
|---|-----------|-----------|------|----------------------|
| R1 | Overview / Architecture | LangGraph `StateGraph` workflow | MUST | ✅ `app/graph/workflow.py` |
| R2 | Node 1 | Query Analysis: rewrite/expand query | MUST | ✅ `analyze_query` node |
| R3 | Node 1 | Optionally classify query type | SHOULD | ✅ classified into 4 types |
| R4 | Node 2 | Vector store similarity search (Chroma/FAISS) | MUST | ✅ Chroma + **hybrid BM25** |
| R5 | Node 2 | Return top-k chunks **with source metadata** | MUST | ✅ metadata preserved end-to-end |
| R6 | Node 3 | LLM grades each chunk relevant/irrelevant | MUST | ✅ `grade_documents` (structured) |
| R7 | Node 3 | If all irrelevant → fallback path | MUST | ✅ rewrite→retry→web→"I don't know" |
| R8 | Node 3 | If some relevant → filter, proceed | MUST | ✅ filtered list flows on |
| R9 | Node 4 | LLM generates grounded answer | MUST | ✅ `generate` node |
| R10 | Node 4 | Include citations/references | MUST | ✅ inline `[n]` + structured sources |
| R11 | Conditional Edges | ≥1 conditional edge on grading outcome | MUST | ✅ `decide_to_generate` router |
| R12 | Conditional Edges | Retry limit on rewrite/re-retrieve loop | MUST | ✅ `retry_count`/`max_retries` in state |
| R13 | Callout box | Deliberate state schema + retry tracking | MUST (scored) | ✅ `app/graph/state.py` (documented) |
| R14 | Ingestion | Load from files (md/txt/html) or URLs | MUST | ✅ `app/core/ingestion.py` loaders |
| R15 | Ingestion | Chunk with appropriate strategy | MUST | ✅ markdown-header + recursive |
| R16 | Ingestion | Generate embeddings | MUST | ✅ pluggable (local default) |
| R17 | Ingestion | Store in vector store | MUST | ✅ persistent Chroma |
| R18 | Ingestion | Document & justify chunking strategy | MUST | ✅ README + §6 here |
| R19 | API | `POST /query` → answer + sources | MUST | ✅ |
| R20 | API | `POST /ingest` → files or URLs | MUST | ✅ multipart files + JSON URLs |
| R21 | API | `GET /documents` → list corpus | MUST | ✅ |
| R22 | API | `POST /feedback` → 👍/👎 + comment | MUST | ✅ persisted to SQLite |
| R23 | API | Error handling, validation, HTTP codes | MUST | ✅ Pydantic + handlers + codes |
| R24 | Deliverables | GitHub repo, all source | MUST | ✅ repo layout |
| R25 | Deliverables | README (overview, arch, setup, keys, run, examples, design) | MUST | ✅ `README.md` |
| R26 | Deliverables | Working FastAPI app runnable locally | MUST | ✅ uvicorn / docker |
| R27 | Deliverables | Corpus or fetch script | MUST | ✅ bundled corpus + `fetch_corpus.py` |
| R28 | Deliverables | Write-up: reasoning, improvements, assumptions, chunking/embedding | MUST | ✅ README + this doc |
| B1 | Bonus | Hallucination / groundedness check (Self-RAG) | BONUS | ✅ `grade_generation` node |
| B2 | Bonus | Web-search fallback (Tavily/Serper/Exa) | BONUS | ✅ Tavily node (feature-flagged) |
| B3 | Bonus | Conversation memory (session history) | BONUS | ✅ checkpointer + session_id |
| B4 | Bonus | Simple UI (Streamlit/Gradio) | BONUS | ✅ Streamlit app |

---

## 3. Functional Requirements

- **FR-1 Ingestion**: Accept Markdown/text/HTML files and URLs; load, clean, chunk, embed, and
  persist to a vector store. Idempotent re-ingest (content-hash dedup). Each chunk carries
  `source`, `title`, `doc_id`, `chunk_id`, `section`/header path, and char offsets.
- **FR-2 Query analysis**: Normalize and rewrite the question for retrieval; produce 1–N search
  queries (multi-query expansion); classify into `{conceptual, how_to, troubleshooting, api_reference}`.
- **FR-3 Retrieval**: Hybrid (dense + BM25) retrieval fused by Reciprocal Rank Fusion, optional
  cross-encoder rerank, return top-k chunks with scores and metadata.
- **FR-4 Grading**: Per-chunk binary relevance via LLM structured output; keep relevant chunks.
- **FR-5 Corrective routing**: If no relevant chunk → rewrite & re-retrieve up to `max_retries`;
  then optional web-search fallback; else return a grounded "I don't know."
- **FR-6 Generation**: Produce a concise, accurate answer grounded only in kept context, with
  inline numbered citations mapped to source documents.
- **FR-7 Self-check (Self-RAG)**: Verify the answer is supported by context (groundedness) and
  actually addresses the question; one bounded regeneration on failure.
- **FR-8 Conversation memory**: Maintain per-`session_id` history to support follow-ups.
- **FR-9 API surface**: The 4 required endpoints + `GET /healthz`, `GET /metrics`, `GET /` docs.
- **FR-10 Feedback**: Persist thumbs up/down + comment, linked to a `trace_id`/query.
- **FR-11 Observability**: Structured request-scoped logs, per-node trace in the response,
  latency/usage metrics, optional LangSmith tracing.

---

## 4. Non-Functional Requirements

| Attribute | Target / Approach |
|-----------|-------------------|
| **Runnability** | `pip install` + one command; **runs with zero paid API keys** for embeddings (local `sentence-transformers`) — only an LLM key is needed, and any provider works. |
| **Portability** | Provider-agnostic LLM via `init_chat_model` (OpenAI / Anthropic / Groq / Google / Ollama). Vector store, embeddings, reranker all swappable by env. |
| **Reliability** | Tenacity retry+backoff on every LLM/embedding call; graceful degradation (rerank/web-search/grader failures never crash a query — they fall back). |
| **Resilience** | Hard retry ceiling in graph state prevents infinite loops; recursion limit on compiled graph. |
| **Validation** | Pydantic v2 request/response models; bounded `top_k`, `max_retries`; file-type/size guards on upload. |
| **Observability** | JSON structured logging with `trace_id`, per-query step trace, `/metrics`, optional LangSmith. |
| **Security** | Secrets only via env (`.env` git-ignored); input size limits; upload extension allow-list; CORS configurable; no secret logging. |
| **Performance** | Hybrid retrieval over a small corpus is sub-second; reranking bounded to candidate set; embeddings batched; persistent index avoids re-embedding. |
| **Scalability** | Stateless API workers; vector store and feedback externalizable (Chroma→server mode, SQLite→Postgres); checkpointer swappable (Memory→Postgres). |
| **Maintainability** | Layered modules (api/core/graph/services), typed, single-responsibility nodes, config centralized, tests with mocked LLMs (no network in CI). |
| **Testability** | LLM/embeddings injectable; deterministic fakes; routing logic unit-tested without a real model. |

---

## 5. Edge Cases Identified & Handling

| # | Edge case | Handling |
|---|-----------|----------|
| E1 | Empty / whitespace / overlong question | Pydantic validation → `422`; length cap. |
| E2 | Query against an **empty index** | Detected pre-retrieval → immediate grounded "no documents indexed" answer, `needs_ingestion` flag. |
| E3 | **All chunks graded irrelevant** | Corrective loop: rewrite → re-retrieve (≤ `max_retries`) → web fallback → "I don't know." |
| E4 | **Retry limit exhausted** | Stop looping; return best-effort/"I don't know" with `route="give_up"`. Never infinite loop. |
| E5 | **Hallucination** (answer unsupported) | `grade_generation` flags ungrounded → one bounded regeneration, else return with low-confidence caveat. |
| E6 | LLM/API timeout or 5xx | Tenacity exponential backoff + retry; final failure → `503` with clean message, no stack leak. |
| E7 | Grader returns malformed output | `with_structured_output` + try/except → default to "relevant" (fail-open to recall) and log. |
| E8 | Duplicate / re-ingested document | Content-hash dedup; existing `doc_id` upserted, not duplicated. |
| E9 | Unsupported file type / oversized upload | Extension allow-list + size cap → `400/413`. |
| E10 | Unreachable / non-HTML URL on ingest | Per-URL try/except; partial success report `{ingested, failed}`. |
| E11 | Embedding dim mismatch on model switch | Detected at startup; clear error instructing re-ingest into a fresh collection. |
| E12 | Web-search disabled but path reached | Skip cleanly to "I don't know" (no crash, flagged in trace). |
| E13 | Citations referencing filtered-out chunks | Citations built only from chunks actually passed to the generator. |
| E14 | Concurrent ingest + query | Chroma client is process-safe for our usage; reads tolerate writes; registry guarded. |
| E15 | Follow-up with no prior session | Unknown `session_id` → treated as fresh session (no error). |
| E16 | Feedback for unknown `trace_id` | Accepted and stored (analytics-friendly) but flagged `orphan=true`. |
| E17 | Non-English / code-heavy query | Embeddings are multilingual-capable; code fences preserved by chunker. |
| E18 | Extremely long document | Streaming/iterative chunking; batched embedding to bound memory. |

---

## 6. Chunking & Embedding Strategy (justification)

**Chunking — two-stage, structure-aware:**
1. **Markdown header split first** (`MarkdownHeaderTextSplitter`): technical docs are hierarchical
   (`# > ## > ###`). Splitting on headers keeps a chunk within one logical section and lets us
   record the **header path** as metadata (great for citations: *"FastAPI › Dependencies › Sub-dependencies"*).
2. **Recursive character split within sections** (`RecursiveCharacterTextSplitter`, separators
   tuned for prose + code, with `tiktoken`-based length). Default **chunk_size ≈ 800 tokens,
   overlap ≈ 120 (15%)** — large enough to hold a full API signature + explanation, small enough
   for precise grading and to avoid "lost in the middle." Code blocks are kept intact where possible.

**Why not naïve fixed-size character chunking?** It severs code blocks, splits a parameter from its
description, and produces low-precision chunks. Header-aware + token-aware splitting measurably
improves retrieval relevance and citation quality on technical docs.

**Embeddings — local-first, pluggable:** default `sentence-transformers/all-MiniLM-L6-v2`
(384-dim, fast, **free, no API key**) so the project runs out-of-the-box; switchable to
`OpenAIEmbeddings` (text-embedding-3-small) or any provider via config for higher quality.
Embeddings are batched and the index is persisted to disk so re-ingestion is incremental.

**Retrieval upgrade over the brief:** the PDF asks for plain similarity search. We implement
**hybrid retrieval** (dense vectors + BM25 lexical, fused with Reciprocal Rank Fusion) plus an
optional **cross-encoder reranker** — the current best-practice pattern, because exact API/symbol
names (e.g., `add_conditional_edges`) are matched by BM25 while semantics are matched by vectors.

---

## 7. Proposed Architecture

### 7.1 Component view
```
                         ┌──────────────────────────────────────────┐
   client / Streamlit ──▶│                FastAPI                    │
                         │  /query /ingest /documents /feedback      │
                         │  /healthz /metrics   (Pydantic + handlers)│
                         └───────┬───────────────────────┬──────────┘
                                 │                        │
                    ┌────────────▼─────────┐   ┌──────────▼───────────┐
                    │  Ingestion pipeline   │   │  RAG LangGraph app    │
                    │ load→chunk→embed→store│   │  (compiled StateGraph)│
                    └────────────┬─────────┘   └──────────┬───────────┘
                                 │                          │
            ┌────────────────────▼──────────┐     ┌─────────▼─────────┐
            │  Vector store (Chroma, persist)│◀────│ Hybrid retriever  │
            │  + BM25 sparse index           │     │ dense+BM25+rerank │
            └───────────────────────────────┘     └───────────────────┘
   Cross-cutting: provider-agnostic LLM/embeddings factory · tenacity retries ·
   structured logging+trace_id · feedback SQLite store · document registry · checkpointer
```

### 7.2 LangGraph workflow (state machine)
```
        START
          │
          ▼
   ┌─────────────┐
   │ analyze_query│  rewrite + expand + classify type
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │  retrieve   │  hybrid search → raw_documents (+metadata, scores)
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │grade_documents│ LLM per-chunk relevant? → keep relevant
   └──────┬──────┘
          ▼  (conditional: decide_to_generate)
   ┌──────┴───────────────────────────────────────────┐
   │ relevant docs?                                    │
   │   yes ───────────────────────────────▶ generate  │
   │   no & retries left ──▶ transform_query ──▶ retrieve (loop)
   │   no & retries gone & web enabled ───▶ web_search ─▶ generate
   │   no & retries gone & web off ───────▶ generate (grounded "I don't know")
   └───────────────────────────────────────────────────┘
          ▼
   ┌─────────────┐
   │  generate   │  grounded answer + numbered citations
   └──────┬──────┘
          ▼  (conditional: grade_generation — Self-RAG)
   ┌──────┴───────────────────────────────────────────┐
   │ grounded AND answers question? → END              │
   │ ungrounded & gen-retries left  → generate (retry) │
   │ not useful & retries left      → transform_query  │
   │ otherwise                      → END (with caveat)│
   └───────────────────────────────────────────────────┘
```

### 7.3 State schema (the scored core — see `app/graph/state.py`)
Key fields: `question`, `session_id`, `chat_history`, `query_type`, `search_queries`,
`raw_documents`, `documents` (kept), `relevance_grades`, **`retry_count` / `max_retries`**
(retry tracking), `generation_attempts`, `route`, `web_search_used`, `generation`,
`citations`, `hallucination_grade`, `answer_grade`, `steps` (trace), `error`, `confidence`.

---

## 8. Documented Engineering Decisions (gaps & ambiguities in the PDF)

| Ambiguity in PDF | Decision | Rationale |
|------------------|----------|-----------|
| LLM provider not fixed ("pick one") | **Provider-agnostic** via `init_chat_model`; default `openai:gpt-4o-mini`, but Anthropic/Groq/Google/Ollama supported by env | Maximizes portability & lets reviewers run with whatever key they have. |
| Embedding model unspecified | Default **local** `all-MiniLM-L6-v2`; OpenAI optional | Runs with **no paid key**; reviewer friction → 0. |
| Vector store Chroma *or* FAISS | **Chroma**, persistent | Built-in metadata filtering + persistence; simpler for the API/registry. |
| Corpus left to candidate | **FastAPI + LangGraph/LangChain + Pydantic** docs bundled as Markdown + `fetch_corpus.py` | Self-referential, well-structured, recognizable for reviewers. |
| top-k / chunk size / retries not given | `top_k=5`, `chunk≈800/overlap≈120`, `max_retries=2`, `recursion_limit=25` | Sensible defaults from §6 research; all env-configurable. |
| "fallback path" is open-ended | Tiered: rewrite→retry→(web)→grounded "I don't know" | Matches CRAG/Adaptive-RAG patterns; always terminates. |
| Grading granularity (per-chunk vs batch) | **Per-chunk** structured binary grade | Enables true filtering (R8) and explainable trace. |
| Feedback storage unspecified | **SQLite** table keyed by `trace_id` | Zero-config persistence; swappable to Postgres. |
| Session memory scope (bonus) | Per-`session_id` via LangGraph checkpointer | Standard LangGraph follow-up pattern. |
| Auth not mentioned | **No auth** by default; CORS + size limits + optional API-key header hook | Take-home scope; documented as a production gap. |

**Conscious scope limits (honest tradeoffs):** no multi-tenant auth, no distributed vector store,
no streaming token responses by default, single-node deployment. All are noted in the README
"What I'd improve with more time" section.
