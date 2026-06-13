# Phase 3 — Testing Strategy, Validation Checklist & Traceability

---

## 1. Testing Strategy

### 1.1 Principles
- **Offline & deterministic.** The whole suite runs with **no API keys, no network, no model
  downloads** (CI-safe). LLM calls and the vector store are replaced with fakes/monkeypatches, so
  tests are fast and reproducible.
- **Test the logic that matters most.** The assignment singles out the **state schema + retry
  tracking + routing** as a core criterion, so routing is unit-tested exhaustively as pure
  functions over state.
- **Test pyramid.** Many fast unit tests (chunking, fusion, routing, nodes) + a layer of
  integration tests (FastAPI contract via `TestClient`) + a manual/e2e checklist for real LLMs.

### 1.2 Test layers

| Layer | File | What it proves | LLM? |
|-------|------|----------------|------|
| Unit — ingestion | `tests/test_chunking.py` | Structure-aware chunking, complete metadata, unique chunk ids, HTML stripping | No |
| Unit — retrieval fusion | `tests/test_fusion.py` | RRF de-dups and ranks docs appearing in multiple lists higher | No |
| Unit — routing | `tests/test_routing.py` | Every conditional-edge branch incl. **retry-limit boundary** and empty-index short-circuit | No |
| Unit — nodes | `tests/test_nodes.py` | Grading filters irrelevant + **fails open**; generation builds numbered citations; "I don't know" path; retry increment; empty-index detection | Mocked |
| Integration — API | `tests/test_api.py` | All endpoints, status codes, **422 validation**, feedback orphan/known handling, ingest-requires-input | Mocked |
| E2E (manual) | `docs` checklist §3 | Real provider answers, citations, corrective loop, hallucination caveat | Real |

### 1.3 Verified results
- `pytest -q` → **31 passed** (offline).
- **End-to-end graph run with fakes** (documented in README) confirms the compiled `StateGraph`
  executes both the happy path (grade filters 1/2 → cite → grounded, confidence 0.85) and the
  corrective loop (no relevant docs → rewrite ×2 → **stops at retry limit** → grounded
  "I don't know"), proving termination with no infinite loop.

### 1.4 How to run
```bash
pip install -r requirements.txt
pytest -q                 # unit + integration (offline)
ruff check app tests      # lint (optional)
```

### 1.5 Suggested next steps (not implemented, documented honestly)
- **RAGAS-style eval harness** (faithfulness, answer-relevancy, context-precision/recall) over a
  small labeled question set, gated in CI with thresholds.
- **Load test** (`locust`/`k6`) on `/query` to characterize latency under concurrency.
- **Contract snapshot tests** for the OpenAPI schema to catch breaking API changes.

---

## 2. Validation Checklist

### Functional
- [x] LangGraph `StateGraph` compiles with all nodes (`analyze_query, retrieve, grade_documents,
      transform_query, web_search, generate, grade_generation`) — verified by smoke test.
- [x] Query Analysis rewrites/expands the query and classifies its type.
- [x] Retrieval returns top-k chunks **with source metadata**.
- [x] Document Grading scores each chunk relevant/irrelevant via structured LLM output.
- [x] If all chunks irrelevant → fallback (rewrite→retry→web→"I don't know").
- [x] If some relevant → irrelevant filtered out, relevant proceed.
- [x] Generation produces a grounded answer **with inline `[n]` citations**.
- [x] ≥1 **conditional edge** routes on the grading outcome.
- [x] **Retry limit** enforced (counter in state + recursion limit) — verified terminates.
- [x] Ingestion loads files (md/txt/html/rst) **and** URLs.
- [x] Chunking strategy implemented **and documented** (Phase 1 §6 + README).
- [x] Embeddings generated and stored in a persistent vector store (Chroma).
- [x] `POST /query`, `POST /ingest`, `GET /documents`, `POST /feedback` all present & working.

### Non-functional
- [x] Input validation (Pydantic) → **422** on bad input; **400/413** on bad ingest.
- [x] Meaningful HTTP status codes (200/201/400/413/422/500).
- [x] Error handling: handlers return clean JSON with `trace_id`; no stack-trace leakage.
- [x] Retries with backoff on LLM/embedding calls (tenacity); graders fail open.
- [x] Structured logging with per-request `trace_id`; `/metrics` exposed.
- [x] Secrets only via env; `.env` git-ignored; upload size/type guards.
- [x] Runs with **no paid key** for retrieval (local embeddings); provider-agnostic LLM.
- [x] Tests pass offline (31).

### Deliverables
- [x] Source code (repo layout), README with overview/arch/setup/keys/run/examples/design.
- [x] Working FastAPI app runnable locally (uvicorn + Docker).
- [x] Document corpus bundled + fetch script.
- [x] Write-up: reasoning, assumptions, chunking/embedding choices, what to improve.

### Bonus
- [x] Hallucination/groundedness check (Self-RAG). 
- [x] Web-search fallback (Tavily, feature-flagged).
- [x] Conversation memory (checkpointer + `session_id`).
- [x] Simple UI (Streamlit).

---

## 3. Manual E2E Checklist (with a real LLM key)

1. `cp .env.example .env`, set `LLM_PROVIDER` + key, `uvicorn app.main:app --reload`.
2. `GET /healthz` → `index_documents >= 4` (auto-ingested corpus).
3. `POST /query` "How do conditional edges work in LangGraph?" → grounded answer, `query_type`
   `how_to`, citations referencing `langgraph_guide.md`, `grounded=grounded`.
4. `POST /query` an out-of-corpus question (e.g. "What is the capital of France?") → corrective
   loop runs (`retries` > 0) and returns an honest "I don't know" (no fabrication).
5. `POST /ingest` a new URL → `indexed` non-empty; re-POST the same URL → it appears in `skipped`
   (idempotent dedup).
6. `GET /documents` → lists corpus + the newly ingested doc with chunk counts.
7. `POST /feedback` with the `trace_id` from step 3 → `201`, `orphan=false`.
8. `GET /metrics` → counters for `queries_total`, `feedback_total`, latency summaries.
9. (If enabled) follow-up query with same `session_id` + `chat_history` resolves a pronoun.

---

## 4. Traceability Matrix — PDF requirement → implementation → test

| PDF requirement | Implementation (file:symbol) | Test / evidence |
|-----------------|------------------------------|-----------------|
| LangGraph `StateGraph` workflow | `app/graph/workflow.py:build_graph` | smoke: nodes listed; `test_routing.py` |
| Node 1: rewrite/expand query | `app/graph/nodes.py:analyze_query`, `graders.analyze_query` | `test_nodes` (via graph e2e) |
| Node 1: classify query type | `graders.QueryAnalysis.query_type` | e2e prints `query_type=how_to` |
| Node 2: vector similarity search | `app/core/vectorstore.py:VectorStoreManager.retrieve` | `test_nodes.test_retrieve_*`, `test_fusion` |
| Node 2: top-k + source metadata | `ingestion.chunk_document` metadata; `nodes.retrieve` | `test_chunking.test_chunk_metadata_is_complete` |
| Node 3: LLM grades each chunk | `app/graph/graders.py:grade_document` (structured) | `test_nodes.test_grade_documents_filters_irrelevant` |
| Node 3: all irrelevant → fallback | `workflow.decide_to_generate` → transform/web/give_up | `test_routing.test_no_docs_*` + e2e corrective loop |
| Node 3: some relevant → filter | `nodes.grade_documents` keeps `binary_score=="yes"` | `test_nodes.test_grade_documents_filters_irrelevant` |
| Node 4: grounded answer | `nodes.generate` + `prompts.GENERATE_SYSTEM` | e2e answer output |
| Node 4: citations | `nodes._format_context` → numbered `citations` | `test_nodes.test_generate_builds_numbered_citations` |
| Conditional edge on grading | `workflow.add_conditional_edges("grade_documents", decide_to_generate, ...)` | `test_routing` |
| Retry limit on rewrite loop | `state.retry_count`/`max_retries`; `decide_to_generate` | `test_routing.test_retry_counter_is_respected_as_limit` + e2e (stops at 2) |
| State schema design | `app/graph/state.py:RAGState` | documented; used throughout |
| Ingestion: files (md/txt/html) + URLs | `ingestion.load_file`/`load_url`/`load_corpus_dir` | `test_chunking.test_html_is_stripped_on_load` |
| Ingestion: chunking strategy | `ingestion.chunk_document` (header + recursive) | `test_chunking.test_header_path_captured_in_section` |
| Ingestion: embeddings | `app/core/llm.py:get_embeddings` | smoke (corpus → 32 chunks) |
| Ingestion: vector store | `vectorstore.index_documents` (Chroma persist) | runtime / `/healthz` |
| Document chunking, documented | Phase 1 §6 + README "Chunking" | docs |
| `POST /query` → answer + sources | `app/api/routes.py:query` | `test_api.test_query_returns_answer_and_sources` |
| `POST /ingest` files/URLs | `routes.ingest` | `test_api.test_ingest_requires_input` (+ manual) |
| `GET /documents` | `routes.list_documents` | `test_api.test_documents_listing` |
| `POST /feedback` 👍/👎 + comment | `routes.submit_feedback`, `services/feedback.py` | `test_api.test_feedback_*` |
| Error handling / validation / codes | `app/main.py` handlers; `schemas.py`; `routes` | `test_api.test_query_validation_*` |
| README (overview…design) | `README.md` | — |
| Working app locally | `app/main.py`; Dockerfile | smoke + manual |
| Corpus or fetch script | `data/corpus/*`, `scripts/fetch_corpus.py` | smoke (4 docs) |
| Write-up | README "Design Decisions"; Phase 1 §8 | — |
| **Bonus** hallucination check | `nodes.grade_generation`, `graders.grade_groundedness` | `test_routing.test_ungrounded_*` + e2e |
| **Bonus** web search fallback | `app/graph/web_search.py`; `decide_to_generate→web_search` | wiring; flag-gated |
| **Bonus** conversation memory | `MemorySaver` in `workflow.get_graph`; `session_id`/`chat_history` | smoke (MemorySaver OK) |
| **Bonus** simple UI | `app/ui/streamlit_app.py` | manual |
