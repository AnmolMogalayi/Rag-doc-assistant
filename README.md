<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=1000&color=6C63FF&center=true&vCenter=true&multiline=true&repeat=true&width=700&height=100&lines=%F0%9F%A7%A0+RAG+Documentation+Assistant;Self-Corrective+%E2%80%A2+LangGraph+%E2%80%A2+FastAPI" alt="Typing SVG" />
</p>

<p align="center">
  <a href="#-highlights"><img src="https://img.shields.io/badge/✨_Highlights-6C63FF?style=for-the-badge" alt="Highlights"/></a>
  <a href="#-architecture"><img src="https://img.shields.io/badge/🏗️_Architecture-FF6B6B?style=for-the-badge" alt="Architecture"/></a>
  <a href="#-quick-start"><img src="https://img.shields.io/badge/🚀_Quick_Start-00C851?style=for-the-badge" alt="Quick Start"/></a>
  <a href="#-api--examples"><img src="https://img.shields.io/badge/🔌_API_Docs-FF9800?style=for-the-badge" alt="API"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/ChromaDB-FF6F00?style=flat-square&logo=databricks&logoColor=white" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/tests-31_passing-00C851?style=flat-square" alt="Tests"/>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"/>
</p>

<p align="center">
  <em>A <strong>self-corrective</strong> (CRAG + Self-RAG inspired) Retrieval-Augmented Generation system<br/>
  that answers questions about technical documentation with <strong>grounded citations</strong>.</em>
</p>

---

> **📋 Submission** for the **Express Analytics — AI/ML Engineer Intern** take-home assignment.  
> Implements every required component and adds production-grade engineering: hybrid retrieval,
> reranking, Self-RAG self-checks, web-search fallback, conversation memory, structured logging,
> metrics, retries, tests, and Docker — while staying fully compliant with the brief.
>
> 📄 **Deeper docs:** [`PHASE1_ANALYSIS.md`](docs/PHASE1_ANALYSIS.md) (requirements + architecture)
> · [`PHASE3_TESTING_AND_TRACEABILITY.md`](docs/PHASE3_TESTING_AND_TRACEABILITY.md)
> (testing strategy + requirement→code mapping)

---

## ✨ Highlights

<table>
<tr>
<td width="50%">

### 🔄 Self-Corrective RAG
- LangGraph `StateGraph` workflow
- Per-chunk LLM grading + conditional routing
- Bounded retry loop → honest "I don't know"
- Self-RAG groundedness & quality checks

</td>
<td width="50%">

### 🔍 Hybrid Retrieval
- Dense (Chroma) + BM25 lexical search
- Reciprocal Rank Fusion (RRF)
- Optional cross-encoder reranker (FlashRank)
- Multi-query expansion for wider recall

</td>
</tr>
<tr>
<td width="50%">

### 🛡️ Production Engineering
- Pydantic validation everywhere
- Tenacity retries + graceful degradation
- Structured JSON logging with `trace_id`
- 31 offline tests, Docker support

</td>
<td width="50%">

### 🌐 Provider Agnostic
- OpenAI / Anthropic / Groq / Google / Ollama
- Switch via a single env var
- Local embeddings by default — **no paid key** needed for retrieval
- Web-search fallback via Tavily

</td>
</tr>
</table>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=14&duration=4000&pause=2000&color=00C851&center=true&vCenter=true&repeat=true&width=500&lines=All+4+required+endpoints+%2B+%2Fhealthz+%2B+%2Fmetrics;Hallucination+check+%E2%9C%93+Web+fallback+%E2%9C%93+Memory+%E2%9C%93+UI+%E2%9C%93;31+offline+tests+%E2%80%A2+Zero+paid+keys+for+retrieval" alt="Features" />
</p>

---

## 🏗️ Architecture

### System Overview

```
              ┌──────────────────── FastAPI ────────────────────┐
 client ─────▶│  /query  /ingest  /documents  /feedback          │
 Streamlit    │  /healthz  /metrics   (Pydantic + error handlers)│
              └──────┬───────────────────────────┬──────────────┘
                     │                            │
          ┌──────────▼─────────┐      ┌───────────▼────────────┐
          │ Ingestion pipeline │      │  RAG LangGraph (compiled│
          │ load→chunk→embed→  │      │  StateGraph + memory)   │
          │ store              │      └───────────┬────────────┘
          └──────────┬─────────┘                  │
          ┌──────────▼────────────┐    ┌──────────▼──────────┐
          │ Chroma (persistent)   │◀───│ Hybrid retriever:    │
          │ + BM25 sparse index   │    │ dense+BM25 (RRF)+rerank
          └───────────────────────┘    └─────────────────────┘
```

<details>
<summary><strong>🔀 Click to expand: LangGraph Workflow Diagram</strong></summary>
<br/>

```
START → analyze_query → retrieve → grade_documents
                                       │  (conditional: decide_to_generate)
        ┌──────────────────────────────┼─────────────────────────────────┐
   relevant docs        no docs, retries left      retries gone           empty index
        │                     │                  (web on│off)                  │
        ▼                     ▼                     ▼     ▼                     ▼
     generate ◀── web_search  transform_query    web   give_up ──────────▶  generate
        │            ▲            │  (loop)      search    │              (grounded
        │            └────────────┘                │       │               "I don't know")
        ▼                                          ▼       ▼
  grade_generation  ◀──────────────────────────  generate
   (Self-RAG: grounded? useful?)
        │  (conditional: decide_after_generation)
        ├─ grounded & useful ─────────────▶ END
        ├─ ungrounded (budget left) ──────▶ generate   (regenerate)
        └─ not useful (retries left) ─────▶ transform_query
```

Both loops are bounded by counters in the **state schema** (`retry_count`/`max_retries`,
`generation_attempts`/`max_generation_attempts`) plus a graph `recursion_limit`, so the workflow
**always terminates**.

</details>

---

## 📁 Project Structure

```
rag-doc-assistant/
│
├── 📂 app/                          # Application source
│   ├── main.py                      # FastAPI app: lifespan, middleware, handlers
│   ├── config.py                    # pydantic-settings (all env config)
│   ├── logging_config.py            # Structured JSON logging + trace_id
│   ├── metrics.py                   # In-process metrics (/metrics)
│   ├── schemas.py                   # Pydantic request/response models
│   │
│   ├── 📂 api/
│   │   └── routes.py                # /query /ingest /documents /feedback /healthz /metrics
│   │
│   ├── 📂 core/
│   │   ├── llm.py                   # Provider-agnostic LLM + embeddings factory
│   │   ├── ingestion.py             # Loaders + structure-aware chunking
│   │   ├── vectorstore.py           # Chroma + BM25 hybrid + RRF
│   │   ├── reranker.py              # FlashRank cross-encoder (optional)
│   │   └── retries.py               # Tenacity backoff decorator
│   │
│   ├── 📂 graph/
│   │   ├── state.py                 # RAGState TypedDict (retry tracking)
│   │   ├── nodes.py                 # Node functions
│   │   ├── graders.py               # Structured LLM graders + query analysis
│   │   ├── prompts.py               # Prompt templates
│   │   ├── web_search.py            # Tavily fallback (bonus)
│   │   └── workflow.py              # StateGraph assembly + routers
│   │
│   ├── 📂 services/
│   │   ├── rag_service.py           # Graph runner / response shaping
│   │   ├── feedback.py              # SQLite feedback store
│   │   └── registry.py              # Document registry (JSON)
│   │
│   └── 📂 ui/
│       └── streamlit_app.py         # Streamlit UI (bonus)
│
├── 📂 scripts/
│   ├── fetch_corpus.py              # Fetch docs corpus from URLs
│   └── ingest.py                    # Standalone ingestion CLI
│
├── 📂 data/corpus/                  # Bundled Markdown corpus (4 docs)
├── 📂 tests/                        # 31 offline tests (no keys/network)
├── 📂 docs/                         # Phase 1 & Phase 3 deliverables
│
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata
├── Dockerfile                       # Container build
├── docker-compose.yml               # Multi-service orchestration
├── Makefile                         # Dev shortcuts
├── .env.example                     # Environment template
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- An API key for at least one LLM provider (Groq free tier works great)

### 1️⃣ Install

```bash
git clone https://github.com/AnmolMogalayi/Rag-doc-assistant.git
cd Rag-doc-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Configure

```bash
cp .env.example .env
```

Edit `.env` and set the key for your chosen provider:

<details>
<summary><strong>🔑 Provider Configuration Examples</strong></summary>

```ini
# ── OpenAI (easiest) ──
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# ── Groq (generous free tier — recommended to start) ──
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...

# ── Anthropic ──
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-haiku-latest
ANTHROPIC_API_KEY=sk-ant-...

# ── Google ──
LLM_PROVIDER=google_genai
LLM_MODEL=gemini-1.5-flash
GOOGLE_API_KEY=...

# ── Fully Local (no key at all) ──
# Run Ollama first, then:
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
```

> 💡 Embeddings stay local and free by default — **no embedding key needed**.

</details>

### 3️⃣ Run

```bash
uvicorn app.main:app --reload
```


### 4️⃣ (Optional) Streamlit UI

```bash
streamlit run app/ui/streamlit_app.py    
```

### 🐳 Docker

```bash
docker compose up                 # API on :8000
docker compose --profile ui up    # + Streamlit UI on :8501
```

---

## 🔌 API & Examples

<table>
<tr>
<th>Method</th>
<th>Endpoint</th>
<th>Purpose</th>
</tr>
<tr><td><code>POST</code></td><td><code>/query</code></td><td>Ask a question → grounded answer + sources</td></tr>
<tr><td><code>POST</code></td><td><code>/ingest</code></td><td>Ingest file uploads and/or URLs</td></tr>
<tr><td><code>GET</code></td><td><code>/documents</code></td><td>List documents in the corpus</td></tr>
<tr><td><code>POST</code></td><td><code>/feedback</code></td><td>Thumbs up/down + optional comment</td></tr>
<tr><td><code>GET</code></td><td><code>/healthz</code></td><td>Health + index/config status</td></tr>
<tr><td><code>GET</code></td><td><code>/metrics</code></td><td>Prometheus-style metrics</td></tr>
</table>

<details>
<summary><strong>📮 POST /query — Example Request & Response</strong></summary>

```bash
curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I route based on document grading in LangGraph?",
    "session_id": "demo",
    "top_k": 5
  }'
```

```jsonc
{
  "trace_id": "9f3a1c2b7e10",
  "question": "How do I route based on document grading in LangGraph?",
  "answer": "Use `workflow.add_conditional_edges(...)` with a router function ...",
  "sources": [
    {
      "title": "LangGraph Guide",
      "source": "langgraph_guide.md",
      "section": "LangGraph Guide › Core Concepts › Conditional Edges",
      "source_type": "file"
    }
  ],
  "citations": [{"n": 1, "doc_id": "langgraph-guide-ab12cd", "title": "LangGraph Guide"}],
  "query_type": "how_to",
  "confidence": 0.85,
  "retries": 0,
  "web_search_used": false,
  "grounded": "grounded",
  "answer_quality": "useful",
  "latency_ms": 2143,
  "steps": [
    "analyze_query: type=how_to ...",
    "retrieve: ...",
    "grade_documents: kept 3/5 ...",
    "generate: ...",
    "grade_generation: groundedness=grounded, answer=useful"
  ]
}
```
</details>

<details>
<summary><strong>📤 POST /ingest — Example</strong></summary>

```bash
# From URLs
curl -s -X POST http://localhost:8000/ingest \
  -F 'urls=["https://raw.githubusercontent.com/pydantic/pydantic/main/README.md"]'

# From file uploads
curl -s -X POST http://localhost:8000/ingest -F "files=@./mydoc.md"
```

```json
{
  "indexed": [{"doc_id": "...", "title": "...", "chunks": 12}],
  "skipped": [],
  "failed": [],
  "total_chunks": 12,
  "total_documents_in_index": 5
}
```
</details>

<details>
<summary><strong>📋 GET /documents & POST /feedback</strong></summary>

```bash
# List all documents
curl -s http://localhost:8000/documents

# Submit feedback
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"trace_id": "9f3a1c2b7e10", "rating": "up", "comment": "spot on"}'
```

> 💬 **Conversation memory:** Pass a stable `session_id` and the prior turns in `chat_history`;
> the graph resolves pronouns and keeps context across turns.

</details>

---

## 🧠 Design Decisions & Tradeoffs

<details>
<summary><strong>💭 Why a custom <code>TypedDict</code> state, not <code>MessagesState</code>?</strong></summary>

This is a structured pipeline, not a chat-tool agent. Explicit fields (`documents`,
`relevance_grades`, `retry_count`, `steps`) make routing logic and debugging clear, and make the
**retry tracking** the brief emphasizes a first-class, testable concern.
</details>

<details>
<summary><strong>🔄 Self-correction strategy (CRAG + Self-RAG)</strong></summary>

Document grading filters irrelevant chunks; if none survive, the graph rewrites the query and
re-retrieves up to `max_retries`, then optionally falls back to web search, and finally returns
an honest "I don't know" rather than hallucinating. After generation, a Self-RAG step checks
**groundedness** (is every claim supported?) and **answer quality** (does it actually answer?),
with one bounded regeneration. Every loop is counter-bounded + `recursion_limit`, guaranteeing
termination (verified in tests).
</details>

<details>
<summary><strong>🔍 Retrieval beyond the brief</strong></summary>

The PDF asks for similarity search; we add **hybrid (dense+BM25) + RRF + reranking** because
technical Q&A mixes semantics ("how do I validate input") with exact tokens (`field_validator`,
`add_conditional_edges`). Multi-query expansion further widens recall.
</details>

<details>
<summary><strong>✂️ Chunking strategy</strong></summary>

Two-stage: Markdown-header split (keeps a chunk in one section, records the header path for nice
citations) then token-aware recursive split (~800 tokens / 120 overlap). See
[Phase 1 §6](docs/PHASE1_ANALYSIS.md) for the full justification.
</details>

<details>
<summary><strong>🛡️ Reliability & graceful degradation</strong></summary>

Every external call is wrapped in tenacity backoff; graders **fail open** (keep context / accept
answer) so an LLM hiccup degrades quality but never 500s a request. The reranker, BM25, and web
search all degrade gracefully if unavailable.
</details>

<details>
<summary><strong>⚖️ Tradeoffs & future improvements</strong></summary>

**Scope limits (honest):** no auth/multi-tenancy, single-node deployment, SQLite + in-memory
checkpointer by default, non-streaming responses. Self-checks add LLM calls (latency/cost) —
they're feature-flagged via `ENABLE_SELF_CHECK`.

**What I'd improve with more time:** parent-document / late-chunking retrieval; streaming token
responses (SSE); a proper eval harness (RAGAS-style faithfulness/recall on a labeled question set
wired into CI); Postgres for feedback + `PostgresSaver` for durable memory; per-key rate limiting
and auth; caching of embeddings/answers; LangSmith dashboards wired by default.
</details>

---

## ⚙️ Configuration Reference

All settings live in `app/config.py` and are overridable via `.env` / env vars.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM backend: openai / anthropic / groq / google_genai / ollama |
| `LLM_MODEL` | `gpt-4o-mini` | Model name for the chosen provider |
| `EMBEDDING_PROVIDER` | `huggingface` | Embedding backend: huggingface (local) / openai |
| `CHUNK_SIZE` | `800` | Token chunk size for splitting |
| `TOP_K` | `5` | Number of documents to retrieve |
| `USE_HYBRID_SEARCH` | `true` | Enable BM25 + dense fusion |
| `USE_RERANKER` | `true` | Enable FlashRank cross-encoder |
| `MAX_RETRIES` | `2` | Max query rewrite retries |
| `ENABLE_SELF_CHECK` | `true` | Enable Self-RAG groundedness check |
| `ENABLE_WEB_SEARCH` | `false` | Enable Tavily web fallback |
| `ENABLE_MEMORY` | `true` | Enable conversation memory |

> 📝 See [`.env.example`](.env.example) for the annotated full list.

---

## ✅ Testing

```bash
pytest -q          # 31 tests, fully offline (no API keys, no network, no model downloads)
```

The suite covers:
- ✅ Chunking & metadata extraction
- ✅ RRF fusion logic
- ✅ Every routing decision (retry-limit behavior)
- ✅ Node logic with mocked LLMs (incl. fail-open)
- ✅ Full API contract (validation → 422, status codes, feedback orphan handling)

> 📄 See [`PHASE3_TESTING_AND_TRACEABILITY.md`](docs/PHASE3_TESTING_AND_TRACEABILITY.md) for the
> strategy, validation checklist, and the PDF-requirement → implementation matrix.

---

## 📚 Corpus

Ships with 4 curated Markdown docs in `data/corpus/`:

| Document | Topic |
|----------|-------|
| `langgraph_guide.md` | LangGraph concepts & patterns |
| `fastapi_guide.md` | FastAPI development guide |
| `pydantic_guide.md` | Pydantic validation |
| `rag_concepts.md` | RAG architecture concepts |

Works offline. Refresh/extend from upstream:
```bash
python -m scripts.fetch_corpus
```

---

## 🤝 Contributing

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=16&duration=4000&pause=2000&color=6C63FF&center=true&vCenter=true&repeat=true&width=500&lines=Built+with+%E2%9D%A4%EF%B8%8F+by+Anmol+Mogalayi;Express+Analytics+%E2%80%A2+AI%2FML+Engineer+Intern" alt="Footer" />
</p>

<p align="center">
  <a href="https://github.com/AnmolMogalayi/Rag-doc-assistant/stargazers">
    <img src="https://img.shields.io/github/stars/AnmolMogalayi/Rag-doc-assistant?style=social" alt="Stars"/>
  </a>
  <a href="https://github.com/AnmolMogalayi/Rag-doc-assistant/network/members">
    <img src="https://img.shields.io/github/forks/AnmolMogalayi/Rag-doc-assistant?style=social" alt="Forks"/>
  </a>
</p>
