# How to Run — Setup Guide & Manual Steps

This guide walks you through running the **RAG-Based Technical Documentation Assistant** from
scratch, and clearly separates **what works automatically** from **what you must add manually**.

> TL;DR: install deps → add **one LLM API key** to `.env` → `uvicorn app.main:app --reload`.
> Everything else (embeddings, corpus ingestion, vector store) is automatic and needs no key.

---

## 0. What you MUST add manually vs. what is automatic

| Thing | Required? | Automatic? | What you do |
|-------|-----------|------------|-------------|
| Python 3.10–3.13 | ✅ Required | — | Install it |
| Python dependencies | ✅ Required | — | `pip install -r requirements.txt` |
| **One LLM API key** (OpenAI/Groq/etc.) | ✅ Required* | ❌ Manual | Put it in `.env` (see §3) |
| `.env` file | ✅ Required | ❌ Manual | `cp .env.example .env` |
| Embedding model (local) | ✅ Required | ✅ Auto-downloads | Nothing — happens on first run |
| Document corpus | ✅ Required | ✅ Auto-ingested on startup | Nothing (bundled in `data/corpus/`) |
| Vector store (Chroma) | ✅ Required | ✅ Auto-created | Nothing |
| Web search (Tavily) | ⬜ Optional bonus | ❌ Manual | Add key + enable flag (see §6) |
| Non-OpenAI provider package | ⬜ Only if used | ❌ Manual | Uncomment 1 line in `requirements.txt` (see §3) |
| Reranker model (FlashRank) | ⬜ Optional | ✅ Auto-downloads | Nothing (degrades gracefully if offline) |

\* The **only mandatory manual step is one LLM key.** If you run a fully local LLM via **Ollama**,
you need no key at all (see §3, Option D).

---

## 1. Prerequisites

- **Python 3.10–3.13** — check with `python --version`
- **pip** and the ability to create a virtual environment
- ~2 GB free disk (for the local embedding model + dependencies)
- Internet access on first run (to download the embedding model) — afterwards it works offline
- *(Optional)* Docker Desktop if you prefer containers

---

## 2. Install

Open a terminal in the project folder (`rag-doc-assistant`).

### Windows (PowerShell)
```powershell
cd rag-doc-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux
```bash
cd rag-doc-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> First install can take a few minutes because `sentence-transformers` pulls in PyTorch.

---

## 3. Configure — add your LLM key (the one manual step that matters)

Create your `.env` from the template:
```bash
cp .env.example .env        # Windows PowerShell: copy .env.example .env
```

Open `.env` and pick **ONE** provider option below. Embeddings stay **local and free** in every
case — you only need a key for the LLM that grades and generates.

### Option A — OpenAI (default, recommended)
```ini
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...your-key...
```
Get a key at https://platform.openai.com/api-keys

### Option B — Groq (generous free tier)
1. Uncomment this line in `requirements.txt` and re-run `pip install -r requirements.txt`:
   ```
   langchain-groq>=0.2.1
   ```
2. In `.env`:
   ```ini
   LLM_PROVIDER=groq
   LLM_MODEL=llama-3.3-70b-versatile
   GROQ_API_KEY=gsk_...your-key...
   ```
   Get a key at https://console.groq.com/keys

### Option C — Anthropic / Google
- Anthropic: uncomment `langchain-anthropic>=0.2.4`, then
  ```ini
  LLM_PROVIDER=anthropic
  LLM_MODEL=claude-3-5-haiku-latest
  ANTHROPIC_API_KEY=sk-ant-...
  ```
- Google Gemini: uncomment `langchain-google-genai>=2.0`, then
  ```ini
  LLM_PROVIDER=google_genai
  LLM_MODEL=gemini-1.5-flash
  GOOGLE_API_KEY=...
  ```

### Option D — Fully local, NO key (Ollama)
1. Install Ollama from https://ollama.com and pull a model: `ollama pull llama3.1`
2. In `.env`:
   ```ini
   LLM_PROVIDER=ollama
   LLM_MODEL=llama3.1
   OLLAMA_BASE_URL=http://localhost:11434
   ```

---

## 4. Run the API

```bash
uvicorn app.main:app --reload
```

On first startup the app **automatically**:
- downloads the local embedding model (one-time),
- ingests the bundled corpus from `data/corpus/` (4 docs),
- creates the Chroma vector store under `data/chroma/`.

When you see `Application startup complete`, open the interactive docs:

👉 **http://localhost:8000/docs**

Quick health check:
```bash
curl http://localhost:8000/healthz
```
You should see `index_documents` ≥ 4.

### Try a query
```bash
curl -s http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"How do conditional edges work in LangGraph?\"}"
```

---

## 5. Run the tests (no keys needed)

The test suite is fully offline (no API key, no network):
```bash
pytest -q
```
Expected: **31 passed**.

---

## 6. OPTIONAL — enable bonus features manually

### 6a. Web-search fallback (Tavily)
When the corpus has no answer, the system can fall back to web search.
1. Uncomment in `requirements.txt` and reinstall:
   ```
   langchain-tavily>=0.1
   ```
2. In `.env`:
   ```ini
   ENABLE_WEB_SEARCH=true
   TAVILY_API_KEY=tvly-...your-key...
   ```
   Free key (1,000 searches/month) at https://tavily.com

### 6b. Streamlit UI
With the API running, in a second terminal:
```bash
streamlit run app/ui/streamlit_app.py
```
Opens a chat UI at http://localhost:8501 (set `RAG_API_URL` if the API is not on localhost:8000).

### 6c. Higher-quality embeddings (OpenAI)
In `.env`:
```ini
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
```
⚠️ Switching embedding models changes the vector dimension — **reset the index** afterwards:
```bash
python -m scripts.ingest --reset
```

### 6d. LangSmith tracing (observability)
```ini
ENABLE_LANGSMITH=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=rag-doc-assistant
```

---

## 7. Managing the corpus

- **Add your own docs at runtime** via the API:
  ```bash
  curl -X POST http://localhost:8000/ingest -F "files=@./my_doc.md"
  curl -X POST http://localhost:8000/ingest -F "urls=[\"https://example.com/docs.md\"]"
  ```
- **Ingest from the command line:**
  ```bash
  python -m scripts.ingest --corpus-dir data/corpus
  python -m scripts.ingest --url https://raw.githubusercontent.com/pydantic/pydantic/main/README.md
  python -m scripts.ingest --reset        # wipe and re-index
  ```
- **Refresh the bundled corpus from upstream:**
  ```bash
  python -m scripts.fetch_corpus
  ```

---

## 8. Run with Docker (alternative to §2–4)

```bash
cp .env.example .env        # add your LLM key first
docker compose up                  # API at http://localhost:8000
docker compose --profile ui up     # also starts the Streamlit UI at :8501
```
Your `data/` folder is mounted, so the index and feedback persist across restarts.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `401`/auth error on `/query` | Missing/invalid LLM key | Check the key in `.env` matches `LLM_PROVIDER` |
| `ModuleNotFoundError: langchain_groq` (or anthropic/google/tavily) | Provider package commented out | Uncomment it in `requirements.txt`, reinstall |
| `/healthz` shows `index_documents: 0` | Corpus didn't ingest | Run `python -m scripts.ingest --corpus-dir data/corpus` |
| First request very slow | Embedding model downloading | One-time; subsequent runs are fast |
| Answers say "I don't know" for in-corpus questions | Wrong/empty index after model switch | `python -m scripts.ingest --reset` |
| Reranker/web-search warnings in logs | Optional component unavailable | Safe to ignore — the system degrades gracefully |
| Want to start completely fresh | Stale index/feedback | `python -m scripts.ingest --reset` (or delete `data/chroma/`, `data/feedback.db`) |

---

## 10. What to check that it's working

1. `pytest -q` → **31 passed** (no keys needed).
2. `GET /healthz` → `status: ok`, `index_documents >= 4`.
3. `POST /query` with a LangGraph/FastAPI/Pydantic/RAG question → grounded answer **with citations**
   and a `sources` list.
4. `POST /query` with an unrelated question → honest "I don't know" after the corrective retry loop
   (`retries > 0`), not a hallucination.

You're done. 🎉
