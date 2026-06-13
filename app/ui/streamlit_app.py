"""Minimal Streamlit UI (bonus) for interactive Q&A against the FastAPI backend.

Run the API first (uvicorn app.main:app), then:  streamlit run app/ui/streamlit_app.py
Set RAG_API_URL to point at a non-local backend.
"""
from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Docs Assistant", page_icon="📚", layout="centered")
st.title("📚 Technical Docs Assistant")
st.caption("Self-corrective RAG · LangGraph + FastAPI")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "history" not in st.session_state:
    st.session_state.history = []   # list of {role, content, meta}

# ---- Sidebar: corpus + settings ----
with st.sidebar:
    st.subheader("Corpus")
    try:
        docs = httpx.get(f"{API_URL}/documents", timeout=15).json()
        st.metric("Documents", docs.get("count", 0))
        st.metric("Chunks", docs.get("total_chunks", 0))
        for d in docs.get("documents", []):
            st.write(f"• **{d.get('title')}** ({d.get('chunks')} chunks)")
    except Exception as exc:
        st.error(f"API unreachable at {API_URL}: {exc}")

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("top_k", 1, 15, 5)
    max_retries = st.slider("max_retries", 0, 5, 2)

    st.divider()
    with st.expander("Ingest a URL"):
        url = st.text_input("Doc URL")
        if st.button("Ingest") and url:
            try:
                r = httpx.post(f"{API_URL}/ingest", data={"urls": url}, timeout=120).json()
                st.success(f"Indexed {len(r.get('indexed', []))} doc(s), "
                           f"{r.get('total_chunks', 0)} chunks.")
            except Exception as exc:
                st.error(str(exc))

# ---- Render chat history ----
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        meta = turn.get("meta")
        if meta:
            with st.expander("Sources, citations & trace"):
                for s in meta.get("sources", []):
                    st.write(f"- **{s.get('title')}** — {s.get('section')} ({s.get('source')})")
                st.caption(
                    f"type={meta.get('query_type')} · confidence={meta.get('confidence')} · "
                    f"grounded={meta.get('grounded')} · retries={meta.get('retries')} · "
                    f"{meta.get('latency_ms')}ms"
                )
                st.code("\n".join(meta.get("steps", [])), language="text")
                # Feedback buttons
                tid = meta.get("trace_id")
                c1, c2 = st.columns(2)
                if c1.button("👍", key=f"up-{tid}"):
                    httpx.post(f"{API_URL}/feedback",
                               json={"trace_id": tid, "rating": "up"}, timeout=15)
                    st.toast("Thanks for the feedback!")
                if c2.button("👎", key=f"down-{tid}"):
                    httpx.post(f"{API_URL}/feedback",
                               json={"trace_id": tid, "rating": "down"}, timeout=15)
                    st.toast("Thanks — we'll use this to improve.")

# ---- Chat input ----
if prompt := st.chat_input("Ask about the indexed documentation…"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking (retrieve → grade → generate → self-check)…"):
            try:
                payload = {
                    "question": prompt,
                    "session_id": st.session_state.session_id,
                    "top_k": top_k,
                    "max_retries": max_retries,
                    "chat_history": [
                        {"role": t["role"], "content": t["content"]}
                        for t in st.session_state.history[:-1]
                        if t["role"] in ("user", "assistant")
                    ][-6:],
                }
                resp = httpx.post(f"{API_URL}/query", json=payload, timeout=180).json()
                answer = resp.get("answer", "(no answer)")
                st.markdown(answer)
                st.session_state.history.append(
                    {"role": "assistant", "content": answer, "meta": resp})
                st.rerun()
            except Exception as exc:
                st.error(f"Query failed: {exc}")
