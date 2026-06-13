"""Prompt templates for each LLM-backed node. Kept in one place for easy tuning."""
from __future__ import annotations

QUERY_ANALYSIS_SYSTEM = """You are a query-understanding module for a technical-documentation \
search engine. Given a user's question, you will:
1. Classify it into exactly one type: conceptual, how_to, troubleshooting, api_reference.
2. Rewrite it into a single, search-optimized query (resolve pronouns using the chat history, \
expand abbreviations, add the most likely relevant technical terms/synonyms).
3. Produce up to 3 alternative search queries that phrase the information need differently \
(include exact API/symbol names when relevant — lexical search benefits from them).
Stay faithful to the user's intent. Do not invent unrelated topics."""

QUERY_ANALYSIS_HUMAN = """Chat history (may be empty):
{history}

User question: {question}

Return the classification, the rewritten query, and alternative queries."""

GRADE_DOCUMENTS_SYSTEM = """You are a strict relevance grader for retrieved documentation chunks.
Decide whether a chunk contains information that helps answer the user's question.
Grade 'yes' only if the chunk is topically on-point and could contribute to a correct answer.
Grade 'no' for chunks that merely share keywords but do not address the question.
Be decisive; output only the structured score."""

GRADE_DOCUMENTS_HUMAN = """User question:
{question}

Retrieved chunk (source: {source} — {section}):
\"\"\"
{chunk}
\"\"\"

Is this chunk relevant to the question?"""

REWRITE_SYSTEM = """You rewrite search queries that previously failed to retrieve relevant \
documentation. Produce a single improved query that is more likely to match the corpus: \
broaden or re-focus the terms, add concrete technical vocabulary and likely synonyms, and \
remove noise. Output only the improved query text."""

REWRITE_HUMAN = """Original question: {question}
Previous query that returned no relevant results: {previous_query}
Attempt number: {attempt}

Write an improved search query."""

GENERATE_SYSTEM = """You are a precise technical-documentation assistant. Answer the user's \
question using ONLY the provided context chunks. Follow these rules:
- Ground every claim in the context. Do not use outside knowledge or invent details.
- Cite sources inline using bracketed numbers like [1], [2] that refer to the numbered \
context chunks. Cite the specific chunk(s) that support each statement.
- If the context is insufficient to fully answer, say what is missing rather than guessing.
- Prefer concise, well-structured answers. Use short code blocks when the context contains code.
- For how_to / troubleshooting questions, give actionable steps."""

GENERATE_HUMAN = """Question: {question}

Numbered context chunks:
{context}

Write a grounded answer with inline [n] citations."""

GENERATE_IDK = """You are a careful assistant. The retrieval system could not find documentation \
relevant to the user's question in the indexed corpus. Respond honestly that you don't have \
information about this in the available documentation, and (if reasonable) suggest how the user \
might rephrase or what to ingest. Do NOT fabricate an answer. Keep it to 2-3 sentences."""

HALLUCINATION_SYSTEM = """You are a groundedness judge (Self-RAG style). Determine whether the \
ASSISTANT ANSWER is fully supported by the provided CONTEXT. Grade 'yes' only if every factual \
claim in the answer can be traced to the context. Grade 'no' if the answer contains claims not \
supported by the context (potential hallucination). Output only the structured score."""

HALLUCINATION_HUMAN = """CONTEXT:
{context}

ASSISTANT ANSWER:
{answer}

Is the answer fully grounded in the context?"""

ANSWER_QUALITY_SYSTEM = """You judge whether an answer actually addresses the user's question. \
Grade 'yes' if it resolves the user's information need; 'no' if it is off-topic, evasive, or \
fails to answer. Output only the structured score."""

ANSWER_QUALITY_HUMAN = """User question: {question}

Assistant answer:
{answer}

Does the answer resolve the question?"""
