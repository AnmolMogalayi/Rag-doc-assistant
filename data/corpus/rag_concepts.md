# Retrieval-Augmented Generation (RAG) Concepts

Retrieval-Augmented Generation grounds an LLM's answers in an external knowledge base. Instead of
relying only on the model's parameters, a RAG system retrieves relevant documents at query time
and conditions generation on them, improving factual accuracy and enabling citations.

## The RAG Pipeline

1. **Ingestion**: load documents, split them into chunks, embed each chunk, and store the
   embeddings in a vector store.
2. **Retrieval**: embed the user's query and find the most similar chunks.
3. **Generation**: pass the retrieved chunks as context to the LLM to produce a grounded answer.

## Chunking Strategies

Chunking splits long documents into retrievable units. Key parameters are **chunk size** and
**overlap**. Naïve fixed-size character splitting can cut sentences or code blocks in half.
Better strategies for technical content:

- **Structure-aware splitting**: split on Markdown headers or code boundaries first, so a chunk
  stays within one logical section.
- **Recursive splitting**: recursively split on a hierarchy of separators (paragraphs, then
  lines, then sentences) to respect natural boundaries.
- **Token-aware sizing**: size chunks by tokens (not characters) to match the embedding model's
  window. A common default is around 500–1000 tokens with 10–20% overlap.

Overlap preserves context across chunk boundaries so an answer is not lost at a split point.

## Embeddings

An embedding model maps text to a dense vector such that semantically similar text is nearby in
vector space. Options include local models (for example `sentence-transformers/all-MiniLM-L6-v2`,
which is small, fast, and free) and hosted models (for example OpenAI `text-embedding-3-small`).
Switching embedding models changes the vector dimension, so the index must be rebuilt.

## Vector Stores

A vector store indexes embeddings for fast similarity search. ChromaDB runs locally and supports
metadata filtering and persistence. FAISS is a very fast similarity-search library. Similarity is
typically measured by cosine similarity or L2 distance.

## Hybrid Search and Reranking

- **Dense retrieval** matches meaning via embeddings.
- **Sparse retrieval** (for example BM25) matches exact tokens, which is valuable for API names
  and symbols.
- **Hybrid search** combines both, often fused with **Reciprocal Rank Fusion (RRF)**, where each
  document's score is the sum of `1 / (k + rank)` across result lists.
- **Reranking** uses a cross-encoder to reorder candidates by true query relevance, then keeps
  the top few. This mitigates the "lost in the middle" problem where models ignore information
  buried in a long context.

A common production pattern is: retrieve many candidates with hybrid search, rerank them, and
pass only the top 5–10 to the LLM.

## Self-Corrective RAG: CRAG and Self-RAG

Basic RAG always trusts retrieval. **Corrective RAG (CRAG)** adds a grading step that judges
whether retrieved documents are relevant; if not, it rewrites the query and retrieves again, or
falls back to web search. **Self-RAG** adds reflection: after generating an answer, the system
checks whether the answer is **grounded** in the retrieved context (a hallucination check) and
whether it actually **answers the question**. These checks are implemented as additional grading
nodes with conditional routing, and every loop must be bounded by a retry limit to guarantee
termination.

## Handling Insufficient Context

If no relevant context is found even after rewriting and (optionally) web search, a good RAG
system returns an honest "I don't know" rather than hallucinating an answer. Citations should
reference only the chunks actually used to generate the answer.
