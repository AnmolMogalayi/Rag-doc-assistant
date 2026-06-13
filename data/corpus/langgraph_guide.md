# LangGraph Guide

LangGraph is a library for building stateful, multi-step applications with LLMs. It models
your application as a **graph** of nodes connected by edges, where a shared **state** object
flows between nodes. It is well suited to agentic and self-corrective RAG workflows.

## Core Concepts

### State
The state is a typed dictionary (often a `TypedDict`) that every node reads from and writes to.
Each node returns a partial update; LangGraph merges updates into the running state. You can
attach **reducers** to fields (for example, to append to a list instead of overwriting it) using
`Annotated[list, reducer_fn]`.

### Nodes
A node is a Python function (or runnable) that takes the current state and returns a dictionary
of state updates. Nodes are added with `workflow.add_node("name", fn)`. Keep nodes small and
single-purpose for testability.

### Edges
Edges connect nodes. A normal edge (`workflow.add_edge("a", "b")`) always goes from `a` to `b`.
The special constants `START` and `END` mark the graph entry and exit points.

### Conditional Edges
A conditional edge routes to different nodes based on a router function that inspects the state
and returns a key:

```python
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,            # returns a string key
    {"generate": "generate", "rewrite": "transform_query"},
)
```

Conditional edges are how you implement branching logic such as "if no relevant documents were
found, rewrite the query and retry."

## Building and Running a Graph

You construct a `StateGraph`, add nodes and edges, then `compile()` it:

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(MyState)
workflow.add_node("retrieve", retrieve)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", END)
graph = workflow.compile()

result = graph.invoke({"question": "..."})
```

### Recursion Limit
To prevent infinite loops in cyclic graphs, pass a `recursion_limit` in the config:
`graph.invoke(state, config={"recursion_limit": 25})`. If the limit is exceeded, LangGraph raises
a `GraphRecursionError`. Always bound retry loops with explicit counters in the state in addition
to the recursion limit.

## Checkpointers and Memory
LangGraph supports persistence via **checkpointers**. A checkpointer saves the state after each
super-step so a run can be resumed and so multi-turn conversations can share memory.

- `MemorySaver` — in-process, good for development and tests.
- `SqliteSaver` / `PostgresSaver` — durable persistence for production.

To use memory, compile with a checkpointer and pass a `thread_id`:

```python
graph = workflow.compile(checkpointer=MemorySaver())
graph.invoke(state, config={"configurable": {"thread_id": "user-123"}})
```

The `thread_id` scopes memory to a conversation, enabling follow-up questions.

## Self-Corrective RAG Patterns
LangGraph is commonly used to implement Corrective RAG (CRAG) and Self-RAG:

- **Document grading**: an LLM grades each retrieved chunk as relevant or not.
- **Query rewriting**: if retrieval fails, rewrite the query and retrieve again (bounded by a
  retry counter).
- **Groundedness checking**: after generation, verify the answer is supported by the context to
  catch hallucinations.

These reflective steps are implemented as ordinary nodes plus conditional edges that route on
the grading outcome.
