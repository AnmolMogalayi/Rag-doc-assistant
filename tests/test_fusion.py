"""Reciprocal Rank Fusion tests (pure)."""
from __future__ import annotations

from app.core.vectorstore import _reciprocal_rank_fusion
from tests.conftest import make_doc


def test_rrf_dedups_and_boosts_common_docs():
    a = make_doc("alpha", doc_id="a", n=0)
    b = make_doc("beta", doc_id="b", n=1)
    c = make_doc("gamma", doc_id="c", n=2)
    dense = [a, b, c]
    sparse = [b, a]  # b and a appear in both lists -> should rank above c
    fused = _reciprocal_rank_fusion([dense, sparse])
    ids = [d.metadata["doc_id"] for d in fused]
    assert set(ids) == {"a", "b", "c"}          # de-duplicated
    assert ids.index("c") == len(ids) - 1        # c (only one list) ranks last
    assert all("rrf_score" in d.metadata for d in fused)


def test_rrf_empty_lists():
    assert _reciprocal_rank_fusion([[], []]) == []
