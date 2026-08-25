"""
Phase 4 + Phase 5 retrieval tests.

Phase 4 tests verify FAISS mechanics.

Phase 5 tests verify:
- lexical relevance can correct a dense-retrieval weakness
- active documents outrank superseded documents
- non-authoritative documents are penalized
- ranking remains deterministic
"""

import pytest

from app.chunking import Chunk, chunk_documents
from app.config import settings
from app.embeddings import HashingEmbedder
from app.ingestion import load_documents
from app.retrieval import VectorIndex, build_index


try:
    import sentence_transformers  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def _make_chunk(
    chunk_id: str,
    text: str,
    heading: str = "Section",
    status: str = "active",
    policy_authority: str = "official",
    customer_answering: bool = True,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        heading=heading,
        source_filename=f"{chunk_id}.md",
        document_id=f"FAKE-{chunk_id}",
        title="Fake Document",
        status=status,
        policy_authority=policy_authority,
        audience="customer",
        customer_answering=customer_answering,
        effective_date=None,
        supersedes=None,
        superseded_by=None,
    )


def test_search_returns_most_similar_chunk_first():
    chunks = [
        _make_chunk("a", "aster and row ships to canada with duties unpaid"),
        _make_chunk("b", "gift cards never expire and are final sale"),
        _make_chunk("c", "the standard return window is thirty calendar days"),
    ]

    index = build_index(chunks, HashingEmbedder())

    results = index.search("does aster and row ship to canada", k=1)

    assert results[0].chunk.chunk_id == "a"


def test_search_respects_k():
    chunks = [
        _make_chunk(str(i), f"chunk number {i} about topic {i}")
        for i in range(10)
    ]

    index = build_index(chunks, HashingEmbedder())

    results = index.search("chunk number 3 topic", k=3)

    assert len(results) == 3


def test_search_k_larger_than_corpus_does_not_crash():
    chunks = [
        _make_chunk("a", "only one chunk here"),
    ]

    index = build_index(chunks, HashingEmbedder())

    results = index.search("only one chunk", k=50)

    assert len(results) == 1


def test_build_rejects_empty_chunk_list():
    with pytest.raises(ValueError):
        build_index([], HashingEmbedder())


def test_search_before_build_raises():
    index = VectorIndex(HashingEmbedder())

    with pytest.raises(RuntimeError):
        index.search("anything")


def test_search_rejects_empty_query():
    index = build_index(
        [_make_chunk("a", "some text")],
        HashingEmbedder(),
    )

    with pytest.raises(ValueError):
        index.search("   ")


def test_search_rejects_non_positive_k():
    index = build_index(
        [_make_chunk("a", "some text")],
        HashingEmbedder(),
    )

    with pytest.raises(ValueError):
        index.search("some text", k=0)


def test_scores_are_descending():
    chunks = [
        _make_chunk(str(i), f"topic {i} content here")
        for i in range(5)
    ]

    index = build_index(chunks, HashingEmbedder())

    results = index.search("topic 2 content", k=5)

    scores = [r.score for r in results]

    assert scores == sorted(scores, reverse=True)


def test_active_document_outranks_superseded_document_when_other_signals_are_equal():
    chunks = [
        _make_chunk(
            "current",
            "standard return window customer return merchandise",
            status="active",
        ),
        _make_chunk(
            "legacy",
            "standard return window customer return merchandise",
            status="superseded",
        ),
    ]

    index = build_index(chunks, HashingEmbedder())

    results = index.search(
        "standard return window customer return merchandise",
        k=2,
    )

    assert results[0].chunk.chunk_id == "current"


def test_official_document_outranks_non_authoritative_document_when_other_signals_are_equal():
    chunks = [
        _make_chunk(
            "official",
            "standard return window customer return",
            policy_authority="official",
        ),
        _make_chunk(
            "unapproved",
            "standard return window customer return",
            policy_authority="none",
        ),
    ]

    index = build_index(chunks, HashingEmbedder())

    results = index.search(
        "standard return window customer return",
        k=2,
    )

    assert results[0].chunk.chunk_id == "official"


def test_customer_answering_document_outranks_internal_document_when_other_signals_are_equal():
    chunks = [
        _make_chunk(
            "customer",
            "return window customer return",
            customer_answering=True,
        ),
        _make_chunk(
            "internal",
            "return window customer return",
            customer_answering=False,
        ),
    ]

    index = build_index(chunks, HashingEmbedder())

    results = index.search(
        "return window customer return",
        k=2,
    )

    assert results[0].chunk.chunk_id == "customer"


@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed in this environment",
)
def test_real_embedder_retrieves_current_returns_policy_above_warranty():
    """
    Real-world regression test discovered during Phase 4.

    Before Phase 5, the warranty document incorrectly ranked first because
    it contains the word "backpacks".

    The correct returns-policy document should now rank first.
    """
    from app.embeddings import get_embedder

    documents = load_documents(settings.knowledge_base_dir)
    chunks = chunk_documents(documents)

    index = build_index(
        chunks,
        get_embedder(settings.embedding_model),
    )

    results = index.search(
        "How long does a regular customer have to return an unused backpack?",
        k=3,
    )

    top_sources = [
        result.chunk.source_filename
        for result in results
    ]

    assert "01-returns-policy-current.md" in top_sources
    assert top_sources[0] == "01-returns-policy-current.md"

def test_lexical_relevance_prefers_policy_terms_over_incidental_product_term():
    chunks = [
        _make_chunk(
            "warranty",
            "Aster and Row bags and backpacks are covered for two years.",
            heading="Warranty periods",
        ),
        _make_chunk(
            "returns",
            "Customers may request a return within thirty calendar days.",
            heading="Standard return window",
        ),
    ]

    index = build_index(chunks, HashingEmbedder())

    results = index.search(
        "How long does a customer have to return an unused backpack?",
        k=2,
    )

    assert results[0].chunk.chunk_id == "returns"