from app.chunking import Chunk
from app.evidence import (
    analyze_evidence,
    authoritative_citations,
    build_citations,
    citation_for_chunk,
    detect_conflicts,
    is_customer_authoritative,
)
from app.retrieval import SearchResult


def _chunk(
    filename: str,
    heading: str,
    text: str,
    *,
    status: str = "active",
    authority: str = "official",
    customer_answering: bool = True,
    document_id: str | None = None,
    supersedes: list[str] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{filename}:{heading}",
        text=text,
        heading=heading,
        source_filename=filename,
        document_id=document_id or filename,
        title=heading,
        status=status,
        policy_authority=authority,
        audience="customer",
        customer_answering=customer_answering,
        effective_date=None,
        supersedes=supersedes,
        superseded_by=None,
    )


def _result(chunk: Chunk, score: float = 0.8) -> SearchResult:
    return SearchResult(
        chunk=chunk,
        score=score,
        dense_score=score,
        lexical_score=0.5,
        precedence_score=0.5,
    )


def test_citation_contains_filename_and_heading():
    chunk = _chunk(
        "01-returns-policy-current.md",
        "Standard return window",
        "Customers may return eligible items within 30 days.",
    )

    citation = citation_for_chunk(chunk)

    assert citation.source_filename == "01-returns-policy-current.md"
    assert citation.heading == "Standard return window"
    assert "01-returns-policy-current.md" in citation.citation_text
    assert "Standard return window" in citation.citation_text


def test_build_citations_deduplicates_same_source_and_heading():
    chunk = _chunk(
        "01-returns-policy-current.md",
        "Standard return window",
        "30 days.",
    )

    results = [
        _result(chunk, 0.9),
        _result(chunk, 0.8),
    ]

    citations = build_citations(results)

    assert len(citations) == 1


def test_active_official_customer_source_is_authoritative():
    chunk = _chunk(
        "01-returns-policy-current.md",
        "Standard return window",
        "30 days.",
    )

    assert is_customer_authoritative(chunk)


def test_superseded_source_is_not_customer_authoritative():
    chunk = _chunk(
        "02-returns-policy-legacy.md",
        "Return window",
        "45 days.",
        status="superseded",
    )

    assert not is_customer_authoritative(chunk)


def test_internal_source_is_not_customer_authoritative():
    chunk = _chunk(
        "14-internal-content-migration-notes.md",
        "Unapproved legacy copy",
        "60 days for everyone.",
        authority="none",
        customer_answering=False,
    )

    assert not is_customer_authoritative(chunk)


def test_authoritative_citations_exclude_internal_sources():
    customer = _chunk(
        "01-returns-policy-current.md",
        "Standard return window",
        "30 days.",
    )

    internal = _chunk(
        "14-internal-content-migration-notes.md",
        "Unapproved legacy copy",
        "60 days for everyone.",
        authority="none",
        customer_answering=False,
    )

    citations = authoritative_citations(
        [
            _result(customer),
            _result(internal),
        ]
    )

    assert len(citations) == 1
    assert citations[0].source_filename == "01-returns-policy-current.md"


def test_legacy_current_relationship_is_not_reported_as_unresolved_conflict():
    current = _chunk(
        "01-returns-policy-current.md",
        "Standard return window",
        "Customers may return items within 30 calendar days.",
        document_id="returns-current",
        supersedes=["02-returns-policy-legacy.md", "returns-legacy"],
    )

    legacy = _chunk(
        "02-returns-policy-legacy.md",
        "Return window",
        "Customers could return items within 45 calendar days.",
        status="superseded",
        document_id="returns-legacy",
    )

    conflicts = detect_conflicts(
        [
            _result(current),
            _result(legacy),
        ]
    )

    assert conflicts == []


def test_tumbler_active_official_conflict_requires_handoff():
    care = _chunk(
        "11-product-care.md",
        "Breeze Tumbler care",
        "The Breeze Tumbler body must be hand-washed.",
    )

    product_card = _chunk(
        "12-breeze-tumbler-product-card.md",
        "Dishwasher safety",
        "All Breeze Tumbler components are dishwasher-safe.",
    )

    conflicts = detect_conflicts(
        [
            _result(care, 0.9),
            _result(product_card, 0.85),
        ]
    )

    assert len(conflicts) == 1
    assert conflicts[0].source_a.source_filename != conflicts[0].source_b.source_filename
    assert "Breeze Tumbler" in conflicts[0].reason


def test_analyze_evidence_sets_handoff_for_unresolved_conflict():
    care = _chunk(
        "11-product-care.md",
        "Breeze Tumbler care",
        "The Breeze Tumbler body must be hand-washed.",
    )

    product_card = _chunk(
        "12-breeze-tumbler-product-card.md",
        "Dishwasher safety",
        "All Breeze Tumbler components are dishwasher-safe.",
    )

    evidence = analyze_evidence(
        [
            _result(care),
            _result(product_card),
        ]
    )

    assert len(evidence.sources) == 2
    assert len(evidence.authoritative_sources) == 2
    assert len(evidence.conflicts) == 1
    assert evidence.handoff is True


def test_analyze_evidence_has_no_handoff_for_single_authoritative_source():
    chunk = _chunk(
        "01-returns-policy-current.md",
        "Standard return window",
        "Customers may return items within 30 days.",
    )

    evidence = analyze_evidence([_result(chunk)])

    assert len(evidence.sources) == 1
    assert len(evidence.authoritative_sources) == 1
    assert evidence.conflicts == []
    assert evidence.handoff is False