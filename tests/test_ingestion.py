"""
Phase 2 tests.

These pin down the exact facts about the corpus that later phases (chunking,
retrieval precedence, citation) will rely on. If someone edits a knowledge-base
file's front matter later, these tests should fail loudly rather than let a
precedence bug slip in silently.
"""

import pytest

from app.config import settings
from app.ingestion import load_documents


@pytest.fixture(scope="module")
def documents():
    return load_documents(settings.knowledge_base_dir)


def by_filename(documents, filename):
    return next(d for d in documents if d.source_filename == filename)


def test_loads_all_fourteen_documents(documents):
    assert len(documents) == 14


def test_current_returns_policy_is_active_and_official(documents):
    d = by_filename(documents, "01-returns-policy-current.md")
    assert d.status == "active"
    assert d.policy_authority == "official"
    assert d.document_id == "RET-2026-01"
    assert d.supersedes == "RET-2024-01"
    assert d.is_authoritative_for_customers() is True


def test_legacy_returns_policy_is_superseded(documents):
    d = by_filename(documents, "02-returns-policy-legacy.md")
    assert d.status == "superseded"
    assert d.superseded_by == "RET-2026-01"
    # Superseded docs must never be treated as citable authority.
    assert d.is_authoritative_for_customers() is False


def test_internal_migration_notes_are_excluded_from_authority(documents):
    d = by_filename(documents, "14-internal-content-migration-notes.md")
    assert d.status == "draft"
    assert d.policy_authority == "none"
    assert d.audience == "internal"
    assert d.customer_answering is False
    assert d.is_authoritative_for_customers() is False
    # Sanity check the injection payload is present in the body (as data,
    # not as something we execute) — Phase 10 will test that the agent
    # never follows it.
    assert "SYSTEM INSTRUCTION" in d.body


def test_customer_answering_defaults_true_when_unspecified(documents):
    # Only doc 14 sets customer_answering explicitly. Every other document
    # should default to True.
    d = by_filename(documents, "01-returns-policy-current.md")
    assert d.customer_answering is True


def test_headings_extracted_for_citation(documents):
    d = by_filename(documents, "01-returns-policy-current.md")
    heading_texts = [h.text for h in d.headings]
    assert "Standard return window" in heading_texts
    assert "Return shipping and refunds" in heading_texts


def test_two_active_official_conflicting_tumbler_docs_both_present(documents):
    # This is the genuine, unresolvable conflict from Phase 0 — confirm
    # ingestion doesn't accidentally drop or misclassify either side.
    care = by_filename(documents, "11-product-care.md")
    product_card = by_filename(documents, "12-breeze-tumbler-product-card.md")
    assert care.is_authoritative_for_customers() is True
    assert product_card.is_authoritative_for_customers() is True
    assert care.supersedes is None and care.superseded_by is None
    assert product_card.supersedes is None and product_card.superseded_by is None
