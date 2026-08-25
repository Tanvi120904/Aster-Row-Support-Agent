import pytest

from app.chunking import chunk_document, chunk_documents
from app.config import settings
from app.ingestion import Document, Heading, load_documents


@pytest.fixture(scope="module")
def documents():
    return load_documents(settings.knowledge_base_dir)


@pytest.fixture(scope="module")
def chunks(documents):
    return chunk_documents(documents)


def test_chunk_count_is_stable(chunks):
    # Pinned exact count: if someone edits knowledge-base headings, this
    # should fail loudly rather than silently changing retrieval behavior.
    assert len(chunks) == 51


def test_chunk_ids_are_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_inherits_parent_document_metadata(chunks):
    legacy_chunks = [c for c in chunks if c.source_filename == "02-returns-policy-legacy.md"]
    assert legacy_chunks
    for c in legacy_chunks:
        assert c.status == "superseded"
        assert c.document_id == "RET-2024-01"


def test_chunk_text_contains_expected_content(chunks):
    window_chunk = next(
        c
        for c in chunks
        if c.source_filename == "01-returns-policy-current.md" and c.heading == "Standard return window"
    )
    assert "30 calendar days" in window_chunk.text


def test_citation_includes_filename_and_heading(chunks):
    window_chunk = next(
        c
        for c in chunks
        if c.source_filename == "01-returns-policy-current.md" and c.heading == "Standard return window"
    )
    citation = window_chunk.citation()
    assert "01-returns-policy-current.md" in citation
    assert "Standard return window" in citation


def test_document_with_no_sections_raises():
    doc = Document(
        source_filename="fake.md",
        document_id="FAKE-1",
        title="Fake",
        status="active",
        policy_authority="official",
        audience="customer",
        customer_answering=True,
        effective_date=None,
        last_reviewed=None,
        supersedes=None,
        superseded_by=None,
        superseded_date=None,
        body="# Just a title, no ## sections at all",
        headings=(Heading(level=1, text="Just a title, no ## sections at all"),),
    )
    with pytest.raises(ValueError):
        chunk_document(doc)
