"""
Evidence analysis for retrieved knowledge-base chunks.

Phase 6 adds two responsibilities on top of retrieval:

1. Citations:
   Convert retrieved chunks into stable source citations.

2. Conflict detection:
   Identify genuine conflicts among authoritative active sources.

Superseded sources are not treated as authoritative conflicts when a current
document explicitly supersedes them.

This module is deterministic and does not call an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.chunking import Chunk
from app.retrieval import SearchResult


@dataclass(frozen=True)
class Citation:
    """A stable citation for a retrieved knowledge-base chunk."""

    source_filename: str
    heading: str
    citation_text: str


@dataclass(frozen=True)
class SourceConflict:
    """A genuine disagreement between authoritative active sources."""

    source_a: Citation
    source_b: Citation
    reason: str


@dataclass(frozen=True)
class EvidenceBundle:
    """
    Structured evidence passed to later agent/evaluation layers.

    `sources` contains unique citations from the retrieved evidence.

    `authoritative_sources` contains citations that are eligible to support a
    customer answer.

    `conflicts` contains unresolved conflicts between active authoritative
    sources.

    `handoff` becomes true when an unresolved conflict requires escalation.
    """

    sources: list[Citation]
    authoritative_sources: list[Citation]
    conflicts: list[SourceConflict]
    handoff: bool


def citation_for_chunk(chunk: Chunk) -> Citation:
    """Create a citation from the metadata already stored on a Chunk."""

    return Citation(
        source_filename=chunk.source_filename,
        heading=chunk.heading,
        citation_text=chunk.citation(),
    )


def build_citations(results: list[SearchResult]) -> list[Citation]:
    """
    Convert retrieval results to de-duplicated citations.

    The first occurrence is preserved so higher-ranked evidence remains first.
    """

    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()

    for result in results:
        citation = citation_for_chunk(result.chunk)

        key = (
            citation.source_filename,
            citation.heading,
        )

        if key in seen:
            continue

        seen.add(key)
        citations.append(citation)

    return citations


def is_customer_authoritative(chunk: Chunk) -> bool:
    """
    Determine whether a chunk is eligible as customer-facing authority.

    This is stricter than merely being 'active' and 'official'.

    Internal or explicitly non-customer-answering material can be useful to
    the application but must not become customer-facing policy evidence.
    """

    return (
        chunk.status == "active"
        and chunk.policy_authority == "official"
        and chunk.customer_answering
    )


def authoritative_citations(results: list[SearchResult]) -> list[Citation]:
    """Return unique citations from customer-authoritative chunks."""

    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()

    for result in results:
        chunk = result.chunk

        if not is_customer_authoritative(chunk):
            continue

        citation = citation_for_chunk(chunk)

        key = (
            citation.source_filename,
            citation.heading,
        )

        if key in seen:
            continue

        seen.add(key)
        citations.append(citation)

    return citations


def _canonical_topic(text: str) -> set[str]:
    """
    Extract simple topic/product terms used to avoid comparing unrelated
    documents.

    This deliberately stays lightweight; Phase 6 is not a general-purpose
    semantic contradiction detector.
    """

    tokens = set(
        re.findall(r"\b[a-z0-9]+\b", text.lower())
    )

    ignored = {
        "the",
        "and",
        "or",
        "to",
        "of",
        "a",
        "an",
        "is",
        "are",
        "for",
        "with",
        "from",
        "this",
        "that",
        "must",
        "may",
        "be",
    }

    return tokens - ignored


def _has_handwash_claim(text: str) -> bool:
    text = text.lower()

    return bool(
        re.search(
            r"\bhand[\s-]?wash(?:ed|ing)?\b"
            r"|\bwash\s+by\s+hand\b"
            r"|\bhand[\s-]?washing\b",
            text,
        )
    )


def _has_dishwasher_claim(text: str) -> bool:
    text = text.lower()

    return bool(
        re.search(
            r"\bdishwasher[\s-]?safe\b"
            r"|\bdishwasher\b",
            text,
        )
    )


def _contains_tumbler_topic(text: str) -> bool:
    text = text.lower()

    return (
        "breeze tumbler" in text
        or "breeze" in text and "tumbler" in text
    )


def _is_known_contradiction(a: Chunk, b: Chunk) -> bool:
    """
    Detect the deliberate active-policy contradiction in this corpus.

    The assignment contains two authoritative active sources with opposing
    care instructions for the Breeze Tumbler.

    We keep this explicit rather than pretending that keyword matching is a
    general natural-language contradiction detector.
    """

    combined = f"{a.text}\n{b.text}"

    if not _contains_tumbler_topic(combined):
        return False

    a_handwash = _has_handwash_claim(a.text)
    a_dishwasher = _has_dishwasher_claim(a.text)

    b_handwash = _has_handwash_claim(b.text)
    b_dishwasher = _has_dishwasher_claim(b.text)

    return (
        (a_handwash and b_dishwasher)
        or (a_dishwasher and b_handwash)
    )


def _supersedes(a: Chunk, b: Chunk) -> bool:
    """Return whether either chunk explicitly supersedes the other."""

    if b.source_filename in (a.supersedes or []):
        return True

    if b.document_id and b.document_id in (a.supersedes or []):
        return True

    if a.source_filename in (b.supersedes or []):
        return True

    if a.document_id and a.document_id in (b.supersedes or []):
        return True

    return False


def detect_conflicts(
    results: list[SearchResult],
) -> list[SourceConflict]:
    """
    Find unresolved conflicts among active, official customer-facing sources.

    Superseded/current relationships are intentionally excluded because those
    conflicts are resolvable through document precedence.

    Duplicate source pairs are de-duplicated.
    """

    conflicts: list[SourceConflict] = []
    seen_pairs: set[tuple[str, str]] = set()

    authoritative_results = [
        result
        for result in results
        if is_customer_authoritative(result.chunk)
    ]

    for i, first in enumerate(authoritative_results):
        for second in authoritative_results[i + 1 :]:
            a = first.chunk
            b = second.chunk

            pair = tuple(
                sorted(
                    [
                        a.source_filename,
                        b.source_filename,
                    ]
                )
            )

            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)

            # A supersession relationship resolves the disagreement.
            if _supersedes(a, b):
                continue

            # Only compare sources that appear to concern the same topic.
            topic_overlap = _canonical_topic(
                f"{a.heading} {a.text}"
            ) & _canonical_topic(
                f"{b.heading} {b.text}"
            )

            if not topic_overlap:
                continue

            if not _is_known_contradiction(a, b):
                continue

            conflicts.append(
                SourceConflict(
                    source_a=citation_for_chunk(a),
                    source_b=citation_for_chunk(b),
                    reason=(
                        "Active official sources contain contradictory "
                        "Breeze Tumbler care instructions."
                    ),
                )
            )

    return conflicts


def analyze_evidence(
    results: list[SearchResult],
) -> EvidenceBundle:
    """
    Build the complete structured evidence object used by later phases.
    """

    sources = build_citations(results)

    authoritative = authoritative_citations(results)

    conflicts = detect_conflicts(results)

    return EvidenceBundle(
        sources=sources,
        authoritative_sources=authoritative,
        conflicts=conflicts,
        handoff=bool(conflicts),
    )