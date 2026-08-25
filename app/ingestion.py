"""
Knowledge-base ingestion.

Turns each knowledge-base/*.md file into a Document: the YAML front-matter
metadata plus the markdown body, kept as separate fields.

Why we don't just embed the raw file text:
Two files can be about the exact same topic (e.g. the 30-day vs 45-day
return window) and be highly similar in embedding space, but have totally
different authority — one is `status: active`, the other `status:
superseded`. That distinction only exists in the front matter. If we threw
the metadata away here, no later phase could ever recover it.

Front matter format (all 14 supplied files use this):

    ---
    document_id: RET-2026-01
    title: Returns Policy
    status: active
    ...
    ---
    # Returns Policy
    ## Standard return window
    ...
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Matches a leading "---\n<yaml>\n---\n" block at the very start of the file.
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)

# Matches markdown ATX headings, e.g. "## Standard return window".
# We capture the level (number of #) and the heading text separately.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# Fields we expect on every document. Missing ones are logged as warnings
# rather than raising, so one malformed file doesn't take down ingestion —
# but a warning is still visible, which matters for a real content corpus.
_REQUIRED_FIELDS = ("document_id", "title", "status", "policy_authority")


@dataclass(frozen=True)
class Heading:
    level: int
    text: str


@dataclass(frozen=True)
class Document:
    # Identity / provenance
    source_filename: str  # e.g. "01-returns-policy-current.md" — used in citations
    document_id: str
    title: str

    # Precedence metadata — this is what lets us rank/filter beyond similarity
    status: str  # "active" | "superseded" | "draft"
    policy_authority: str  # "official" | "none"
    audience: str  # "customer" | "internal"
    customer_answering: bool  # explicit opt-out flag (doc 14 sets this False)

    # Optional dates / supersession chain
    effective_date: str | None
    last_reviewed: str | None
    supersedes: str | None
    superseded_by: str | None
    superseded_date: str | None

    # Content
    body: str
    headings: tuple[Heading, ...]

    def is_authoritative_for_customers(self) -> bool:
        """A single, reusable definition of "safe to cite as policy".

        Used by retrieval (Phase 5) so the rule lives in exactly one place
        instead of being re-implemented (and possibly re-implemented
        inconsistently) wherever we rank or filter documents.
        """
        return (
            self.status == "active"
            and self.policy_authority == "official"
            and self.customer_answering
        )


def _parse_front_matter(raw_text: str, source_filename: str) -> tuple[dict, str]:
    match = _FRONT_MATTER_RE.match(raw_text)
    if not match:
        raise ValueError(
            f"{source_filename}: expected a leading '---' YAML front-matter block, found none"
        )
    yaml_block, body = match.group(1), match.group(2)
    metadata = yaml.safe_load(yaml_block) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{source_filename}: front matter did not parse to a mapping")
    return metadata, body


def _extract_headings(body: str) -> tuple[Heading, ...]:
    return tuple(
        Heading(level=len(hashes), text=text.strip())
        for hashes, text in _HEADING_RE.findall(body)
    )


def _document_from_file(path: Path) -> Document:
    raw_text = path.read_text(encoding="utf-8")
    metadata, body = _parse_front_matter(raw_text, path.name)

    missing = [f for f in _REQUIRED_FIELDS if f not in metadata]
    if missing:
        logger.warning("%s is missing expected front-matter fields: %s", path.name, missing)

    return Document(
        source_filename=path.name,
        document_id=metadata.get("document_id", ""),
        title=metadata.get("title", path.stem),
        status=metadata.get("status", "unknown"),
        policy_authority=metadata.get("policy_authority", "none"),
        audience=metadata.get("audience", "customer"),
        # Only doc 14 sets this explicitly; everything else defaults to
        # answerable, which matches the supplied corpus.
        customer_answering=bool(metadata.get("customer_answering", True)),
        effective_date=metadata.get("effective_date"),
        last_reviewed=metadata.get("last_reviewed"),
        supersedes=metadata.get("supersedes"),
        superseded_by=metadata.get("superseded_by"),
        superseded_date=metadata.get("superseded_date"),
        body=body.strip(),
        headings=_extract_headings(body),
    )


def load_documents(knowledge_base_dir: Path) -> list[Document]:
    """Load and parse every .md file in knowledge_base_dir.

    Returns documents sorted by filename so ingestion order is deterministic
    (helpful for reproducible logs and tests).
    """
    paths = sorted(knowledge_base_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md files found in {knowledge_base_dir}")

    documents = [_document_from_file(p) for p in paths]
    logger.info("Ingested %d documents from %s", len(documents), knowledge_base_dir)
    return documents
