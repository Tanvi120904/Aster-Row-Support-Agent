"""
Chunking.

Turns each Document's body into a list of Chunks — the actual units we
embed, index, and retrieve.

Why chunk at all instead of embedding whole documents:
Every document here mixes several sub-topics under one file (e.g.
01-returns-policy-current.md covers window, condition, refunds, and
exclusions). If we embedded the whole file as one vector, a question about
refund timing and a question about item condition would get the same
retrieval score against that file — we'd retrieve the right document but
have no way to point the model (or the citation) at the right paragraph.
Chunking at heading level fixes both problems: better retrieval precision,
and citations that name a real heading instead of just a filename.

Why chunk by "## heading" specifically (not fixed-size windows):
We inspected section lengths across all 14 files: every "##" section is
between roughly 90 and 720 characters (a paragraph or two). That is already
a good embedding chunk size — small enough to be topically coherent, large
enough to carry real information. Splitting further (e.g. by sentence)
would fragment context for no benefit here; splitting less (whole document)
would lose precision. A fixed-size sliding window is the more common
approach for long-form prose, but it's the wrong tool for a corpus that's
already structured with meaningful headings.

Every chunk carries a copy of its parent document's precedence metadata
(status, policy_authority, supersession, etc.) so retrieval and citation
never need to look anything up elsewhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion import Document

# Splits a document body into (heading_text, section_body) pairs at each
# level-2 "## " heading. The text before the first "##" (just the H1 title)
# is discarded — it never contains citable content in this corpus.
_SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # e.g. "01-returns-policy-current.md#standard-return-window"
    text: str  # heading + section body, what actually gets embedded
    heading: str
    source_filename: str
    document_id: str
    title: str
    status: str
    policy_authority: str
    audience: str
    customer_answering: bool
    effective_date: str | None
    supersedes: str | None
    superseded_by: str | None

    def citation(self) -> str:
        """Human-readable source reference: filename + heading.

        The assignment requires every policy/product answer to name at
        least the filename and relevant heading — this is that string.
        """
        return f"{self.source_filename} — {self.heading}"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def chunk_document(document: Document) -> list[Chunk]:
    """Split one document into one Chunk per "##" section."""
    matches = list(_SECTION_RE.finditer(document.body))
    chunks: list[Chunk] = []

    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(document.body)
        section_body = document.body[start:end].strip()

        # Embed the heading together with the body: short sections like
        # "Gift cards\n\nAster & Row gift cards do not expire..." retrieve
        # better when the heading's own words are part of the embedded text.
        text = f"{heading}\n\n{section_body}".strip()

        chunks.append(
            Chunk(
                chunk_id=f"{document.source_filename}#{_slugify(heading)}",
                text=text,
                heading=heading,
                source_filename=document.source_filename,
                document_id=document.document_id,
                title=document.title,
                status=document.status,
                policy_authority=document.policy_authority,
                audience=document.audience,
                customer_answering=document.customer_answering,
                effective_date=document.effective_date,
                supersedes=document.supersedes,
                superseded_by=document.superseded_by,
            )
        )

    if not chunks:
        raise ValueError(f"{document.source_filename}: no '##' sections found to chunk")

    return chunks


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc))
    return chunks
