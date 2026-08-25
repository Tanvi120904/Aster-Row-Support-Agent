"""
Prints a summary table of every ingested knowledge-base document, so we can
manually verify ingestion matches what we found by hand in Phase 0 —
especially the supersession chain and the internal/draft document.

Usage:
    python scripts/inspect_kb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion import load_documents


def main() -> None:
    docs = load_documents(settings.knowledge_base_dir)

    header = f"{'file':<38} {'doc_id':<14} {'status':<11} {'authority':<9} {'audience':<9} {'supersedes/by'}"
    print(header)
    print("-" * len(header))

    for d in docs:
        chain = ""
        if d.supersedes:
            chain = f"supersedes {d.supersedes}"
        elif d.superseded_by:
            chain = f"superseded_by {d.superseded_by}"
        print(
            f"{d.source_filename:<38} {d.document_id:<14} {d.status:<11} "
            f"{d.policy_authority:<9} {d.audience:<9} {chain}"
        )

    print()
    print(f"Total documents: {len(docs)}")
    authoritative = [d for d in docs if d.is_authoritative_for_customers()]
    print(f"Authoritative for customer answers: {len(authoritative)}")
    not_authoritative = [d for d in docs if not d.is_authoritative_for_customers()]
    print("Excluded from customer authority:")
    for d in not_authoritative:
        print(f"  - {d.source_filename} (status={d.status}, audience={d.audience})")


if __name__ == "__main__":
    main()
