"""
Manual retrieval sanity-check.

Usage:
    python scripts/query_kb.py "How long does a regular customer have to return an unused backpack?"

Requires sentence-transformers to be installed (real embedding model) —
this is the one script in the project that needs the full dependency set,
since its whole purpose is to eyeball real semantic retrieval quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import chunk_documents
from app.config import settings
from app.embeddings import get_embedder
from app.ingestion import load_documents
from app.retrieval import build_index


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/query_kb.py "your question here"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    print(f"Loading knowledge base and building index (embedding model: {settings.embedding_model})...")
    documents = load_documents(settings.knowledge_base_dir)
    chunks = chunk_documents(documents)
    embedder = get_embedder(settings.embedding_model)
    index = build_index(chunks, embedder)

    print(f"\nQuery: {query}\n")
    for rank, result in enumerate(index.search(query, k=5), start=1):
        c = result.chunk
        print(f"{rank}. score={result.score:.3f}  {c.citation()}  [status={c.status}, authority={c.policy_authority}]")
        print(f"   {c.text[:140].strip()}...")
        print()


if __name__ == "__main__":
    main()
