"""
Hybrid retrieval over knowledge-base chunks.

Phase 4:
- FAISS IndexFlatIP over normalized embeddings for dense retrieval.

Phase 5:
- BM25-style lexical retrieval to complement dense similarity.
- Metadata-based precedence for active/official/customer-facing content.

Dense retrieval finds semantically related candidates. Lexical retrieval
helps when exact policy terminology matters. Metadata precedence provides
a deterministic signal for document authority.

Conflict detection is intentionally left for Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import faiss
import numpy as np

from app.chunking import Chunk
from app.embeddings import Embedder


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0
    precedence_score: float = 0.0


class VectorIndex:
    """FAISS-backed hybrid retrieval over a fixed set of chunks."""

    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._index: faiss.Index | None = None

        # BM25 statistics.
        self._document_tokens: list[list[str]] = []
        self._document_frequencies: dict[str, int] = {}
        self._average_document_length: float = 0.0

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index over zero chunks")

        self._chunks = chunks

        # Dense index.
        vectors = self._embedder.embed([c.text for c in chunks])
        self._assert_normalized(vectors)

        index = faiss.IndexFlatIP(self._embedder.dimension)
        index.add(vectors)
        self._index = index

        # Lexical index.
        self._document_tokens = [
            self._tokenize(f"{chunk.heading} {chunk.text}")
            for chunk in chunks
        ]

        self._document_frequencies = {}

        for tokens in self._document_tokens:
            for term in set(tokens):
                self._document_frequencies[term] = (
                    self._document_frequencies.get(term, 0) + 1
                )

        total_length = sum(
            len(tokens) for tokens in self._document_tokens
        )

        self._average_document_length = (
            total_length / len(self._document_tokens)
            if self._document_tokens
            else 0.0
        )

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self._index is None:
            raise RuntimeError("VectorIndex.build() must be called before search()")

        if not query.strip():
            raise ValueError("query must not be empty")

        if k <= 0:
            raise ValueError("k must be greater than zero")

        k = min(k, len(self._chunks))

        query_tokens = self._tokenize(query)

        # ----------------------------
        # 1. Dense candidate retrieval
        # ----------------------------
        candidate_k = min(len(self._chunks), max(k * 3, 10))

        query_vector = self._embedder.embed([query])

        dense_scores, dense_indices = self._index.search(
            query_vector,
            candidate_k,
        )

        dense_candidates: set[int] = set()

        dense_score_by_index: dict[int, float] = {}

        for score, idx in zip(dense_scores[0], dense_indices[0]):
            if idx == -1:
                continue

            idx = int(idx)

            dense_candidates.add(idx)
            dense_score_by_index[idx] = float(score)

        # ----------------------------
        # 2. BM25 lexical retrieval
        # ----------------------------
        lexical_scores = self._bm25_scores(query_tokens)

        lexical_candidate_count = min(
            len(self._chunks),
            max(k * 3, 10),
        )

        lexical_order = sorted(
            range(len(self._chunks)),
            key=lambda idx: lexical_scores[idx],
            reverse=True,
        )

        lexical_candidates = set(
            lexical_order[:lexical_candidate_count]
        )

        # Take the union. This prevents a relevant lexical match from
        # disappearing simply because dense retrieval ranked it too low.
        candidate_indices = dense_candidates | lexical_candidates

        if not candidate_indices:
            return []

        # Normalize lexical scores within the candidate pool so they can
        # be combined with the approximately 0..1 dense score.
        max_lexical = max(
            lexical_scores[idx]
            for idx in candidate_indices
        )

        results: list[SearchResult] = []

        for idx in candidate_indices:
            chunk = self._chunks[idx]

            dense_score = dense_score_by_index.get(idx, 0.0)
            lexical_raw = lexical_scores[idx]

            lexical_score = (
                lexical_raw / max_lexical
                if max_lexical > 0.0
                else 0.0
            )

            precedence_score = self._precedence_score(chunk)

            # Dense retrieval remains the largest signal.
            # BM25 provides a meaningful lexical correction.
            # Metadata provides a smaller deterministic authority signal.
            final_score = (
                0.55 * dense_score
                + 0.35 * lexical_score
                + 0.10 * precedence_score
            )

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(final_score),
                    dense_score=float(dense_score),
                    lexical_score=float(lexical_score),
                    precedence_score=float(precedence_score),
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Convert text into lowercase word tokens."""
        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    def _bm25_scores(self, query_tokens: list[str]) -> list[float]:
        """
        Calculate BM25-style scores for every indexed document.

        This is intentionally implemented locally because the corpus is only
        51 chunks and we want the ranking logic to remain transparent.
        """

        if not query_tokens or not self._document_tokens:
            return [0.0] * len(self._document_tokens)

        unique_query_terms = set(query_tokens)

        document_count = len(self._document_tokens)

        k1 = 1.5
        b = 0.75

        scores = [0.0] * document_count

        for idx, document_tokens in enumerate(self._document_tokens):
            document_length = len(document_tokens)

            if document_length == 0:
                continue

            term_counts: dict[str, int] = {}

            for token in document_tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            score = 0.0

            for term in unique_query_terms:
                term_frequency = term_counts.get(term, 0)

                if term_frequency == 0:
                    continue

                document_frequency = self._document_frequencies.get(term, 0)

                # Standard BM25-style inverse-document-frequency term.
                idf = math.log(
                    1.0
                    + (
                        document_count
                        - document_frequency
                        + 0.5
                    )
                    / (
                        document_frequency
                        + 0.5
                    )
                )

                denominator = (
                    term_frequency
                    + k1
                    * (
                        1.0
                        - b
                        + b
                        * (
                            document_length
                            / self._average_document_length
                        )
                    )
                )

                score += idf * (
                    term_frequency * (k1 + 1.0)
                ) / denominator

            scores[idx] = score

        return scores

    @staticmethod
    def _precedence_score(chunk: Chunk) -> float:
        """
        Return a deterministic document-precedence score.

        This is a ranking signal only. It does not resolve genuine conflicts.
        """

        score = 0.0

        if chunk.status == "active":
            score += 0.5
        elif chunk.status == "superseded":
            score -= 0.5
        elif chunk.status == "draft":
            score -= 0.5

        if chunk.policy_authority == "official":
            score += 0.4
        elif chunk.policy_authority == "none":
            score -= 0.4

        if chunk.customer_answering:
            score += 0.1
        else:
            score -= 0.1

        return score

    @staticmethod
    def _assert_normalized(vectors: np.ndarray) -> None:
        """Verify that embeddings are unit-normalized."""
        norms = np.linalg.norm(vectors, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                "Embedder returned non-normalized vectors; "
                "IndexFlatIP requires unit-normalized embeddings "
                "to behave as cosine similarity."
            )


def build_index(
    chunks: list[Chunk],
    embedder: Embedder,
) -> VectorIndex:
    index = VectorIndex(embedder)
    index.build(chunks)
    return index