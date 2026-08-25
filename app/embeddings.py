"""
Embeddings.

What an embedding is, briefly: a fixed-length vector of numbers that
represents the *meaning* of a piece of text, produced by a model trained so
that semantically similar text ends up as nearby vectors (by cosine
similarity / dot product). "Do you ship to Canada?" and "Is international
shipping to Canada available?" should land close together even though they
share almost no exact words — that's what makes semantic search possible,
as opposed to plain keyword matching.

Why this module defines an Embedder Protocol instead of calling
sentence-transformers directly everywhere:
Retrieval code (Phase 4) should not care *how* a vector was produced, only
that it can call `.embed(texts)` and get back normalized float32 vectors.
That lets us swap in a different embedder later without touching retrieval
code, and — practically, for this project — it lets us unit-test chunking
and FAISS indexing in an environment where installing the real model isn't
possible.

Two implementations live here:

1. SentenceTransformerEmbedder — the real one. Uses a local model
   (default: all-MiniLM-L6-v2, ~90MB, runs on CPU) via the
   sentence-transformers library. No API key or network call per request;
   the model is downloaded once and cached locally. This is what the
   running application uses.

2. HashingEmbedder — a deterministic, dependency-free stand-in used only
   in tests. It is NOT semantically meaningful in the way a trained model
   is (it's a bag-of-words hash, not learned), but it is good enough to
   verify that chunking, indexing, and top-k retrieval mechanics work
   correctly, without requiring a multi-hundred-MB model download in CI
   or in a disk-constrained environment.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dimension) float32 array of unit-normalized vectors."""
        ...


class SentenceTransformerEmbedder:
    """Real embedder used by the running application."""

    def __init__(self, model_name: str):
        # Imported lazily so this module can be imported (and HashingEmbedder
        # used) even in environments where sentence-transformers/torch
        # aren't installed.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,  # unit vectors -> cosine similarity via dot product
            convert_to_numpy=True,
        )
        return vectors.astype("float32")


class HashingEmbedder:
    """Deterministic, dependency-free fake embedder. Tests only — see module docstring."""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _vector_for(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype="float32")
        for token in text.lower().split():
            # md5 (not Python's built-in hash()) so the bucket a word lands
            # in is stable across processes/runs — built-in hash() is
            # randomized per-process unless PYTHONHASHSEED is fixed.
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vector_for(t) for t in texts])


def get_embedder(model_name: str) -> Embedder:
    """Factory used by the real application (scripts, Flask app)."""
    return SentenceTransformerEmbedder(model_name)
