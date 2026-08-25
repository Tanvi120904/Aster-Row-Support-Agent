import numpy as np

from app.embeddings import HashingEmbedder


def test_embed_returns_correct_shape():
    embedder = HashingEmbedder(dimension=128)
    vectors = embedder.embed(["hello world", "another sentence"])
    assert vectors.shape == (2, 128)
    assert vectors.dtype == np.float32


def test_embeddings_are_unit_normalized():
    embedder = HashingEmbedder()
    vectors = embedder.embed(["ship to canada", "return window policy"])
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_embedding_is_deterministic():
    embedder = HashingEmbedder()
    v1 = embedder.embed(["do you ship to canada"])[0]
    v2 = embedder.embed(["do you ship to canada"])[0]
    np.testing.assert_array_equal(v1, v2)


def test_similar_text_scores_higher_than_dissimilar_text():
    # This is the property retrieval actually depends on: shared vocabulary
    # should push cosine similarity up. HashingEmbedder is not a real
    # semantic model, but it must satisfy this basic property or FAISS
    # retrieval tests built on top of it (Phase 4) would be meaningless.
    embedder = HashingEmbedder()
    query = embedder.embed(["does aster and row ship to canada"])[0]
    related = embedder.embed(["aster and row ships to canada"])[0]
    unrelated = embedder.embed(["gift cards never expire"])[0]

    sim_related = float(np.dot(query, related))
    sim_unrelated = float(np.dot(query, unrelated))

    assert sim_related > sim_unrelated
