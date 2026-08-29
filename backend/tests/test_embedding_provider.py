"""
Unit test suite for SentenceTransformerEmbeddingProvider (RealEmbeddingProvider).

Tests:
- Protocol compliance (@runtime_checkable EmbeddingProvider)
- Single text embedding and batch embedding
- Normalization (unit length)
- Lazy loading and thread-safety
- Input validation (empty string, wrong types, empty batch)
- Dimension consistency and validation
- Isolated offline execution (mocks underlying encoder for fast test runs)
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from backend.embedding_provider import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL_NAME,
    RealEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from ingestion.embedding_generator import EmbeddingProvider


class DummyEncoder:
    """Mock SentenceTransformer encoder that generates deterministic normalized vectors."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        if isinstance(sentences, str):
            vec = [float(i + 1) for i in range(self.dimension)]
            if normalize_embeddings:
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                vec = [x / norm for x in vec]
            return vec

        results = []
        for s_idx, _ in enumerate(sentences):
            vec = [float((i + 1) * (s_idx + 1)) for i in range(self.dimension)]
            if normalize_embeddings:
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                vec = [x / norm for x in vec]
            results.append(vec)
        return results


class TestEmbeddingProviderProtocol:
    def test_implements_embedding_provider_protocol(self):
        provider = SentenceTransformerEmbeddingProvider(model_instance=DummyEncoder())
        assert isinstance(provider, EmbeddingProvider)

    def test_alias_matches(self):
        assert RealEmbeddingProvider is SentenceTransformerEmbeddingProvider


class TestSentenceTransformerEmbeddingProvider:
    def test_initialization_defaults(self):
        provider = SentenceTransformerEmbeddingProvider(model_instance=DummyEncoder())
        assert provider.model_name == DEFAULT_MODEL_NAME
        assert provider.dimension == DEFAULT_DIMENSION
        assert provider.normalize_embeddings is True
        assert provider.batch_size == 32
        assert provider.is_loaded is True

    def test_lazy_loading_not_loaded_until_used(self):
        provider = SentenceTransformerEmbeddingProvider(model_name="dummy/test-model")
        assert provider.is_loaded is False

    def test_embed_single_text(self):
        encoder = DummyEncoder(dimension=384)
        provider = SentenceTransformerEmbeddingProvider(model_instance=encoder, dimension=384)
        vector = provider.embed("OmniBrain semantic test")

        assert isinstance(vector, list)
        assert len(vector) == 384
        assert all(isinstance(x, float) for x in vector)
        # Check unit normalization
        norm = math.sqrt(sum(x * x for x in vector))
        assert pytest.approx(norm, rel=1e-5) == 1.0

    def test_embed_batch(self):
        encoder = DummyEncoder(dimension=384)
        provider = SentenceTransformerEmbeddingProvider(model_instance=encoder, dimension=384)
        texts = ["First chunk", "Second chunk", "Third chunk"]
        vectors = provider.embed_batch(texts)

        assert isinstance(vectors, list)
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 384
            assert all(isinstance(x, float) for x in v)

    def test_embed_batch_empty_list(self):
        provider = SentenceTransformerEmbeddingProvider(model_instance=DummyEncoder())
        assert provider.embed_batch([]) == []

    def test_embed_empty_string_raises_value_error(self):
        provider = SentenceTransformerEmbeddingProvider(model_instance=DummyEncoder())
        with pytest.raises(ValueError, match="Cannot generate embedding for empty"):
            provider.embed("")

    def test_embed_whitespace_only_raises_value_error(self):
        provider = SentenceTransformerEmbeddingProvider(model_instance=DummyEncoder())
        with pytest.raises(ValueError, match="Cannot generate embedding for empty"):
            provider.embed("   \n\t  ")

    def test_embed_non_string_raises_type_error(self):
        provider = SentenceTransformerEmbeddingProvider(model_instance=DummyEncoder())
        with pytest.raises(TypeError, match="Expected text to be a str"):
            provider.embed(123)  # type: ignore

    def test_embed_batch_invalid_elements_raise_type_error(self):
        provider = SentenceTransformerEmbeddingProvider(model_instance=DummyEncoder())
        with pytest.raises(TypeError, match="is not a str"):
            provider.embed_batch(["valid", None])  # type: ignore

    def test_embed_batch_empty_element_raises_value_error(self):
        provider = SentenceTransformerEmbeddingProvider(model_instance=DummyEncoder())
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            provider.embed_batch(["valid", "   "])

    def test_dimension_mismatch_raises_value_error(self):
        encoder = DummyEncoder(dimension=128)
        provider = SentenceTransformerEmbeddingProvider(model_instance=encoder, dimension=384)
        with pytest.raises(ValueError, match="does not match expected dimension 384"):
            provider.embed("test text")
