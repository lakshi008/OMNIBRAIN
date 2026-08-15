"""
Tests for the embedding generator module.

Uses a deterministic test provider (pure math, no external API, no random values)
to thoroughly verify vector generation, dimensional consistency, lineage preservation,
deterministic ordering, and error handling.
"""

from __future__ import annotations

import math
import uuid
import pytest

from ingestion.embedding_generator import EmbeddingProvider, generate_embeddings
from ingestion.models import (
    EmbeddingGenerationResult,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    EmbeddingVectorRecord,
)


# ── Deterministic Test Provider ──────────────────────────────────────────


class DeterministicTestEmbeddingProvider:
    """Deterministic embedding provider for testing without external calls."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector: list[float] = []
        for i in range(self.dimension):
            val = (sum((ord(c) * (i + 1)) for c in text) % 1000) / 1000.0
            vector.append(round(val, 4))
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class SingleOnlyEmbeddingProvider:
    """Provider implementing only embed() without embed_batch()."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        return [float(len(text) + i) for i in range(self.dimension)]


class InconsistentDimensionProvider:
    """Provider returning vectors with varying dimensions to test validation."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0] if i == 0 else [1.0, 2.0] for i in range(len(texts))]


class InvalidValueProvider:
    """Provider returning invalid/non-numeric vector contents."""

    def __init__(self, mode: str = "string") -> None:
        self.mode = mode

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.mode == "string":
            return [["0.1", "0.2"]]  # type: ignore
        elif self.mode == "nan":
            return [[float("nan"), 0.5]]
        elif self.mode == "empty":
            return [[]]
        elif self.mode == "non_list":
            return "not_a_list"  # type: ignore
        return [[0.1, 0.2]]


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_doc_id() -> str:
    """Create sample document UUID."""
    return str(uuid.uuid4())


@pytest.fixture
def provider() -> DeterministicTestEmbeddingProvider:
    """Create standard 8-dimensional test provider."""
    return DeterministicTestEmbeddingProvider(dimension=8)


@pytest.fixture
def sample_records(sample_doc_id: str) -> list[EmbeddingRecord]:
    """Create sample list of mixed EmbeddingRecord instances."""
    return [
        EmbeddingRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=sample_doc_id,
            filename="report.pdf",
            chunk_index=0,
            page_number=1,
            content="OmniBrain architecture overview and supervisor design.",
            content_type="text",
            metadata={"char_count": 55, "section": "intro"},
        ),
        EmbeddingRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=sample_doc_id,
            filename="report.pdf",
            chunk_index=1,
            page_number=1,
            content="| Model | Accuracy |\n| --- | --- |\n| Agent-1 | 95% |",
            content_type="table",
            metadata={"table_index": 0, "rows": 2, "columns": 2},
        ),
        EmbeddingRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=sample_doc_id,
            filename="report.pdf",
            chunk_index=2,
            page_number=2,
            content="[Image on Page 2 (Index 0): format=PNG, width=640px, height=480px]",
            content_type="image",
            metadata={"image_index": 0, "image_format": "png", "width": 640, "height": 480},
        ),
    ]


@pytest.fixture
def sample_prep_result(sample_records: list[EmbeddingRecord], sample_doc_id: str) -> EmbeddingPreparationResult:
    """Create an EmbeddingPreparationResult fixture."""
    return EmbeddingPreparationResult(
        document_id=sample_doc_id,
        filename="report.pdf",
        items=sample_records,
        is_ready=True,
    )


# ── Success Tests ────────────────────────────────────────────────────────


class TestEmbeddingGeneratorSuccess:
    """Tests for successful embedding generation."""

    def test_returns_embedding_generation_result(
        self, sample_records: list[EmbeddingRecord], provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """generate_embeddings returns EmbeddingGenerationResult with vector records."""
        result = generate_embeddings(sample_records, provider)
        assert isinstance(result, EmbeddingGenerationResult)
        assert result.is_ready is True
        assert result.dimension == 8
        assert result.total_items == 3
        for item in result.items:
            assert isinstance(item, EmbeddingVectorRecord)
            assert len(item.vector) == 8
            assert all(isinstance(x, float) for x in item.vector)

    def test_single_item(
        self, sample_records: list[EmbeddingRecord], provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Works for a single EmbeddingRecord."""
        result = generate_embeddings([sample_records[0]], provider)
        assert result.total_items == 1
        assert result.dimension == 8
        assert len(result.items[0].vector) == 8

    def test_accepts_embedding_preparation_result(
        self, sample_prep_result: EmbeddingPreparationResult, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Accepts EmbeddingPreparationResult directly."""
        result = generate_embeddings(sample_prep_result, provider)
        assert result.is_ready is True
        assert result.total_items == 3
        assert result.document_id == sample_prep_result.document_id
        assert result.filename == sample_prep_result.filename

    def test_empty_input_handled_safely(
        self, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Empty input returns result with 0 items, dimension 0, and is_ready=True."""
        result = generate_embeddings([], provider)
        assert result.is_ready is True
        assert result.total_items == 0
        assert result.dimension == 0
        assert result.items == []

    def test_provider_with_only_embed_method(
        self, sample_records: list[EmbeddingRecord]
    ) -> None:
        """Works with providers implementing only embed() without embed_batch()."""
        single_provider = SingleOnlyEmbeddingProvider(dimension=4)
        result = generate_embeddings(sample_records, single_provider)
        assert result.dimension == 4
        assert result.total_items == 3
        assert len(result.items[0].vector) == 4


# ── Lineage & Attribute Preservation Tests ───────────────────────────────


class TestLineageAndAttributes:
    """Tests verifying lineage and citation metadata preservation in vector records."""

    def test_preserves_all_chunk_attributes(
        self, sample_records: list[EmbeddingRecord], provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Preserves chunk_id, document_id, filename, chunk_index, page_number, content_type."""
        result = generate_embeddings(sample_records, provider)
        for orig, vec_rec in zip(sample_records, result.items):
            assert vec_rec.chunk_id == orig.chunk_id
            assert vec_rec.document_id == orig.document_id
            assert vec_rec.filename == orig.filename
            assert vec_rec.chunk_index == orig.chunk_index
            assert vec_rec.page_number == orig.page_number
            assert vec_rec.content_type == orig.content_type
            assert vec_rec.metadata == orig.metadata

    def test_preserves_mixed_content_types(
        self, sample_records: list[EmbeddingRecord], provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Preserves text, table, and image modalities correctly."""
        result = generate_embeddings(sample_records, provider)
        assert result.text_items == 1
        assert result.table_items == 1
        assert result.image_items == 1


# ── Ordering & Determinism Tests ─────────────────────────────────────────


class TestOrderingAndDeterminism:
    """Tests for deterministic ordering and output."""

    def test_shuffled_input_sorted_by_chunk_index(
        self, sample_records: list[EmbeddingRecord], provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Shuffled input records are sorted deterministically by chunk_index."""
        shuffled = list(reversed(sample_records))
        result = generate_embeddings(shuffled, provider)
        indices = [r.chunk_index for r in result.items]
        assert indices == [0, 1, 2]

    def test_deterministic_vectors(
        self, sample_records: list[EmbeddingRecord], provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Repeated generation produces identical vector values."""
        res1 = generate_embeddings(sample_records, provider)
        res2 = generate_embeddings(sample_records, provider)
        assert res1.items[0].vector == res2.items[0].vector
        assert res1.items[1].vector == res2.items[1].vector


# ── Counters & Filtering Helpers Tests ───────────────────────────────────


class TestCountersAndHelpers:
    """Tests for result counter properties and filtering helpers."""

    def test_counters(
        self, sample_records: list[EmbeddingRecord], provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Check total, text, table, and image counters."""
        result = generate_embeddings(sample_records, provider)
        assert result.total_items == 3
        assert result.text_items == 1
        assert result.table_items == 1
        assert result.image_items == 1

    def test_get_vectors_by_type(
        self, sample_records: list[EmbeddingRecord], provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Filter vector records by content type."""
        result = generate_embeddings(sample_records, provider)
        assert len(result.get_vectors_by_type("text")) == 1
        assert len(result.get_vectors_by_type("table")) == 1
        assert len(result.get_vectors_by_type("image")) == 1
        assert len(result.get_vectors_by_type("audio")) == 0

    def test_get_vectors_on_page(
        self, sample_records: list[EmbeddingRecord], provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Filter vector records by page number."""
        result = generate_embeddings(sample_records, provider)
        assert len(result.get_vectors_on_page(1)) == 2
        assert len(result.get_vectors_on_page(2)) == 1
        assert len(result.get_vectors_on_page(99)) == 0


# ── Error Handling Tests ─────────────────────────────────────────────────


class TestErrorHandling:
    """Tests for rejection of invalid inputs and provider anomalies."""

    def test_inconsistent_dimensions_raises_value_error(
        self, sample_records: list[EmbeddingRecord]
    ) -> None:
        """Inconsistent dimensions across batch raises ValueError."""
        bad_provider = InconsistentDimensionProvider()
        with pytest.raises(ValueError, match="Inconsistent vector dimension"):
            generate_embeddings(sample_records, bad_provider)

    def test_empty_vector_raises_value_error(
        self, sample_records: list[EmbeddingRecord]
    ) -> None:
        """Empty vector output raises ValueError."""
        bad_provider = InvalidValueProvider(mode="empty")
        with pytest.raises(ValueError, match="empty or non-list vector"):
            generate_embeddings([sample_records[0]], bad_provider)

    def test_nan_vector_raises_value_error(
        self, sample_records: list[EmbeddingRecord]
    ) -> None:
        """NaN values in vector raise ValueError."""
        bad_provider = InvalidValueProvider(mode="nan")
        with pytest.raises(ValueError, match="invalid value"):
            generate_embeddings([sample_records[0]], bad_provider)

    def test_non_numeric_vector_raises_value_error(
        self, sample_records: list[EmbeddingRecord]
    ) -> None:
        """String values in vector raise ValueError."""
        bad_provider = InvalidValueProvider(mode="string")
        with pytest.raises(ValueError, match="invalid value"):
            generate_embeddings([sample_records[0]], bad_provider)

    def test_non_list_provider_output_raises_value_error(
        self, sample_records: list[EmbeddingRecord]
    ) -> None:
        """Non-list provider return raises ValueError."""
        bad_provider = InvalidValueProvider(mode="non_list")
        with pytest.raises(ValueError, match="instead of list"):
            generate_embeddings([sample_records[0]], bad_provider)

    def test_invalid_provider_raises_type_error(
        self, sample_records: list[EmbeddingRecord]
    ) -> None:
        """Provider without embed/embed_batch raises TypeError."""
        with pytest.raises(TypeError, match="Invalid embedding provider"):
            generate_embeddings(sample_records, "not_a_provider")  # type: ignore

    def test_empty_content_in_record_raises_value_error(
        self, sample_doc_id: str, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Record with whitespace content raises ValueError."""
        bad_rec = EmbeddingRecord(
            chunk_id="c1",
            document_id=sample_doc_id,
            filename="a.pdf",
            chunk_index=0,
            page_number=1,
            content="   ",
            content_type="text",
        )
        with pytest.raises(ValueError, match="empty or whitespace-only content"):
            generate_embeddings([bad_rec], provider)

    def test_invalid_input_type_raises_type_error(
        self, provider: DeterministicTestEmbeddingProvider
    ) -> None:
        """Non-list / non-EmbeddingPreparationResult raises TypeError."""
        with pytest.raises(TypeError, match="Expected list"):
            generate_embeddings("invalid_input", provider)  # type: ignore
