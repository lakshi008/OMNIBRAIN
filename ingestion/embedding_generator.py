"""
Embedding generator module for the OmniBrain ingestion pipeline.

Provides the embedding generation interface and provider abstraction to convert
EmbeddingRecord objects into EmbeddingVectorRecord objects for vector storage.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from ingestion.models import (
    EmbeddingGenerationResult,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    EmbeddingVectorRecord,
)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol defining the interface for pluggable embedding providers.

    Implementations can wrap SentenceTransformers, OpenAI embeddings, HuggingFace,
    or custom VLM/text embedding backends.
    """

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: Non-empty string content to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of text strings.

        Args:
            texts: List of non-empty strings to embed.

        Returns:
            List of embedding vectors corresponding to the input texts.
        """
        ...


def _call_provider(provider: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    """Invoke provider's batch or single embedding method with error handling."""
    if hasattr(provider, "embed_batch") and callable(getattr(provider, "embed_batch")):
        raw_output = provider.embed_batch(texts)
    elif hasattr(provider, "embed") and callable(getattr(provider, "embed")):
        raw_output = [provider.embed(t) for t in texts]
    else:
        raise TypeError(
            f"Provider '{type(provider).__name__}' must implement 'embed' or 'embed_batch' method."
        )

    if not isinstance(raw_output, list):
        raise ValueError(
            f"Provider returned {type(raw_output).__name__} instead of list[list[float]]."
        )

    return raw_output


def generate_embeddings(
    items: list[EmbeddingRecord] | EmbeddingPreparationResult,
    provider: EmbeddingProvider,
) -> EmbeddingGenerationResult:
    """Generate dense embedding vectors for prepared document records.

    Pipeline actions:
    1. Validates input types and provider contract.
    2. Sorts records deterministically by `chunk_index`.
    3. Rejects empty content or malformed records.
    4. Calls provider to generate embedding vectors.
    5. Validates vector types, numeric values, and dimension consistency across batch.
    6. Associates each vector with its chunk ID, document ID, page number, and metadata.

    Args:
        items: List of EmbeddingRecord objects or an EmbeddingPreparationResult.
        provider: EmbeddingProvider implementation.

    Returns:
        EmbeddingGenerationResult containing EmbeddingVectorRecord items and dimension.

    Raises:
        TypeError: If input types or provider do not conform to expected interfaces.
        ValueError: If content is empty, vectors are invalid, or vector dimensions are inconsistent.
    """
    if isinstance(items, EmbeddingPreparationResult):
        doc_id = items.document_id
        filename = items.filename
        record_list = items.items
    elif isinstance(items, list):
        record_list = items
        doc_id = items[0].document_id if items and hasattr(items[0], "document_id") else ""
        filename = items[0].filename if items and hasattr(items[0], "filename") else ""
    else:
        raise TypeError(
            f"Expected list[EmbeddingRecord] or EmbeddingPreparationResult, got {type(items).__name__}"
        )

    if provider is None or not (
        (hasattr(provider, "embed") and callable(getattr(provider, "embed")))
        or (hasattr(provider, "embed_batch") and callable(getattr(provider, "embed_batch")))
    ):
        raise TypeError(
            "Invalid embedding provider: must implement 'embed' or 'embed_batch' method."
        )

    # Handle empty input safely
    if not record_list:
        return EmbeddingGenerationResult(
            document_id=doc_id,
            filename=filename,
            items=[],
            dimension=0,
            is_ready=True,
        )

    # Deterministic sorting
    sorted_records = sorted(record_list, key=lambda r: (r.chunk_index, r.chunk_id))

    # Validate records
    texts: list[str] = []
    for idx, rec in enumerate(sorted_records):
        if not isinstance(rec, EmbeddingRecord):
            raise TypeError(
                f"Item at index {idx} is not an instance of EmbeddingRecord: got {type(rec).__name__}"
            )
        if not rec.content or not isinstance(rec.content, str) or not rec.content.strip():
            raise ValueError(
                f"Record '{rec.chunk_id}' (index {rec.chunk_index}) has empty or whitespace-only content."
            )
        texts.append(rec.content)

    # Call provider
    vectors = _call_provider(provider, texts)

    if len(vectors) != len(texts):
        raise ValueError(
            f"Provider returned {len(vectors)} vectors for {len(texts)} input records."
        )

    # Validate vectors and dimensions
    expected_dim: int | None = None
    validated_vectors: list[list[float]] = []

    for idx, vec in enumerate(vectors):
        if not isinstance(vec, list) or len(vec) == 0:
            raise ValueError(
                f"Provider returned empty or non-list vector for record '{sorted_records[idx].chunk_id}' (index {idx})."
            )

        # Check numeric elements
        cleaned_vec: list[float] = []
        for v_idx, val in enumerate(vec):
            if not isinstance(val, (int, float)) or isinstance(val, bool) or math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"Vector for record '{sorted_records[idx].chunk_id}' contains invalid value at element {v_idx}: {val!r}"
                )
            cleaned_vec.append(float(val))

        # Check dimensional consistency
        dim = len(cleaned_vec)
        if expected_dim is None:
            expected_dim = dim
        elif dim != expected_dim:
            raise ValueError(
                f"Inconsistent vector dimension for record '{sorted_records[idx].chunk_id}' at index {idx}: "
                f"expected {expected_dim}, got {dim}."
            )

        validated_vectors.append(cleaned_vec)

    # Build EmbeddingVectorRecord objects
    vector_records: list[EmbeddingVectorRecord] = []
    for rec, vec in zip(sorted_records, validated_vectors):
        vector_records.append(
            EmbeddingVectorRecord(
                chunk_id=rec.chunk_id,
                document_id=rec.document_id,
                filename=rec.filename,
                chunk_index=rec.chunk_index,
                page_number=rec.page_number,
                content_type=rec.content_type,
                vector=vec,
                metadata=dict(rec.metadata),
            )
        )

    return EmbeddingGenerationResult(
        document_id=doc_id,
        filename=filename,
        items=vector_records,
        dimension=expected_dim or 0,
        is_ready=True,
    )
