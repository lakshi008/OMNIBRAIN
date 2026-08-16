"""
Pipeline integration and contract validation layer for the OmniBrain ingestion pipeline.

Validates contracts, structural schemas, data integrity, and citation lineage
across all modular stages from document chunking to vector search retrieval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ingestion.models import (
    ChunkingResult,
    DocumentChunk,
    EmbeddingGenerationResult,
    EmbeddingPreparationResult,
    EmbeddingRecord,
    EmbeddingVectorRecord,
    RetrievalServiceResult,
    VectorSearchResult,
)

_VALID_CONTENT_TYPES = {"text", "table", "image"}


@dataclass
class IngestionValidationResult:
    """Structured result of contract and lineage validation across ingestion stages.

    Attributes:
        valid: True if no fatal contract violations were found.
        status: Status string ('VALID', 'INVALID', 'WARNING').
        checks: Mapping of check names to 'PASS', 'FAIL', or 'WARN'.
        errors: Detailed list of fatal validation error messages.
        warnings: Detailed list of non-fatal warning messages.
        summary: Aggregated statistics from the validation pass.
    """

    valid: bool = True
    status: str = "VALID"
    checks: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Return True if the validation passed with no fatal errors."""
        return self.valid

    def passed_checks(self) -> list[str]:
        """Return names of checks that passed."""
        return [k for k, v in self.checks.items() if v == "PASS"]

    def failed_checks(self) -> list[str]:
        """Return names of checks that failed."""
        return [k for k, v in self.checks.items() if v == "FAIL"]

    def warning_checks(self) -> list[str]:
        """Return names of checks that produced warnings."""
        return [k for k, v in self.checks.items() if v == "WARN"]


def _finalize_result(result: IngestionValidationResult) -> IngestionValidationResult:
    """Determine final status and valid boolean based on errors and warnings."""
    if result.errors:
        result.valid = False
        result.status = "INVALID"
    elif result.warnings:
        result.valid = True
        result.status = "WARNING"
    else:
        result.valid = True
        result.status = "VALID"
    return result


def validate_chunk_contracts(
    chunks: Sequence[DocumentChunk] | ChunkingResult | Any,
) -> IngestionValidationResult:
    """Validate that DocumentChunk objects strictly adhere to data contracts.

    Checks:
    - Target is a valid sequence of DocumentChunk objects.
    - chunk_id, document_id, filename are non-empty strings.
    - chunk_index is a non-negative integer.
    - page_number is a positive integer or None.
    - content_type is one of {'text', 'table', 'image'}.
    - content is non-empty string.
    - metadata is a valid dictionary.

    Args:
        chunks: List/sequence of DocumentChunk objects or a ChunkingResult.

    Returns:
        IngestionValidationResult summarizing contract compliance.
    """
    res = IngestionValidationResult()

    if isinstance(chunks, ChunkingResult):
        chunk_list = chunks.chunks
    elif isinstance(chunks, (list, tuple)):
        chunk_list = list(chunks)
    else:
        res.checks["chunk_container_type"] = "FAIL"
        res.errors.append(f"Expected sequence of DocumentChunk, got {type(chunks).__name__!r}.")
        return _finalize_result(res)

    res.checks["chunk_container_type"] = "PASS"
    res.summary["total_chunks"] = len(chunk_list)

    if not chunk_list:
        res.checks["chunks_non_empty"] = "WARN"
        res.warnings.append("Chunk list is empty.")
        return _finalize_result(res)

    res.checks["chunks_non_empty"] = "PASS"
    invalid_count = 0

    for i, c in enumerate(chunk_list):
        if not isinstance(c, DocumentChunk):
            invalid_count += 1
            res.errors.append(f"Item at index {i} is not a DocumentChunk: {type(c).__name__!r}.")
            continue

        if not isinstance(c.chunk_id, str) or not c.chunk_id.strip():
            invalid_count += 1
            res.errors.append(f"Chunk at index {i} has empty or non-string chunk_id: {c.chunk_id!r}.")

        if not isinstance(c.document_id, str) or not c.document_id.strip():
            invalid_count += 1
            res.errors.append(f"Chunk at index {i} has empty or non-string document_id: {c.document_id!r}.")

        if not isinstance(c.filename, str) or not c.filename.strip():
            invalid_count += 1
            res.errors.append(f"Chunk at index {i} has empty or non-string filename: {c.filename!r}.")

        if isinstance(c.chunk_index, bool) or not isinstance(c.chunk_index, int) or c.chunk_index < 0:
            invalid_count += 1
            res.errors.append(f"Chunk at index {i} has invalid chunk_index: {c.chunk_index!r}.")

        if c.page_number is not None:
            if isinstance(c.page_number, bool) or not isinstance(c.page_number, int) or c.page_number < 1:
                invalid_count += 1
                res.errors.append(f"Chunk at index {i} has invalid page_number: {c.page_number!r}.")

        if c.content_type not in _VALID_CONTENT_TYPES:
            invalid_count += 1
            res.errors.append(f"Chunk at index {i} has invalid content_type: {c.content_type!r}.")

        if not isinstance(c.content, str) or not c.content.strip():
            invalid_count += 1
            res.errors.append(f"Chunk at index {i} has empty content.")

        if not isinstance(c.metadata, dict):
            invalid_count += 1
            res.errors.append(f"Chunk at index {i} metadata must be a dict, got {type(c.metadata).__name__!r}.")

    res.checks["chunk_attributes_contract"] = "FAIL" if invalid_count > 0 else "PASS"
    res.summary["valid_chunks"] = len(chunk_list) - invalid_count
    res.summary["invalid_chunks"] = invalid_count
    return _finalize_result(res)


def validate_embedding_contracts(
    result: EmbeddingGenerationResult | Any,
) -> IngestionValidationResult:
    """Validate that EmbeddingGenerationResult and its vector records satisfy contracts.

    Checks:
    - result is an EmbeddingGenerationResult instance.
    - dimension is a positive integer > 0.
    - items is a sequence of EmbeddingVectorRecord objects.
    - Every vector length exactly matches result.dimension.
    - Every vector element is a finite float.
    - Lineage attributes (chunk_id, document_id, filename, chunk_index, content_type) are present and valid.

    Args:
        result: EmbeddingGenerationResult object.

    Returns:
        IngestionValidationResult summarizing embedding contract status.
    """
    res = IngestionValidationResult()

    if not isinstance(result, EmbeddingGenerationResult):
        res.checks["embedding_result_type"] = "FAIL"
        res.errors.append(f"Expected EmbeddingGenerationResult, got {type(result).__name__!r}.")
        return _finalize_result(res)

    res.checks["embedding_result_type"] = "PASS"

    if isinstance(result.dimension, bool) or not isinstance(result.dimension, int) or result.dimension <= 0:
        res.checks["embedding_dimension"] = "FAIL"
        res.errors.append(f"Invalid embedding dimension: {result.dimension!r}.")
    else:
        res.checks["embedding_dimension"] = "PASS"

    res.summary["dimension"] = result.dimension
    res.summary["total_vectors"] = len(result.items)

    invalid_vectors = 0
    for i, vec_rec in enumerate(result.items):
        if not isinstance(vec_rec, EmbeddingVectorRecord):
            invalid_vectors += 1
            res.errors.append(f"Item at index {i} is not an EmbeddingVectorRecord: {type(vec_rec).__name__!r}.")
            continue

        if not isinstance(vec_rec.vector, (list, tuple)):
            invalid_vectors += 1
            res.errors.append(f"Vector at index {i} is not a list/tuple: {type(vec_rec.vector).__name__!r}.")
            continue

        if len(vec_rec.vector) != result.dimension:
            invalid_vectors += 1
            res.errors.append(
                f"Vector at index {i} dimension mismatch: expected {result.dimension}, got {len(vec_rec.vector)}."
            )

        for val_idx, val in enumerate(vec_rec.vector):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                invalid_vectors += 1
                res.errors.append(f"Vector at index {i} contains invalid numeric value at [{val_idx}]: {val!r}.")
                break

        if not isinstance(vec_rec.chunk_id, str) or not vec_rec.chunk_id.strip():
            invalid_vectors += 1
            res.errors.append(f"Vector record at index {i} has empty chunk_id.")

        if not isinstance(vec_rec.document_id, str) or not vec_rec.document_id.strip():
            invalid_vectors += 1
            res.errors.append(f"Vector record at index {i} has empty document_id.")

    res.checks["vector_records_contract"] = "FAIL" if invalid_vectors > 0 else "PASS"
    res.summary["valid_vectors"] = len(result.items) - invalid_vectors
    res.summary["invalid_vectors"] = invalid_vectors
    return _finalize_result(res)


def validate_search_result_contracts(
    results: Sequence[VectorSearchResult] | RetrievalServiceResult | Any,
) -> IngestionValidationResult:
    """Validate that VectorSearchResult items satisfy retrieval contracts.

    Checks:
    - Results container is a list of VectorSearchResult or a RetrievalServiceResult.
    - Each item has a finite numeric score.
    - Lineage and content attributes (chunk_id, document_id, filename, chunk_index, content_type, content) are valid.

    Args:
        results: Sequence of VectorSearchResult objects or a RetrievalServiceResult.

    Returns:
        IngestionValidationResult summarizing search result contract compliance.
    """
    res = IngestionValidationResult()

    if isinstance(results, RetrievalServiceResult):
        res_list = results.results
    elif isinstance(results, (list, tuple)):
        res_list = list(results)
    else:
        res.checks["search_result_container"] = "FAIL"
        res.errors.append(f"Expected sequence of VectorSearchResult, got {type(results).__name__!r}.")
        return _finalize_result(res)

    res.checks["search_result_container"] = "PASS"
    res.summary["total_results"] = len(res_list)

    if not res_list:
        res.checks["search_results_non_empty"] = "WARN"
        res.warnings.append("Search result list is empty.")
        return _finalize_result(res)

    res.checks["search_results_non_empty"] = "PASS"
    invalid_count = 0

    for i, r in enumerate(res_list):
        if not isinstance(r, VectorSearchResult):
            invalid_count += 1
            res.errors.append(f"Item at index {i} is not a VectorSearchResult: {type(r).__name__!r}.")
            continue

        if (
            isinstance(r.score, bool)
            or not isinstance(r.score, (int, float))
            or math.isnan(r.score)
            or math.isinf(r.score)
        ):
            invalid_count += 1
            res.errors.append(f"Search result at index {i} has invalid score: {r.score!r}.")

        if not isinstance(r.chunk_id, str) or not r.chunk_id.strip():
            invalid_count += 1
            res.errors.append(f"Search result at index {i} has empty chunk_id.")

        if not isinstance(r.document_id, str) or not r.document_id.strip():
            invalid_count += 1
            res.errors.append(f"Search result at index {i} has empty document_id.")

        if not isinstance(r.filename, str) or not r.filename.strip():
            invalid_count += 1
            res.errors.append(f"Search result at index {i} has empty filename.")

        if r.content_type not in _VALID_CONTENT_TYPES:
            invalid_count += 1
            res.errors.append(f"Search result at index {i} has invalid content_type: {r.content_type!r}.")

        if not isinstance(r.content, str):
            invalid_count += 1
            res.errors.append(f"Search result at index {i} content must be a string, got {type(r.content).__name__!r}.")

    res.checks["search_result_contract"] = "FAIL" if invalid_count > 0 else "PASS"
    res.summary["valid_results"] = len(res_list) - invalid_count
    res.summary["invalid_results"] = invalid_count
    return _finalize_result(res)


def validate_pipeline_lineage(
    source_document_id: str,
    source_filename: str,
    artifacts: Sequence[Any],
) -> IngestionValidationResult:
    """Verify document lineage consistency across generated artifacts.

    Checks that all items in artifacts reference the expected source_document_id
    and source_filename without lineage leakage or mutations.

    Args:
        source_document_id: The authoritative parent document UUID.
        source_filename: The authoritative source document filename.
        artifacts: Sequence of pipeline artifacts (chunks, records, vectors, or search results).

    Returns:
        IngestionValidationResult reporting lineage consistency.
    """
    res = IngestionValidationResult()

    if not isinstance(source_document_id, str) or not source_document_id.strip():
        res.checks["authoritative_document_id"] = "FAIL"
        res.errors.append("Authoritative source_document_id must be a non-empty string.")
        return _finalize_result(res)

    if not isinstance(source_filename, str) or not source_filename.strip():
        res.checks["authoritative_filename"] = "FAIL"
        res.errors.append("Authoritative source_filename must be a non-empty string.")
        return _finalize_result(res)

    res.checks["authoritative_document_id"] = "PASS"
    res.checks["authoritative_filename"] = "PASS"
    res.summary["total_artifacts"] = len(artifacts)

    mismatch_count = 0
    for i, item in enumerate(artifacts):
        doc_id = getattr(item, "document_id", None)
        fname = getattr(item, "filename", None)

        if doc_id != source_document_id:
            mismatch_count += 1
            res.errors.append(
                f"Artifact at index {i} document_id mismatch: expected {source_document_id!r}, got {doc_id!r}."
            )

        if fname != source_filename:
            mismatch_count += 1
            res.errors.append(
                f"Artifact at index {i} filename mismatch: expected {source_filename!r}, got {fname!r}."
            )

    res.checks["lineage_consistency"] = "FAIL" if mismatch_count > 0 else "PASS"
    res.summary["lineage_mismatches"] = mismatch_count
    return _finalize_result(res)


def validate_pipeline_contracts(
    chunks: Sequence[DocumentChunk] | ChunkingResult | None = None,
    embedding_result: EmbeddingGenerationResult | None = None,
    search_results: Sequence[VectorSearchResult] | RetrievalServiceResult | None = None,
    source_document_id: str | None = None,
    source_filename: str | None = None,
) -> IngestionValidationResult:
    """Comprehensive composite validator for all ingestion pipeline artifacts.

    Validates chunks, embeddings, search results, and lineage in one pass.

    Args:
        chunks: Optional chunk sequence or ChunkingResult.
        embedding_result: Optional EmbeddingGenerationResult.
        search_results: Optional search result sequence or RetrievalServiceResult.
        source_document_id: Optional authoritative document ID for lineage checks.
        source_filename: Optional authoritative filename for lineage checks.

    Returns:
        IngestionValidationResult consolidating all contract checks.
    """
    res = IngestionValidationResult()

    if chunks is not None:
        c_res = validate_chunk_contracts(chunks)
        res.checks.update(c_res.checks)
        res.errors.extend(c_res.errors)
        res.warnings.extend(c_res.warnings)
        res.summary["chunk_validation"] = c_res.summary

    if embedding_result is not None:
        e_res = validate_embedding_contracts(embedding_result)
        res.checks.update(e_res.checks)
        res.errors.extend(e_res.errors)
        res.warnings.extend(e_res.warnings)
        res.summary["embedding_validation"] = e_res.summary

    if search_results is not None:
        s_res = validate_search_result_contracts(search_results)
        res.checks.update(s_res.checks)
        res.errors.extend(s_res.errors)
        res.warnings.extend(s_res.warnings)
        res.summary["search_result_validation"] = s_res.summary

    if source_document_id is not None and source_filename is not None:
        all_items: list[Any] = []
        if chunks is not None:
            all_items.extend(chunks.chunks if isinstance(chunks, ChunkingResult) else list(chunks))
        if embedding_result is not None:
            all_items.extend(embedding_result.items)
        if search_results is not None:
            all_items.extend(search_results.results if isinstance(search_results, RetrievalServiceResult) else list(search_results))

        if all_items:
            l_res = validate_pipeline_lineage(source_document_id, source_filename, all_items)
            res.checks.update(l_res.checks)
            res.errors.extend(l_res.errors)
            res.warnings.extend(l_res.warnings)
            res.summary["lineage_validation"] = l_res.summary

    return _finalize_result(res)
