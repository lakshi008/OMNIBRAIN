"""
Lightweight metrics and execution-statistics layer for the OmniBrain ingestion pipeline.

Provides structured, deterministic, dataclass-based metrics with monotonic per-stage
timing, chunk counters, embedding counters, and pipeline-level status summary.

No external dependencies. Standard library only.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Generator


# ── Stage Metrics ─────────────────────────────────────────────────────────


@dataclass
class StageMetrics:
    """Execution statistics for a single ingestion pipeline stage.

    Attributes:
        stage: Name of the pipeline stage (e.g. 'EXTRACTION', 'CHUNKING').
        duration_seconds: Wall-clock time measured with ``time.perf_counter()``.
            Always >= 0.
        success: True when the stage completed without raising an exception.
        error: Optional descriptive error message recorded on failure.
    """

    stage: str
    duration_seconds: float = 0.0
    success: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        if self.duration_seconds < 0:
            raise ValueError(
                f"'duration_seconds' must be non-negative, got {self.duration_seconds!r}."
            )


# ── Ingestion Metrics ─────────────────────────────────────────────────────


@dataclass
class IngestionMetrics:
    """Aggregated execution metrics for a complete ingestion pipeline run.

    Attributes:
        document_id: UUID identifier of the ingested document (populated after extraction).
        filename: Name of the source PDF file.
        status: Final pipeline status string ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED').
        current_stage: The last active or completed stage name.
        total_duration_seconds: Wall-clock time for the full pipeline run. Always >= 0.
        stage_metrics: Ordered list of StageMetrics, one per executed stage.
        total_chunks: Total number of DocumentChunk objects produced.
        text_chunks: Number of text-content chunks.
        table_chunks: Number of table-content chunks.
        image_chunks: Number of image-content chunks.
        total_embedding_items: Total EmbeddingRecord items prepared for embedding.
        total_vectors: Total EmbeddingVectorRecord objects generated.
        total_retrieval_results: Number of retrieval results when retrieval was performed.
        error: Descriptive error message recorded when the pipeline fails.
    """

    document_id: str = ""
    filename: str = ""
    status: str = "PENDING"
    current_stage: str = ""
    total_duration_seconds: float = 0.0
    stage_metrics: list[StageMetrics] = field(default_factory=list)
    total_chunks: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    total_embedding_items: int = 0
    total_vectors: int = 0
    total_retrieval_results: int = 0
    error: str | None = None

    # ── Private timing state ──────────────────────────────────────────────

    _pipeline_start: float = field(default=0.0, repr=False, compare=False)

    def start_pipeline(self) -> None:
        """Record the pipeline start time using a monotonic clock."""
        self._pipeline_start = time.perf_counter()
        self.status = "RUNNING"

    def finish_pipeline(self, *, success: bool, error: str | None = None) -> None:
        """Compute total pipeline duration and set final status.

        Args:
            success: True when all stages completed without error.
            error: Optional error message for failed pipelines.
        """
        elapsed = time.perf_counter() - self._pipeline_start
        self.total_duration_seconds = max(0.0, elapsed)
        if success:
            self.status = "COMPLETED"
        else:
            self.status = "FAILED"
            self.error = error

    @contextlib.contextmanager
    def track_stage(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager that measures a single pipeline stage.

        Records start time, end time, duration, and success/failure.
        Exceptions propagate unchanged after the failure is recorded.

        Args:
            stage_name: Human-readable stage identifier.

        Yields:
            Nothing — use the ``with`` block to execute the stage body.

        Example::

            with metrics.track_stage("CHUNKING"):
                chunking_result = chunk_document(...)
        """
        self.current_stage = stage_name
        stage = StageMetrics(stage=stage_name)
        t0 = time.perf_counter()
        try:
            yield
            elapsed = time.perf_counter() - t0
            stage.duration_seconds = max(0.0, elapsed)
            stage.success = True
            self.stage_metrics.append(stage)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            stage.duration_seconds = max(0.0, elapsed)
            stage.success = False
            stage.error = str(exc)
            self.stage_metrics.append(stage)
            raise

    # ── Counter helpers ───────────────────────────────────────────────────

    def record_chunks(self, chunking_result: object) -> None:
        """Populate chunk counters from a ChunkingResult object.

        Reads ``total_chunks``, ``text_chunks``, ``table_chunks``, and
        ``image_chunks`` from the result's existing properties without
        duplicating any calculation logic.

        Args:
            chunking_result: A ChunkingResult produced by ``chunk_document()``.
        """
        self.total_chunks = getattr(chunking_result, "total_chunks", 0)
        self.text_chunks = getattr(chunking_result, "text_chunks", 0)
        self.table_chunks = getattr(chunking_result, "table_chunks", 0)
        self.image_chunks = getattr(chunking_result, "image_chunks", 0)

    def record_embeddings(self, generation_result: object) -> None:
        """Populate embedding counters from an EmbeddingGenerationResult object.

        Reads ``total_items`` for the embedding item count and ``len(items)``
        for the vector count from the result's existing properties.

        Args:
            generation_result: An EmbeddingGenerationResult from ``generate_embeddings()``.
        """
        self.total_embedding_items = getattr(generation_result, "total_items", 0)
        self.total_vectors = getattr(generation_result, "total_items", 0)

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def succeeded(self) -> bool:
        """True if the pipeline completed successfully."""
        return self.status == "COMPLETED"

    @property
    def failed(self) -> bool:
        """True if the pipeline ended in a failed state."""
        return self.status == "FAILED"

    def get_stage(self, stage_name: str) -> StageMetrics | None:
        """Return the StageMetrics for the given stage name, or None if not found.

        Args:
            stage_name: The stage identifier to look up.
        """
        for sm in self.stage_metrics:
            if sm.stage == stage_name:
                return sm
        return None

    def successful_stages(self) -> list[str]:
        """Return names of all successfully completed stages, in execution order."""
        return [sm.stage for sm in self.stage_metrics if sm.success]

    def failed_stage(self) -> StageMetrics | None:
        """Return the first failed StageMetrics, or None if all stages succeeded."""
        for sm in self.stage_metrics:
            if not sm.success:
                return sm
        return None
