"""
Structured logging layer for the OmniBrain ingestion pipeline.

Provides a reusable named logger and helper functions for recording structured
pipeline events (start, stage complete, stage failed, ingestion complete/failed)
at appropriate log levels using Python's standard ``logging`` module only.

Design principles:
- One named logger (``ingestion.pipeline``) shared across the module.
- No duplicate handlers — idempotent module import.
- No print() calls.
- No secrets, passwords, API keys, or sensitive config values are logged.
- Logging failures must NEVER propagate to callers or affect pipeline results.
"""

from __future__ import annotations

import logging
from typing import Any

# ── Module-level named logger ─────────────────────────────────────────────

_LOGGER_NAME = "ingestion.pipeline"


def get_ingestion_logger(
    name: str = _LOGGER_NAME,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """Return (or create) the shared ingestion pipeline logger.

    Calling this function multiple times with the same ``name`` always returns
    the same ``logging.Logger`` instance without adding duplicate handlers.

    Args:
        name: Logger name. Defaults to ``'ingestion.pipeline'``.
        level: Minimum log level for the logger. Defaults to ``DEBUG`` so
               that the calling application can control filtering via its
               own handler configuration.

    Returns:
        A configured ``logging.Logger``.
    """
    logger = logging.getLogger(name)
    # Only set level if it hasn't been configured yet (avoids overriding
    # application-level config on repeated imports).
    if not logger.handlers and logger.level == logging.NOTSET:
        logger.setLevel(level)
    return logger


# ── Formatting helpers ────────────────────────────────────────────────────


def _fmt(**fields: Any) -> str:
    """Format key=value pairs into a compact, readable structured string.

    Only non-None values are included. Boolean False values are included.
    """
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value!r}")
    return " ".join(parts)


# ── Public logging API ────────────────────────────────────────────────────


class IngestionLogger:
    """Structured event logger for a single ingestion pipeline run.

    Wraps a ``logging.Logger`` and provides convenience methods for emitting
    structured events at appropriate levels without duplicating formatting logic
    across the service layer.

    All methods are safe to call: any internal logging exception is silently
    suppressed so that logging never interrupts the ingestion pipeline.

    Attributes:
        logger: The underlying ``logging.Logger`` used for output.
        document_id: Document UUID populated after extraction.
        filename: Source PDF filename populated after extraction.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger: logging.Logger = logger or get_ingestion_logger()
        self.document_id: str = ""
        self.filename: str = ""

    # ── Pipeline-level events ─────────────────────────────────────────────

    def log_ingestion_start(self, filename: str) -> None:
        """Emit an INFO record when the ingestion pipeline starts."""
        try:
            self.filename = filename
            self.logger.info(
                _fmt(event="ingestion_start", filename=filename, status="RUNNING")
            )
        except Exception:
            pass

    def log_ingestion_complete(
        self,
        document_id: str,
        filename: str,
        *,
        total_duration_seconds: float = 0.0,
        total_chunks: int = 0,
        text_chunks: int = 0,
        table_chunks: int = 0,
        image_chunks: int = 0,
        total_embedding_items: int = 0,
        total_vectors: int = 0,
    ) -> None:
        """Emit an INFO record when the ingestion pipeline completes successfully."""
        try:
            self.document_id = document_id
            self.logger.info(
                _fmt(
                    event="ingestion_complete",
                    document_id=document_id,
                    filename=filename,
                    status="COMPLETED",
                    total_duration_seconds=round(total_duration_seconds, 4),
                    total_chunks=total_chunks,
                    text_chunks=text_chunks,
                    table_chunks=table_chunks,
                    image_chunks=image_chunks,
                    total_embedding_items=total_embedding_items,
                    total_vectors=total_vectors,
                )
            )
        except Exception:
            pass

    def log_ingestion_failed(
        self,
        filename: str,
        *,
        stage: str = "",
        error: str = "",
        total_duration_seconds: float = 0.0,
    ) -> None:
        """Emit an ERROR record when the ingestion pipeline fails."""
        try:
            self.logger.error(
                _fmt(
                    event="ingestion_failed",
                    filename=filename,
                    stage=stage or None,
                    status="FAILED",
                    total_duration_seconds=round(total_duration_seconds, 4),
                    error=error or None,
                )
            )
        except Exception:
            pass

    # ── Stage-level events ────────────────────────────────────────────────

    def log_stage_start(self, stage: str, filename: str = "") -> None:
        """Emit an INFO record when a pipeline stage begins."""
        try:
            self.logger.info(
                _fmt(
                    event="stage_start",
                    stage=stage,
                    filename=filename or self.filename or None,
                )
            )
        except Exception:
            pass

    def log_stage_complete(
        self,
        stage: str,
        *,
        duration_seconds: float = 0.0,
        document_id: str = "",
        filename: str = "",
        **extra: Any,
    ) -> None:
        """Emit a DEBUG record when a pipeline stage completes successfully.

        Additional keyword arguments are included as supplementary fields
        (e.g. ``total_chunks``, ``total_vectors``).
        """
        try:
            fields: dict[str, Any] = dict(
                event="stage_complete",
                stage=stage,
                status="COMPLETED",
                duration_seconds=round(duration_seconds, 4),
                document_id=document_id or self.document_id or None,
                filename=filename or self.filename or None,
            )
            fields.update({k: v for k, v in extra.items() if v is not None})
            self.logger.debug(_fmt(**fields))
        except Exception:
            pass

    def log_stage_failed(
        self,
        stage: str,
        *,
        duration_seconds: float = 0.0,
        error: str = "",
        filename: str = "",
    ) -> None:
        """Emit an ERROR record when a pipeline stage fails."""
        try:
            self.logger.error(
                _fmt(
                    event="stage_failed",
                    stage=stage,
                    status="FAILED",
                    duration_seconds=round(duration_seconds, 4),
                    filename=filename or self.filename or None,
                    error=error or None,
                )
            )
        except Exception:
            pass

    # ── Metrics-driven helpers ────────────────────────────────────────────

    def log_from_metrics(self, metrics: object) -> None:
        """Emit a DEBUG record summarising all counter fields from an IngestionMetrics object.

        Reads the following attributes if present (no AttributeError on missing):
        ``total_chunks``, ``text_chunks``, ``table_chunks``, ``image_chunks``,
        ``total_embedding_items``, ``total_vectors``, ``total_duration_seconds``.

        This avoids duplicating the counter/timing logic already implemented
        in Day 16 ``IngestionMetrics``.
        """
        try:
            self.logger.debug(
                _fmt(
                    event="metrics_summary",
                    document_id=getattr(metrics, "document_id", None) or None,
                    filename=getattr(metrics, "filename", None) or None,
                    status=getattr(metrics, "status", None),
                    total_duration_seconds=round(
                        float(getattr(metrics, "total_duration_seconds", 0.0)), 4
                    ),
                    total_chunks=getattr(metrics, "total_chunks", None),
                    text_chunks=getattr(metrics, "text_chunks", None),
                    table_chunks=getattr(metrics, "table_chunks", None),
                    image_chunks=getattr(metrics, "image_chunks", None),
                    total_embedding_items=getattr(metrics, "total_embedding_items", None),
                    total_vectors=getattr(metrics, "total_vectors", None),
                )
            )
        except Exception:
            pass
