"""
Vision Result Normalizer and Execution Trace Layer for OmniBrain Member 3.

Validates, sanitizes, and normalizes output from Vision Model Providers into a standardized
VisionResult contract. Ensures vendor-specific leakages, secrets, raw image bytes, and
malformed structures never leak into the rest of OMNIBRAIN.

Day 39 Scope:
  - VisionExecutionTrace: Lightweight, deterministic trace recorder (zero fake timestamps/latency).
  - VisionResultNormalizer: Strict output validator, lineage preserver, and metadata sanitizer.
  - Zero network, zero LLM, zero fake Vision results in production code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vision.exceptions import (
    VisionInputValidationError,
    VisionProcessingError,
)
from vision.input_builder import VisionModelInput
from vision.models import VisionRequest, VisionResult, VisualEvidence


# ---------------------------------------------------------------------------
# Forbidden metadata keys (Security & Sanitization Boundary)
# ---------------------------------------------------------------------------

FORBIDDEN_METADATA_KEYS: frozenset[str] = frozenset({
    "api_key",
    "apikey",
    "secret",
    "password",
    "token",
    "auth",
    "authorization",
    "credentials",
    "bearer",
    "image_bytes",
    "base64",
    "raw_request_headers",
    "access_token",
})


# ---------------------------------------------------------------------------
# VisionExecutionTrace
# ---------------------------------------------------------------------------


class VisionExecutionTrace:
    """Lightweight, deterministic execution trace recorder for tracking processing stages.

    Maintains an ordered list of safe stage identifiers throughout execution without
    generating fake timestamps, fake latency metrics, or storing credentials.
    """

    DEFAULT_STAGES: tuple[str, ...] = (
        "request_received",
        "validation_started",
        "input_prepared",
        "provider_started",
        "provider_completed",
        "result_normalized",
        "execution_completed",
    )

    def __init__(self, initial_stages: list[str] | None = None) -> None:
        """Initialize VisionExecutionTrace with optional initial stage list."""
        self._stages: list[str] = []
        if initial_stages:
            for stg in initial_stages:
                self.add_stage(stg)

    @property
    def stages(self) -> list[str]:
        """Return a copy of recorded trace stages."""
        return list(self._stages)

    def add_stage(self, stage_name: str) -> None:
        """Record a new stage identifier in the trace sequence.

        Args:
            stage_name: Non-empty string identifying the execution stage.

        Raises:
            VisionInputValidationError: If stage_name is non-string or empty.
        """
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise VisionInputValidationError("stage_name must be a non-empty string.")
        clean = stage_name.strip().lower()
        self._stages.append(clean)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation of the trace."""
        return {
            "stages": list(self._stages),
            "stage_count": len(self._stages),
        }

    @classmethod
    def create_default(cls) -> VisionExecutionTrace:
        """Construct a VisionExecutionTrace populated with default execution stages."""
        return cls(initial_stages=list(cls.DEFAULT_STAGES))


# ---------------------------------------------------------------------------
# VisionResultNormalizer
# ---------------------------------------------------------------------------


class VisionResultNormalizer:
    """Normalizes and sanitizes raw provider output into a clean VisionResult contract.

    Responsibilities:
      1. Validate provider output type (must be VisionResult or serializable dict).
      2. Reject malformed, None, or unexpected objects.
      3. Verify result status (success, no_evidence, error, not_implemented).
      4. Preserve source document lineage (document_id, filename, chunk_id, page_number, etc.).
      5. Recursively sanitize metadata (stripping API keys, secrets, image bytes).
      6. Attach safe VisionExecutionTrace metadata without vendor leakage.
    """

    @classmethod
    def sanitize_metadata(cls, metadata: Any) -> dict[str, Any]:
        """Recursively sanitize metadata dictionary to remove secrets and raw byte payloads.

        Args:
            metadata: Raw metadata dictionary from provider or request.

        Returns:
            Clean dictionary free of forbidden keys.
        """
        if not isinstance(metadata, (dict, Mapping)):
            return {}

        clean: dict[str, Any] = {}
        for key, val in metadata.items():
            if not isinstance(key, str):
                continue
            key_lower = key.strip().lower()
            if key_lower in FORBIDDEN_METADATA_KEYS:
                continue

            if isinstance(val, (dict, Mapping)):
                clean[key] = cls.sanitize_metadata(val)
            elif isinstance(val, list):
                clean[key] = [
                    cls.sanitize_metadata(item) if isinstance(item, (dict, Mapping)) else item
                    for item in val
                    if not isinstance(item, (bytes, bytearray))
                ]
            elif isinstance(val, (bytes, bytearray)):
                # Strip raw byte arrays from metadata dictionary
                continue
            else:
                clean[key] = val

        return clean

    @classmethod
    def normalize(
        cls,
        result: Any,
        request: str | VisionRequest | None = None,
        model_input: VisionModelInput | None = None,
        trace: VisionExecutionTrace | None = None,
    ) -> VisionResult:
        """Validate and normalize raw provider output into a standardized VisionResult.

        Args:
            result: Candidate provider result (VisionResult or dict).
            request: Optional source VisionRequest or query string.
            model_input: Optional Day 35 VisionModelInput for lineage verification.
            trace: Optional VisionExecutionTrace recorder.

        Returns:
            Standardized, lineage-locked VisionResult.

        Raises:
            VisionProcessingError: If result is None, invalid type, or malformed.
        """
        if result is None:
            raise VisionProcessingError("Provider returned None instead of a valid VisionResult.")

        parsed_result: VisionResult
        if isinstance(result, VisionResult):
            parsed_result = result
        elif isinstance(result, dict):
            try:
                parsed_result = VisionResult.from_dict(result)
            except Exception as err:
                raise VisionProcessingError(
                    f"Failed to normalize dictionary provider result: {err}"
                ) from err
        else:
            raise VisionProcessingError(
                f"Provider returned invalid result type '{type(result).__name__}', "
                f"expected VisionResult."
            )

        # Status validation
        if not isinstance(parsed_result.status, str) or not parsed_result.status.strip():
            raise VisionProcessingError("Normalized result status must be a non-empty string.")

        # Lineage reconciliation with model_input if present
        if model_input is not None:
            if not parsed_result.document_id:
                parsed_result.document_id = model_input.document_id
            if not parsed_result.filename:
                parsed_result.filename = model_input.filename
            if parsed_result.page_number is None:
                parsed_result.page_number = model_input.page_number
            if not parsed_result.chunk_id:
                parsed_result.chunk_id = model_input.chunk_id
            if parsed_result.content_type == "image" and model_input.content_type != "image":
                parsed_result.content_type = model_input.content_type

        # Sanitize metadata
        sanitized_meta = cls.sanitize_metadata(parsed_result.metadata)

        # Attach execution trace safely
        if trace is not None:
            trace.add_stage("result_normalized")
            trace.add_stage("execution_completed")
            sanitized_meta["execution_trace"] = trace.to_dict()

        parsed_result.metadata = sanitized_meta
        return parsed_result
