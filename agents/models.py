"""
Data models for OmniBrain agent architecture.

Defines strongly typed, validated representations for agent requests,
responses, citations preserving Member 1 lineage, and LangGraph workflow state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from agents.exceptions import AgentValidationError


@dataclass(frozen=True)
class AgentCitation:
    """Represents a source citation from retrieved document chunks.

    Preserves exact provenance lineage from Member 1 (document_id, filename,
    page_number, chunk_id, content_type, similarity score, and metadata).

    Attributes:
        document_id: Unique identifier of the parent document.
        filename: Name of the source file.
        page_number: 1-indexed page number from which content originated (or None).
        chunk_id: Unique UUID identifier of the source chunk.
        content_type: Modality of content ('text', 'table', 'image').
        score: Relevance/similarity score.
        metadata: Additional contextual metadata dictionary.
    """

    document_id: str
    filename: str
    chunk_id: str
    page_number: int | None = None
    content_type: str = "text"
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate citation fields."""
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise AgentValidationError("document_id must be a non-empty string.")

        if not isinstance(self.filename, str) or not self.filename.strip():
            raise AgentValidationError("filename must be a non-empty string.")

        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise AgentValidationError("chunk_id must be a non-empty string.")

        if self.page_number is not None:
            if not isinstance(self.page_number, int) or isinstance(self.page_number, bool) or self.page_number < 1:
                raise AgentValidationError(
                    f"page_number must be a positive integer (>= 1) or None, got {self.page_number!r}."
                )

        if not isinstance(self.content_type, str) or not self.content_type.strip():
            raise AgentValidationError("content_type must be a non-empty string.")

        if (
            not isinstance(self.score, (int, float))
            or isinstance(self.score, bool)
            or not math.isfinite(self.score)
        ):
            raise AgentValidationError(
                f"score must be a finite numeric float or int, got {self.score!r}."
            )

        if not isinstance(self.metadata, (dict, Mapping)):
            raise AgentValidationError("metadata must be a dictionary.")

        # Ensure score is stored as float and metadata is converted to dict
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_search_result(cls, result: Any) -> AgentCitation:
        """Construct an AgentCitation from a Member 1 VectorSearchResult.

        Args:
            result: VectorSearchResult instance or object with compatible attributes.

        Returns:
            Validated AgentCitation preserving all lineage fields.

        Raises:
            AgentValidationError: If required attributes are missing or invalid.
        """
        for attr in ("document_id", "filename", "chunk_id"):
            if not hasattr(result, attr):
                raise AgentValidationError(f"Search result is missing required attribute '{attr}'.")

        return cls(
            document_id=str(getattr(result, "document_id")),
            filename=str(getattr(result, "filename")),
            chunk_id=str(getattr(result, "chunk_id")),
            page_number=getattr(result, "page_number", None),
            content_type=str(getattr(result, "content_type", "text")),
            score=float(getattr(result, "score", 0.0)),
            metadata=dict(getattr(result, "metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert citation to dictionary representation."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "content_type": self.content_type,
            "score": self.score,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCitation:
        """Construct an AgentCitation from a dictionary.

        Args:
            data: Dictionary containing citation fields.

        Returns:
            Validated AgentCitation.
        """
        if not isinstance(data, dict):
            raise AgentValidationError("Input data must be a dictionary.")

        return cls(
            document_id=data.get("document_id", ""),
            filename=data.get("filename", ""),
            chunk_id=data.get("chunk_id", ""),
            page_number=data.get("page_number"),
            content_type=data.get("content_type", "text"),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class AgentRequest:
    """Represents an incoming query request to the agent system.

    Attributes:
        query: User input prompt or question.
        session_id: Optional unique identifier for conversation/session tracking.
        document_filter: Optional document filtering criteria.
        metadata: Optional metadata dictionary for routing or execution parameters.
    """

    query: str
    session_id: str | None = None
    document_filter: dict[str, Any] | list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate request fields."""
        if not isinstance(self.query, str):
            raise AgentValidationError(
                f"query must be a string, got {type(self.query).__name__}."
            )
        if not self.query.strip():
            raise AgentValidationError("query cannot be empty or whitespace-only.")

        if self.session_id is not None:
            if not isinstance(self.session_id, str) or not self.session_id.strip():
                raise AgentValidationError("session_id must be a non-empty string or None.")

        if self.document_filter is not None:
            if not isinstance(self.document_filter, (dict, list)):
                raise AgentValidationError(
                    "document_filter must be a dict, list of identifiers, or None."
                )

        if not isinstance(self.metadata, (dict, Mapping)):
            raise AgentValidationError("metadata must be a dictionary.")

        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Convert request to dictionary representation."""
        return {
            "query": self.query,
            "session_id": self.session_id,
            "document_filter": self.document_filter,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentRequest:
        """Construct an AgentRequest from a dictionary.

        Args:
            data: Dictionary containing request fields.

        Returns:
            Validated AgentRequest.
        """
        if not isinstance(data, dict):
            raise AgentValidationError("Input data must be a dictionary.")

        return cls(
            query=data.get("query", ""),
            session_id=data.get("session_id"),
            document_filter=data.get("document_filter"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentResponse:
    """Represents a response generated by an agent or multi-agent workflow.

    Attributes:
        answer: Text content of the answer or execution result.
        agent_name: Name of the agent producing the response.
        status: Execution status string ('success', 'error', 'partial').
        citations: List of source citations supporting the answer.
        metadata: Optional execution metadata (latency, routing details, token counts).
        error: Optional error message string if execution failed.
    """

    answer: str
    agent_name: str
    status: str = "success"
    citations: list[AgentCitation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate response fields."""
        if not isinstance(self.answer, str):
            raise AgentValidationError(f"answer must be a string, got {type(self.answer).__name__}.")

        if not isinstance(self.agent_name, str) or not self.agent_name.strip():
            raise AgentValidationError("agent_name must be a non-empty string.")

        if not isinstance(self.status, str) or not self.status.strip():
            raise AgentValidationError("status must be a non-empty string.")

        if not isinstance(self.citations, list):
            raise AgentValidationError(
                f"citations must be a list, got {type(self.citations).__name__}."
            )

        for idx, citation in enumerate(self.citations):
            if not isinstance(citation, AgentCitation):
                raise AgentValidationError(
                    f"Item at index {idx} of citations is not an AgentCitation: got {type(citation).__name__}."
                )

        if not isinstance(self.metadata, (dict, Mapping)):
            raise AgentValidationError("metadata must be a dictionary.")

        if self.error is not None:
            if not isinstance(self.error, str):
                raise AgentValidationError("error must be a string or None.")

        self.metadata = dict(self.metadata)

    @property
    def has_citations(self) -> bool:
        """Whether the response has at least one source citation."""
        return len(self.citations) > 0

    @property
    def total_citations(self) -> int:
        """Total number of citations attached to the response."""
        return len(self.citations)

    @property
    def is_success(self) -> bool:
        """Whether the response indicates successful execution."""
        return self.status == "success" and self.error is None

    @property
    def is_error(self) -> bool:
        """Whether the response indicates an error or failure."""
        return self.status == "error" or self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary representation."""
        return {
            "answer": self.answer,
            "agent_name": self.agent_name,
            "status": self.status,
            "citations": [c.to_dict() for c in self.citations],
            "metadata": dict(self.metadata),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentResponse:
        """Construct an AgentResponse from a dictionary.

        Args:
            data: Dictionary containing response fields.

        Returns:
            Validated AgentResponse.
        """
        if not isinstance(data, dict):
            raise AgentValidationError("Input data must be a dictionary.")

        raw_citations = data.get("citations", [])
        citations: list[AgentCitation] = []
        if isinstance(raw_citations, list):
            for item in raw_citations:
                if isinstance(item, AgentCitation):
                    citations.append(item)
                elif isinstance(item, dict):
                    citations.append(AgentCitation.from_dict(item))
                else:
                    raise AgentValidationError(
                        f"Invalid citation in data: expected dict or AgentCitation, got {type(item).__name__}."
                    )

        return cls(
            answer=data.get("answer", ""),
            agent_name=data.get("agent_name", ""),
            status=data.get("status", "success"),
            citations=citations,
            metadata=data.get("metadata", {}),
            error=data.get("error"),
        )


@dataclass
class AgentState:
    """State structure representing workflow context for LangGraph graph execution.

    Maintains current query, routing decision, retrieved candidate results,
    synthesized context, citations, generated answer, error history, and lifecycle status.

    Attributes:
        query: User input query string.
        route: Selected agent or pipeline route (e.g., 'search', 'vision', 'sql').
        retrieved_results: Raw or processed search results retrieved for the query.
        context: Synthesized textual context prepared for answer generation.
        citations: Provenance citations extracted from retrieved results.
        answer: Intermediate or final generated answer text.
        errors: List of error messages accumulated during execution.
        status: Lifecycle status of graph execution ('initialized', 'running', 'completed', 'failed').
        metadata: Extensible metadata dictionary.
    """

    query: str
    route: str | None = None
    retrieved_results: list[Any] = field(default_factory=list)
    context: str = ""
    citations: list[AgentCitation] = field(default_factory=list)
    answer: str = ""
    errors: list[str] = field(default_factory=list)
    status: str = "initialized"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate state fields."""
        if not isinstance(self.query, str):
            raise AgentValidationError(f"query must be a string, got {type(self.query).__name__}.")
        if not self.query.strip():
            raise AgentValidationError("query cannot be empty or whitespace-only.")

        if self.route is not None and not isinstance(self.route, str):
            raise AgentValidationError("route must be a string or None.")

        if not isinstance(self.retrieved_results, list):
            raise AgentValidationError("retrieved_results must be a list.")

        if not isinstance(self.context, str):
            raise AgentValidationError("context must be a string.")

        if not isinstance(self.citations, list):
            raise AgentValidationError("citations must be a list.")

        for idx, citation in enumerate(self.citations):
            if not isinstance(citation, AgentCitation):
                raise AgentValidationError(
                    f"Item at index {idx} of citations is not an AgentCitation: got {type(citation).__name__}."
                )

        if not isinstance(self.answer, str):
            raise AgentValidationError("answer must be a string.")

        if not isinstance(self.errors, list):
            raise AgentValidationError("errors must be a list.")

        for idx, err in enumerate(self.errors):
            if not isinstance(err, str):
                raise AgentValidationError(
                    f"Item at index {idx} of errors is not a string: got {type(err).__name__}."
                )

        if not isinstance(self.status, str) or not self.status.strip():
            raise AgentValidationError("status must be a non-empty string.")

        if not isinstance(self.metadata, (dict, Mapping)):
            raise AgentValidationError("metadata must be a dictionary.")

        self.metadata = dict(self.metadata)

    def add_error(self, error: str) -> None:
        """Append an error message to the state error log.

        Args:
            error: Descriptive error message string.
        """
        if not isinstance(error, str) or not error.strip():
            raise AgentValidationError("error must be a non-empty string.")
        self.errors.append(error)

    def add_citation(self, citation: AgentCitation) -> None:
        """Append a citation to the state citations list.

        Args:
            citation: AgentCitation instance.
        """
        if not isinstance(citation, AgentCitation):
            raise AgentValidationError(
                f"citation must be an AgentCitation, got {type(citation).__name__}."
            )
        self.citations.append(citation)

    def update(self, **kwargs: Any) -> None:
        """Update state attributes dynamically with validation.

        Args:
            **kwargs: Attribute-value pairs to update on the state instance.
        """
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise AgentValidationError(f"Unknown state attribute: {key}")
            setattr(self, key, val)
        self.__post_init__()

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            "query": self.query,
            "route": self.route,
            "retrieved_results": self.retrieved_results,
            "context": self.context,
            "citations": [c.to_dict() for c in self.citations],
            "answer": self.answer,
            "errors": list(self.errors),
            "status": self.status,
            "metadata": dict(self.metadata),
        }
