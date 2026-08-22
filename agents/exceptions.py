"""
Domain exception hierarchy for OmniBrain agents.

Defines the base AgentError and specialized exceptions for validation,
routing, and execution failures within Member 2 agent subsystems.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base exception for all agent domain errors."""

    pass


class AgentValidationError(AgentError):
    """Raised when agent inputs, state, requests, citations, or parameters fail validation."""

    pass


class AgentRoutingError(AgentError):
    """Raised when routing decisions cannot be determined or fail."""

    pass


class AgentExecutionError(AgentError):
    """Raised during agent tool execution, node processing, or workflow failure."""

    pass
