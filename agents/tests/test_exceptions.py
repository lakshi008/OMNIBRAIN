"""
Unit tests for agent domain exceptions.
"""

from __future__ import annotations

import pytest

from agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentRoutingError,
    AgentValidationError,
)


class TestAgentExceptions:
    """Test suite for agent domain exception hierarchy."""

    def test_base_agent_error_inherits_from_exception(self) -> None:
        """Verify AgentError subclasses standard Exception."""
        assert issubclass(AgentError, Exception)
        err = AgentError("Base agent error")
        assert str(err) == "Base agent error"
        assert isinstance(err, Exception)

    def test_agent_validation_error_hierarchy(self) -> None:
        """Verify AgentValidationError subclasses AgentError."""
        assert issubclass(AgentValidationError, AgentError)
        err = AgentValidationError("Invalid field value")
        assert str(err) == "Invalid field value"
        assert isinstance(err, AgentError)
        assert isinstance(err, Exception)

    def test_agent_routing_error_hierarchy(self) -> None:
        """Verify AgentRoutingError subclasses AgentError."""
        assert issubclass(AgentRoutingError, AgentError)
        err = AgentRoutingError("Failed to determine route for query")
        assert str(err) == "Failed to determine route for query"
        assert isinstance(err, AgentError)
        assert isinstance(err, Exception)

    def test_agent_execution_error_hierarchy(self) -> None:
        """Verify AgentExecutionError subclasses AgentError."""
        assert issubclass(AgentExecutionError, AgentError)
        err = AgentExecutionError("Search tool execution timed out")
        assert str(err) == "Search tool execution timed out"
        assert isinstance(err, AgentError)
        assert isinstance(err, Exception)

    def test_catch_subclasses_via_base_agent_error(self) -> None:
        """Verify catching AgentError catches all specialized agent exceptions."""
        with pytest.raises(AgentError) as exc_info:
            raise AgentValidationError("Validation failure")
        assert "Validation failure" in str(exc_info.value)

        with pytest.raises(AgentError) as exc_info:
            raise AgentRoutingError("Routing failure")
        assert "Routing failure" in str(exc_info.value)

        with pytest.raises(AgentError) as exc_info:
            raise AgentExecutionError("Execution failure")
        assert "Execution failure" in str(exc_info.value)

    def test_exception_chaining(self) -> None:
        """Verify exceptions can chain from underlying causes."""
        cause = ValueError("Underlying value error")
        try:
            try:
                raise cause
            except ValueError as e:
                raise AgentExecutionError("Execution failed due to invalid data") from e
        except AgentExecutionError as agent_err:
            assert agent_err.__cause__ is cause
            assert str(agent_err) == "Execution failed due to invalid data"
