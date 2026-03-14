"""
Unit tests for agent node human-in-the-loop (HITL) clarification support.

This test suite validates the extensibility hooks for HITL support in agent nodes.
The actual pause/resume functionality will be implemented in follow-up PRs.
"""

from unittest.mock import MagicMock
from typing import Any

import pytest

from core.workflow.nodes.agent.agent_node import AgentNode
from core.workflow.nodes.agent.entities import AgentNodeData


class TestAgentNodeClarificationExtensibility:
    """Test agent node clarification detection extensibility."""

    def test_agent_node_data_enable_clarification_default(self):
        """Test that enable_human_clarification defaults to False."""
        node_data = AgentNodeData(
            agent_strategy_provider_name="test_provider",
            agent_strategy_name="test_strategy",
            agent_strategy_label="Test Strategy",
            agent_parameters={},
        )

        assert node_data.enable_human_clarification is False

    def test_agent_node_data_enable_clarification_true(self):
        """Test setting enable_human_clarification to True."""
        node_data = AgentNodeData(
            agent_strategy_provider_name="test_provider",
            agent_strategy_name="test_strategy",
            agent_strategy_label="Test Strategy",
            agent_parameters={},
            enable_human_clarification=True,
        )

        assert node_data.enable_human_clarification is True

    def test_agent_node_data_serialization(self):
        """Test that enable_human_clarification can be serialized."""
        node_data = AgentNodeData(
            agent_strategy_provider_name="test_provider",
            agent_strategy_name="test_strategy",
            agent_strategy_label="Test Strategy",
            agent_parameters={},
            enable_human_clarification=True,
        )

        # Test model_dump (Pydantic v2)
        dumped = node_data.model_dump()
        assert dumped["enable_human_clarification"] is True

        # Test model_validate (Pydantic v2)
        restored = AgentNodeData.model_validate(dumped)
        assert restored.enable_human_clarification is True

    def test_extract_clarification_request_found(self):
        """Test extracting clarification request from agent output."""
        agent_node = self._create_mock_agent_node()

        json_output = [
            {"id": "log1", "status": "completed"},
            {
                "clarification_needed": True,
                "clarification_prompt": "Please clarify the user's intent",
                "context": {"key": "value"},
            },
        ]

        result = agent_node._extract_clarification_request(json_output)

        assert result is not None
        assert result["clarification_needed"] is True
        assert result["clarification_prompt"] == "Please clarify the user's intent"
        assert result["context"] == {"key": "value"}

    def test_extract_clarification_request_not_found(self):
        """Test when no clarification request is found."""
        agent_node = self._create_mock_agent_node()

        json_output = [
            {"id": "log1", "status": "completed"},
            {"data": "some result"},
        ]

        result = agent_node._extract_clarification_request(json_output)

        assert result is None

    def test_extract_clarification_request_empty_list(self):
        """Test with empty json output."""
        agent_node = self._create_mock_agent_node()

        result = agent_node._extract_clarification_request([])

        assert result is None

    def test_extract_clarification_request_false_flag(self):
        """Test when clarification_needed is False."""
        agent_node = self._create_mock_agent_node()

        json_output = [
            {
                "clarification_needed": False,
                "clarification_prompt": "This should not be detected",
            }
        ]

        result = agent_node._extract_clarification_request(json_output)

        assert result is None

    def test_extract_clarification_request_non_dict_items(self):
        """Test with non-dict items in json output."""
        agent_node = self._create_mock_agent_node()

        json_output = [
            "string item",
            123,
            ["list", "item"],
            {
                "clarification_needed": True,
                "clarification_prompt": "Valid clarification",
            },
        ]

        result = agent_node._extract_clarification_request(json_output)

        assert result is not None
        assert result["clarification_prompt"] == "Valid clarification"

    def test_extract_clarification_request_minimal_data(self):
        """Test with minimal clarification request data."""
        agent_node = self._create_mock_agent_node()

        json_output = [
            {
                "clarification_needed": True,
            }
        ]

        result = agent_node._extract_clarification_request(json_output)

        assert result is not None
        assert result["clarification_needed"] is True

    def test_extract_clarification_request_multiple_items(self):
        """Test that only the first clarification request is returned."""
        agent_node = self._create_mock_agent_node()

        json_output = [
            {
                "clarification_needed": True,
                "clarification_prompt": "First clarification",
            },
            {
                "clarification_needed": True,
                "clarification_prompt": "Second clarification",
            },
        ]

        result = agent_node._extract_clarification_request(json_output)

        assert result is not None
        assert result["clarification_prompt"] == "First clarification"

    # Helper methods

    def _create_mock_agent_node(self) -> AgentNode:
        """Create a mock agent node for testing."""
        node_data = AgentNodeData(
            agent_strategy_provider_name="test_provider",
            agent_strategy_name="test_strategy",
            agent_strategy_label="Test Strategy",
            agent_parameters={},
            enable_human_clarification=True,
        )

        agent_node = MagicMock(spec=AgentNode)
        agent_node.node_data = node_data
        agent_node._node_id = "test_agent_node"

        # Bind the actual method we want to test
        agent_node._extract_clarification_request = (
            AgentNode._extract_clarification_request.__get__(agent_node, AgentNode)
        )

        return agent_node

