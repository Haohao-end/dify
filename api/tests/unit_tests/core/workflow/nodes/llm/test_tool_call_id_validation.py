"""Unit tests for tool_call_id validation in LLM node resume logic."""

import pytest

from dify_graph.model_runtime.entities.message_entities import AssistantPromptMessage, ToolPromptMessage
from dify_graph.nodes.llm.exc import InvalidToolCallIdError


class MockToolResult:
    """Mock tool result for testing."""

    def __init__(self, tool_call_id: str, output: str):
        self.tool_call_id = tool_call_id
        self.output = output


def test_valid_tool_call_ids():
    """Test normal case: all tool_call_ids match expected."""
    from dify_graph.nodes.llm.node import LLMNode

    assistant_msg = AssistantPromptMessage(
        content="",
        tool_calls=[
            AssistantPromptMessage.ToolCall(
                id="call_123",
                type="function",
                function=AssistantPromptMessage.ToolCall.ToolCallFunction(name="get_weather", arguments="{}"),
            ),
            AssistantPromptMessage.ToolCall(
                id="call_456",
                type="function",
                function=AssistantPromptMessage.ToolCall.ToolCallFunction(name="get_time", arguments="{}"),
            ),
        ],
    )

    state = {
        "round": 1,
        "stop": [],
        "tool_call_history": [{"round": 1, "tool_calls": [], "tool_results": []}],
        "prompt_messages": [],
    }

    original_deserialize = LLMNode._deserialize_prompt_messages
    LLMNode._deserialize_prompt_messages = staticmethod(lambda msgs: [assistant_msg])

    try:
        node = type(
            "MockNode",
            (),
            {
                "_node_id": "test_node",
                "_resume_from_tool_call_state": LLMNode._resume_from_tool_call_state.__get__(
                    type("obj", (), {"_node_id": "test_node"})()
                ),
            },
        )()

        tool_results = [
            MockToolResult(tool_call_id="call_123", output="result1"),
            MockToolResult(tool_call_id="call_456", output="result2"),
        ]

        prompt_messages, stop, history, round_num = node._resume_from_tool_call_state(state, None, tool_results)

        assert len(prompt_messages) == 3
        assert isinstance(prompt_messages[1], ToolPromptMessage)
        assert isinstance(prompt_messages[2], ToolPromptMessage)

    finally:
        LLMNode._deserialize_prompt_messages = original_deserialize


def test_unexpected_tool_call_id():
    """Test error case: tool_results contain unexpected tool_call_id."""
    from dify_graph.nodes.llm.node import LLMNode

    assistant_msg = AssistantPromptMessage(
        content="",
        tool_calls=[
            AssistantPromptMessage.ToolCall(
                id="call_123",
                type="function",
                function=AssistantPromptMessage.ToolCall.ToolCallFunction(name="get_weather", arguments="{}"),
            )
        ],
    )

    state = {
        "round": 1,
        "stop": [],
        "tool_call_history": [],
        "prompt_messages": [],
    }

    original_deserialize = LLMNode._deserialize_prompt_messages
    LLMNode._deserialize_prompt_messages = staticmethod(lambda msgs: [assistant_msg])

    try:
        node = type(
            "MockNode",
            (),
            {
                "_node_id": "test_node",
                "_resume_from_tool_call_state": LLMNode._resume_from_tool_call_state.__get__(
                    type("obj", (), {"_node_id": "test_node"})()
                ),
            },
        )()

        tool_results = [
            MockToolResult(tool_call_id="call_123", output="valid"),
            MockToolResult(tool_call_id="call_999", output="invalid"),
        ]

        with pytest.raises(InvalidToolCallIdError) as exc_info:
            node._resume_from_tool_call_state(state, None, tool_results)

        assert "call_999" in str(exc_info.value)
        assert "Invalid tool_call_id" in str(exc_info.value)

    finally:
        LLMNode._deserialize_prompt_messages = original_deserialize


def test_empty_tool_call_ids():
    """Test error case: no valid tool_call_ids in tool_results."""
    from dify_graph.nodes.llm.node import LLMNode

    assistant_msg = AssistantPromptMessage(
        content="",
        tool_calls=[
            AssistantPromptMessage.ToolCall(
                id="call_123",
                type="function",
                function=AssistantPromptMessage.ToolCall.ToolCallFunction(name="get_weather", arguments="{}"),
            )
        ],
    )

    state = {
        "round": 1,
        "stop": [],
        "tool_call_history": [],
        "prompt_messages": [],
    }

    original_deserialize = LLMNode._deserialize_prompt_messages
    LLMNode._deserialize_prompt_messages = staticmethod(lambda msgs: [assistant_msg])

    try:
        node = type(
            "MockNode",
            (),
            {
                "_node_id": "test_node",
                "_resume_from_tool_call_state": LLMNode._resume_from_tool_call_state.__get__(
                    type("obj", (), {"_node_id": "test_node"})()
                ),
            },
        )()

        # tool_results with invalid structure (no valid tool_call_id)
        tool_results = [{"output": "result"}]  # Missing tool_call_id

        with pytest.raises(InvalidToolCallIdError) as exc_info:
            node._resume_from_tool_call_state(state, None, tool_results)

        assert "No tool_call_ids found" in str(exc_info.value)

    finally:
        LLMNode._deserialize_prompt_messages = original_deserialize
