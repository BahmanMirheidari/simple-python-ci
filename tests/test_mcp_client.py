
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_client.client import (
    convert_mcp_tools,
    ollama_chat,
    mcp_result_to_text,
    execute_tool,
    ask_ollama,
)


# --------------------------------------------------
# Test MCP tool conversion
# --------------------------------------------------

def test_convert_mcp_tools():

    mcp_tools = [
        SimpleNamespace(
            name="search_documents",
            description="Search the knowledge base",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
            },
        )
    ]

    result = convert_mcp_tools(mcp_tools)

    assert len(result) == 1
    assert result[0]["type"] == "function"

    function = result[0]["function"]

    assert function["name"] == "search_documents"
    assert function["description"] == "Search the knowledge base"
    assert "query" in function["parameters"]["properties"]


# --------------------------------------------------
# Test Ollama HTTP request
# --------------------------------------------------

@patch("mcp_client.client.requests.post")
def test_ollama_chat(mock_post):

    mock_response = Mock()

    mock_response.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "Hello"
        }
    }

    mock_post.return_value = mock_response

    result = ollama_chat(
        [{"role": "user", "content": "Hello"}],
        []
    )

    assert result["content"] == "Hello"

    mock_post.assert_called_once()

    args, kwargs = mock_post.call_args

    assert args[0].endswith("/api/chat")
    assert kwargs["json"]["model"]
    assert kwargs["json"]["stream"] is False

    mock_response.raise_for_status.assert_called_once()


# --------------------------------------------------
# Test MCP result conversion
# --------------------------------------------------

def test_mcp_result_to_text():

    result = SimpleNamespace(
        structuredContent=None,
        content=[
            SimpleNamespace(text="First result"),
            SimpleNamespace(text="Second result"),
        ],
        isError=False,
    )

    output = mcp_result_to_text(result)

    assert "First result" in output
    assert "Second result" in output


def test_mcp_structured_result_to_text():

    result = SimpleNamespace(
        structuredContent={"answer": "Test answer"},
        content=[],
        isError=False,
    )

    output = mcp_result_to_text(result)

    assert '"answer": "Test answer"' in output


# --------------------------------------------------
# Test MCP tool execution
# --------------------------------------------------

def test_execute_tool():

    async def run_test():

        session = Mock()

        session.call_tool = AsyncMock(
            return_value=SimpleNamespace(
                structuredContent=None,
                content=[
                    SimpleNamespace(text="Found document")
                ],
                isError=False,
            )
        )

        output = await execute_tool(
            session,
            "search_documents",
            {"query": "dementia"},
            {"search_documents"},
        )

        assert output == "Found document"

        session.call_tool.assert_awaited_once_with(
            "search_documents",
            {"query": "dementia"},
        )

    asyncio.run(run_test())


def test_execute_unknown_tool():

    async def run_test():

        session = Mock()

        output = await execute_tool(
            session,
            "unknown_tool",
            {},
            {"search_documents"},
        )

        assert "Unknown MCP tool" in output

        session.call_tool.assert_not_called()

    asyncio.run(run_test())


# --------------------------------------------------
# Test Ollama without tool calls
# --------------------------------------------------

def test_ask_ollama_direct_answer():

    async def run_test():

        session = Mock()

        tools = [
            SimpleNamespace(
                name="search_documents",
                description="Search documents",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

        with patch(
            "mcp_client.client.ollama_chat",
            return_value={
                "content": "This is a direct answer.",
                "tool_calls": [],
            },
        ):

            answer = await ask_ollama(
                session,
                tools,
                "Hello",
            )

        assert answer == "This is a direct answer."

        session.call_tool.assert_not_called()

    asyncio.run(run_test())


# --------------------------------------------------
# Test complete MCP tool-calling workflow
# --------------------------------------------------

def test_ask_ollama_calls_mcp_tool():

    async def run_test():

        session = Mock()

        session.call_tool = AsyncMock(
            return_value=SimpleNamespace(
                structuredContent=None,
                content=[
                    SimpleNamespace(
                        text="Dementia can affect memory."
                    )
                ],
                isError=False,
            )
        )

        tools = [
            SimpleNamespace(
                name="search_documents",
                description="Search documents",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                },
            )
        ]

        responses = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_documents",
                            "arguments": {
                                "query": "dementia"
                            },
                        }
                    }
                ],
            },
            {
                "content": (
                    "Dementia can affect memory."
                ),
                "tool_calls": [],
            },
        ]

        with patch(
            "mcp_client.client.ollama_chat",
            side_effect=responses,
        ) as mock_chat:

            answer = await ask_ollama(
                session,
                tools,
                "What does dementia affect?",
            )

        assert "memory" in answer.lower()

        session.call_tool.assert_awaited_once_with(
            "search_documents",
            {"query": "dementia"},
        )

        # One call to select the tool, another for the final answer.
        assert mock_chat.call_count == 2

    asyncio.run(run_test())
