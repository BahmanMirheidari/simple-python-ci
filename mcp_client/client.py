
import asyncio
import json
import os
import sys
from pathlib import Path

import requests

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Project configuration
ROOT = Path(__file__).resolve().parents[1]

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:8b"
)

MAX_TOOL_CALLS = 5


def convert_mcp_tools(mcp_tools):
    """Convert MCP tool definitions into Ollama tool definitions."""

    ollama_tools = []

    for tool in mcp_tools:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema
            }
        })

    return ollama_tools


def ollama_chat(messages, tools):
    """Send a chat request to the local Ollama API."""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    return response.json()["message"]


def mcp_result_to_text(result):
    """Convert an MCP tool result into text."""

    if result.structuredContent is not None:
        return json.dumps(result.structuredContent)

    output = "\n".join(
        item.text
        for item in result.content
        if hasattr(item, "text")
    )

    if result.isError:
        return f"Tool error: {output}"

    return output


async def execute_tool(session, tool_name, arguments, allowed_tools):
    """Execute a tool through the MCP server."""

    if tool_name not in allowed_tools:
        return f"Error: Unknown MCP tool '{tool_name}'"

    try:
        result = await session.call_tool(
            tool_name,
            arguments
        )

        return mcp_result_to_text(result)

    except Exception as exc:
        return f"Tool execution failed: {type(exc).__name__}: {exc}"


async def ask_ollama(session, available_tools, question):
    """
    Let Ollama decide which MCP tools to execute.

    The tool definitions are discovered dynamically from MCP.
    """

    tools = convert_mcp_tools(available_tools)

    allowed_tools = {
        tool.name for tool in available_tools
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a local RAG assistant. "
                "Use the available tools when they help answer "
                "the user's question. "
                "Base answers on retrieved document content. "
                "Do not invent document contents. "
                "If the information is unavailable, say so."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for iteration in range(MAX_TOOL_CALLS):

        print(f"\n[Ollama interaction {iteration + 1}]")

        message = ollama_chat(messages, tools)

        tool_calls = message.get("tool_calls", [])

        # No more tool calls: return the final answer.
        if not tool_calls:
            return message.get("content", "")

        # Preserve the assistant's tool-call request.
        messages.append({
            "role": "assistant",
            "content": message.get("content", ""),
            "tool_calls": tool_calls
        })

        # Execute requested MCP tools.
        for tool_call in tool_calls:

            function = tool_call["function"]

            tool_name = function["name"]
            arguments = function.get("arguments", {})

            print(f"[MCP] Calling tool: {tool_name}")
            print(f"[MCP] Arguments: {arguments}")

            output = await execute_tool(
                session,
                tool_name,
                arguments,
                allowed_tools
            )

            print(f"[MCP] Tool completed: {tool_name}")

            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": output
            })

    return (
        "Maximum tool-call limit reached. "
        "Please try a simpler question."
    )


async def main():

    # Start the MCP server as a subprocess.
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            str(ROOT / "mcp_server" / "server.py")
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT)
        }
    )

    print("Starting MCP server...")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Ollama model: {OLLAMA_MODEL}")

    async with stdio_client(server_parameters) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialise MCP connection.
            await session.initialize()

            # Discover available MCP tools.
            tool_result = await session.list_tools()

            available_tools = tool_result.tools

            print("\nAvailable MCP tools:")

            for tool in available_tools:
                print(f"- {tool.name}: {tool.description}")

            print("\nLocal Ollama RAG Assistant")
            print("Type 'exit' to quit.\n")

            while True:

                question = input("You: ").strip()

                if question.lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break

                if not question:
                    continue

                try:

                    answer = await ask_ollama(
                        session,
                        available_tools,
                        question
                    )

                    print(f"\nAssistant: {answer}\n")

                except requests.RequestException as exc:
                    print(f"\nOllama connection error: {exc}\n")

                except Exception as exc:
                    print(
                        f"\nError: {type(exc).__name__}: {exc}\n"
                    )


if __name__ == "__main__":
    asyncio.run(main())
