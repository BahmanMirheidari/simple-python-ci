
import pytest

from mcp.shared.memory import (
    create_connected_server_and_client_session,
)

from rag.retriever import Retriever
from mcp_server.server import create_mcp_server


class FakeEmbeddingModel:
    """Deterministic embeddings for unit tests."""

    def encode(self, texts):
        import numpy as np
        return np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32),
            (len(texts), 1),
        )

    def encode_query(self, query):
        import numpy as np
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeGenerator:
    """Avoid calling Ollama during tests."""

    def generate(self, question, retrieved_chunks):
        return {
            "answer": f"Test answer for: {question}",
            "sources": list({
                chunk["source"] for chunk in retrieved_chunks
            }),
        }


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mcp_server():
    retriever = Retriever(FakeEmbeddingModel())
    generator = FakeGenerator()

    initial_chunks = [
        {
            "text": "Dementia can affect memory and communication.",
            "source": "dementia.txt",
        },
        {
            "text": "Alzheimer's disease is a type of dementia.",
            "source": "alzheimers.txt",
        },
    ]

    return create_mcp_server(
        retriever=retriever,
        generator=generator,
        initial_chunks=initial_chunks,
    )

@pytest.fixture
async def mcp_session(mcp_server):
    async with create_connected_server_and_client_session(
        mcp_server
    ) as session:
        yield session


@pytest.mark.anyio
async def test_mcp_tools_are_registered(mcp_session):
    result = await mcp_session.list_tools()

    names = {tool.name for tool in result.tools}

    assert "search_documents" in names
    assert "ask_question" in names
    assert "list_documents" in names
    assert "ingest_document" in names


@pytest.mark.anyio
async def test_list_documents(mcp_session):
    result = await mcp_session.call_tool(
        "list_documents",
        {},
    )

    assert not result.isError

    output = result.content[0].text

    assert "dementia.txt" in output
    assert "alzheimers.txt" in output


@pytest.mark.anyio
async def test_search_documents(mcp_session):
    result = await mcp_session.call_tool(
        "search_documents",
        {
            "query": "What is dementia?",
            "top_k": 2,
        },
    )

    assert not result.isError
    assert "dementia" in result.content[0].text.lower()


@pytest.mark.anyio
async def test_ask_question(mcp_session):
    result = await mcp_session.call_tool(
        "ask_question",
        {
            "question": "What does dementia affect?",
        },
    )

    assert not result.isError
    assert "Test answer" in result.content[0].text


@pytest.mark.anyio
async def test_ingest_document(mcp_session):
    result = await mcp_session.call_tool(
        "ingest_document",
        {
            "text": "Speech difficulties can occur in some neurological conditions.",
            "source": "speech.txt",
        },
    )

    assert not result.isError
    assert "speech.txt" in result.content[0].text

    documents = await mcp_session.call_tool(
        "list_documents",
        {},
    )

    assert "speech.txt" in documents.content[0].text


@pytest.mark.anyio
async def test_empty_search_query_returns_error(mcp_session):
    result = await mcp_session.call_tool(
        "search_documents",
        {
            "query": "   ",
            "top_k": 3,
        },
    )

    assert result.isError
