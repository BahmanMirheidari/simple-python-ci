
import pytest
from unittest.mock import patch, Mock

from rag.generator import RAGGenerator


@pytest.fixture
def generator():
    return RAGGenerator(model="qwen3:8b")


@pytest.fixture
def contexts():
    return [
        {
            "text": "Dementia affects cognitive function.",
            "source": "dementia.pdf",
            "score": 0.91
        },
        {
            "text": "Memory assessment evaluates recall.",
            "source": "assessment.pdf",
            "score": 0.85
        }
    ]


def test_build_prompt(generator, contexts):

    prompt = generator.build_prompt(
        "What is dementia?",
        contexts
    )

    assert "What is dementia?" in prompt
    assert "Dementia affects cognitive function." in prompt
    assert "dementia.pdf" in prompt


def test_generate_success(generator, contexts):

    mock_response = Mock()

    mock_response.json.return_value = {
        "response": "Dementia affects cognitive function."
    }

    mock_response.raise_for_status.return_value = None

    with patch(
        "rag.generator.requests.post",
        return_value=mock_response
    ) as mock_post:

        result = generator.generate(
            "What is dementia?",
            contexts
        )

    assert result["answer"] == (
        "Dementia affects cognitive function."
    )

    assert result["sources"] == [
        "dementia.pdf",
        "assessment.pdf"
    ]

    mock_post.assert_called_once()


def test_generate_without_context(generator):

    result = generator.generate(
        "What is dementia?",
        []
    )

    assert result["answer"] == (
        "No relevant documents were retrieved."
    )

    assert result["sources"] == []


def test_empty_question(generator, contexts):

    with pytest.raises(ValueError):
        generator.generate("   ", contexts)


def test_ollama_connection_error(generator, contexts):

    with patch(
        "rag.generator.requests.post",
        side_effect=Exception("Connection failed")
    ):

        with pytest.raises(Exception):
            generator.generate(
                "What is dementia?",
                contexts
            )
