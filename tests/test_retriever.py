
import numpy as np
import pytest

from rag.retriever import Retriever


class MockEmbeddingModel:

    def encode(self, texts):

        vectors = {
            "Memory assessment": [1.0, 0.0, 0.0],
            "Speech assessment": [0.0, 1.0, 0.0],
            "Cognitive decline": [0.0, 0.0, 1.0]
        }

        return np.array(
            [vectors[text] for text in texts],
            dtype=np.float32
        )

    def encode_query(self, query):

        if query == "memory":
            return np.array([1.0, 0.0, 0.0])

        if query == "speech":
            return np.array([0.0, 1.0, 0.0])

        return np.array([0.0, 0.0, 1.0])


@pytest.fixture
def retriever():

    model = MockEmbeddingModel()

    instance = Retriever(model)

    chunks = [
        {
            "text": "Memory assessment",
            "source": "memory.pdf"
        },
        {
            "text": "Speech assessment",
            "source": "speech.pdf"
        },
        {
            "text": "Cognitive decline",
            "source": "cognitive.pdf"
        }
    ]

    instance.build_index(chunks)

    return instance


def test_retrieve_most_similar_document(retriever):

    results = retriever.search("memory", top_k=1)

    assert len(results) == 1

    assert results[0]["text"] == "Memory assessment"

    assert results[0]["source"] == "memory.pdf"

    assert results[0]["score"] == pytest.approx(1.0)


def test_retrieve_multiple_documents(retriever):

    results = retriever.search("memory", top_k=2)

    assert len(results) == 2

    assert results[0]["text"] == "Memory assessment"


def test_empty_index():

    retriever = Retriever(MockEmbeddingModel())

    with pytest.raises(RuntimeError):
        retriever.search("memory")


def test_invalid_top_k(retriever):

    with pytest.raises(ValueError):
        retriever.search("memory", top_k=0)


def test_empty_chunks():

    retriever = Retriever(MockEmbeddingModel())

    with pytest.raises(ValueError):
        retriever.build_index([])
