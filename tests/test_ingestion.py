from rag.ingestion import (
    load_documents,
    split_text,
    create_chunks
)


def test_split_text():
    text = "one two three four five six"

    chunks = split_text(
        text,
        chunk_size=3,
        overlap=1
    )

    assert len(chunks) == 3
    assert chunks[0] == "one two three"
    assert chunks[1] == "three four five"


def test_load_documents(tmp_path):
    file = tmp_path / "sample.txt"

    file.write_text(
        "This is a test document.",
        encoding="utf-8"
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0]["source"] == "sample.txt"


def test_create_chunks():
    documents = [{
        "text": "one two three four five",
        "source": "sample.txt"
    }]

    chunks = create_chunks(documents)

    assert len(chunks) == 1
    assert chunks[0]["source"] == "sample.txt"