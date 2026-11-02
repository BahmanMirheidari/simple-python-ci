from pathlib import Path


def load_documents(directory):
    documents = []

    for file_path in Path(directory).glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "text": text,
            "source": file_path.name
        })

    return documents


def split_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks(documents):
    chunks = []

    for document in documents:
        text_chunks = split_text(document["text"])

        for chunk in text_chunks:
            chunks.append({
                "text": chunk,
                "source": document["source"]
            })

    return chunks