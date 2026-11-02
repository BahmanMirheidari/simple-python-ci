import numpy as np


class Retriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.embeddings = None
        self.documents = []

    def build_index(self, chunks):
        if not chunks:
            raise ValueError("Cannot build index from empty chunks")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = np.asarray(
            self.embedding_model.encode(texts),
            dtype=np.float32
        )

        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("Invalid embeddings shape")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        if np.any(norms == 0):
            raise ValueError("Embeddings cannot contain zero vectors")

        self.embeddings = embeddings / norms
        self.documents = chunks

    def search(self, query, top_k=3):
        if self.embeddings is None:
            raise RuntimeError("Index has not been built")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_embedding = np.asarray(
            self.embedding_model.encode_query(query),
            dtype=np.float32
        )

        if query_embedding.shape != (self.embeddings.shape[1],):
            raise ValueError("Query embedding has incorrect dimensions")

        norm = np.linalg.norm(query_embedding)

        if norm == 0:
            raise ValueError("Query embedding cannot be a zero vector")

        query_embedding /= norm

        scores = self.embeddings @ query_embedding
        indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "text": self.documents[i]["text"],
                "source": self.documents[i]["source"],
                "score": float(scores[i])
            }
            for i in indices
        ]