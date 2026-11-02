import requests
import numpy as np


class EmbeddingModel:
    def __init__(
        self,
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
        timeout=120
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def encode(self, texts):
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model_name,
                "input": texts
            },
            timeout=self.timeout
        )
        response.raise_for_status()

        embeddings = response.json()["embeddings"]
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query):
        if not query.strip():
            raise ValueError("Query cannot be empty")

        return self.encode([query])[0]