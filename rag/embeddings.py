
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Convert a list of texts into numerical embeddings.
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def encode_query(self, query):
        """
        Convert a single query into an embedding.
        """
        return self.encode([query])[0]
