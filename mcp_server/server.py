
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from rag.embeddings import EmbeddingModel
from rag.retriever import Retriever
from rag.generator import RAGGenerator
from rag.ingestion import load_documents, create_chunks


def create_mcp_server(
    retriever=None,
    generator=None,
    initial_chunks=None,
):
    """Create and configure the local RAG MCP server."""

    mcp = FastMCP("Local RAG Assistant")

    # Initialise the embedding model and retriever.
    if retriever is None:
        embedding_model = EmbeddingModel()
        retriever = Retriever(embedding_model)

    # Initialise the LLM generator.
    if generator is None:
        generator = RAGGenerator()

    # Keep track of all indexed document chunks.
    chunks = list(initial_chunks or [])

    # Build the initial index if documents were supplied.
    if chunks:
        retriever.build_index(chunks)

    # ---------------------------------------------------------
    # Tool 1: Search documents
    # ---------------------------------------------------------

    @mcp.tool()
    def search_documents(query: str, top_k: int = 3) -> list:
        """Search the knowledge base and return relevant document chunks."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")

        if not chunks:
            return []

        return retriever.search(query, top_k=top_k)

    # ---------------------------------------------------------
    # Tool 2: Ask a question using RAG
    # ---------------------------------------------------------

    @mcp.tool()
    def ask_question(question: str) -> dict:
        """Answer a question using retrieved knowledge-base documents."""

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not chunks:
            return {
                "answer": "No documents have been indexed.",
                "sources": [],
            }

        retrieved_chunks = retriever.search(question, top_k=3)

        if not retrieved_chunks:
            return {
                "answer": "No relevant documents were found.",
                "sources": [],
            }

        return generator.generate(question, retrieved_chunks)

    # ---------------------------------------------------------
    # Tool 3: List indexed documents
    # ---------------------------------------------------------

    @mcp.tool()
    def list_documents() -> str:
        """List the document sources currently in the knowledge base."""

        sources = sorted({
            chunk["source"]
            for chunk in chunks
        })

        if not sources:
            return "No documents have been indexed."

        return "\n".join(sources)

    # ---------------------------------------------------------
    # Tool 4: Ingest a new document
    # ---------------------------------------------------------

    @mcp.tool()
    def ingest_document(text: str, source: str) -> dict:
        """Add a text document to the knowledge base."""

        if not text.strip():
            raise ValueError("Document text cannot be empty")

        if not source.strip():
            raise ValueError("Document source cannot be empty")

        # Only accept a filename, not a filesystem path.
        if Path(source).name != source or source in {".", ".."}:
            raise ValueError(
                "Source must be a filename, not a path"
            )

        new_documents = [
            {
                "text": text,
                "source": source,
            }
        ]

        new_chunks = create_chunks(new_documents)

        if not new_chunks:
            raise ValueError("No document chunks were created")

        # Build the updated index before modifying the current list.
        updated_chunks = chunks + new_chunks
        retriever.build_index(updated_chunks)

        # Update the document list only after successful indexing.
        chunks.extend(new_chunks)

        return {
            "message": "Document ingested successfully",
            "source": source,
            "chunks_added": len(new_chunks),
            "total_chunks": len(chunks),
        }

    # IMPORTANT: Return the FastMCP server instance.
    return mcp


def main():
    """Load initial documents and start the MCP server."""

    project_root = Path(__file__).resolve().parents[1]

    documents_directory = Path(
        os.getenv(
            "RAG_DOCUMENTS_DIR",
            str(project_root / "documents"),
        )
    )

    documents = load_documents(documents_directory)
    initial_chunks = create_chunks(documents)

    server = create_mcp_server(
        initial_chunks=initial_chunks,
    )

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
