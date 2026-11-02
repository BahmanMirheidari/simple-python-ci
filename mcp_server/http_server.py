
import os
from pathlib import Path

from mcp_server.server import create_mcp_server
from rag.ingestion import load_documents, create_chunks


def main():
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
        initial_chunks=initial_chunks
    )

    print("Starting MCP HTTP server...", flush=True)

    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
