
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

from rag.ingestion import create_chunks
from rag.embeddings import EmbeddingModel
from rag.retriever import Retriever
from rag.generator import RAGGenerator

app = FastAPI(title="Calculator and RAG API", version="2.0.0")

# Initialise RAG components
embedding_model = EmbeddingModel()
retriever = Retriever(embedding_model)
generator = RAGGenerator()

# Store ingested chunks in memory
documents = []


class DocumentRequest(BaseModel):
    documents: list[dict]


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
def home():
    return {"message": "Hello from my FastAPI service!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/add")
def add(a: float, b: float):
    return {"a": a, "b": b, "result": a + b}


@app.get("/subtract")
def subtract(a: float, b: float):
    return {"a": a, "b": b, "result": a - b}


@app.get("/divide")
def divide(a: float, b: float):
    if b == 0:
        raise HTTPException(status_code=400, detail="Cannot divide by zero")
    return {"a": a, "b": b, "result": a / b}


@app.post("/ingest")
def ingest(request: DocumentRequest):
    global documents

    try:
        chunks = create_chunks(request.documents)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No document chunks were created"
            )

        retriever.build_index(chunks)
        documents = chunks

        return {
            "message": "Documents ingested successfully",
            "documents": len(request.documents),
            "chunks": len(chunks)
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query")
def query(request: QueryRequest):

    if not documents:
        raise HTTPException(
            status_code=400,
            detail="Please ingest documents first"
        )

    try:
        results = retriever.search(
            request.question,
            top_k=request.top_k
        )

        response = generator.generate(
            request.question,
            results
        )

        return {
            "answer": response["answer"],
            "sources": response["sources"],
            "retrieved_chunks": results
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
