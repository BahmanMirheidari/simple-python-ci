
from fastapi import FastAPI

app = FastAPI(title="My First API", version="1.0.0")


@app.get("/")
def home():
    return {"message": "Hello from my FastAPI service!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/add")
def add(a: float, b: float):
    return {"a": a, "b": b, "result": a + b}
