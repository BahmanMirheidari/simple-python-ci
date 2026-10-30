
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


@app.get("/subtract")
def add(a: float, b: float):
    return {"a": a, "b": b, "result": a - b}


@app.get("/divide")
def divide(a: float, b: float):
    if b == 0:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Cannot divide by zero"
        )
    return {"a": a, "b": b, "result": a / b}
