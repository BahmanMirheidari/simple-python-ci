
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == (
        "Hello from my FastAPI service!"
    )


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_add():
    response = client.get("/add?a=5&b=3")
    assert response.status_code == 200
    assert response.json()["result"] == 8
