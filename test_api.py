
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

def test_subtract():
    response = client.get("/subtract?a=5&b=3")
    assert response.status_code == 200
    assert response.json()["result"] == 2

def test_divide():
    response = client.get("/divide?a=6&b=2")
    assert response.status_code == 200
    assert response.json()["result"] == 3

def test_divide_by_zero():
    response = client.get("/divide?a=10&b=0")

    assert response.status_code == 400
    assert response.json()["detail"] == "Cannot divide by zero"
