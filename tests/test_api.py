from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import feature_names

client = TestClient(app)


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Loan Default Risk API" in response.json()["message"]


def test_predict_valid_input():
    dummy_input = [0.0] * len(feature_names)

    response = client.post(
        "/api/v1/predict",
        json={"features": dummy_input}
    )

    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "prediction" in data


def test_predict_invalid_length():
    dummy_input = [0.0] * (len(feature_names) - 1)

    response = client.post(
        "/api/v1/predict",
        json={"features": dummy_input}
    )

    assert response.status_code == 422


def test_predict_nan():
    
    payload = {
        "features": [None] * 210
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422