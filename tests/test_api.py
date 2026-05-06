from fastapi.testclient import TestClient

from src.api.main import app


def test_api_health_endpoint_works():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "models" in payload
