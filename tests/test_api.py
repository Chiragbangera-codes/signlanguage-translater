import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add workspace root and backend directory to sys.path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_root = os.path.join(workspace_root, "backend")
if workspace_root not in sys.path:
    sys.path.append(workspace_root)
if backend_root not in sys.path:
    sys.path.append(backend_root)

# Import app inside backend
from backend.app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Fixture that provides a FastAPI TestClient with lifespan events executed."""
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """Tests the /api/v1/health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert json_data["model_loaded"] is True
    assert json_data["version"] == "1.0.0"
    assert json_data["tensorflow"] == "2.x"

def test_root_portal(client):
    """Tests the / root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_invalid_prediction_request_empty(client):
    """Tests /api/v1/predict with empty payload."""
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 400
    json_data = response.json()
    assert json_data["error"] == "INVALID_INPUT"
    assert "Validation failed" in json_data["message"]

def test_invalid_prediction_request_wrong_feature_size(client):
    """Tests /api/v1/predict with wrong array length for hand coordinates."""
    bad_payload = {
        "left_hand": [0.0] * 10,  # Should be 63
        "right_hand": [-1.0] * 63,
        "uses_two_hands": 0.0
    }
    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 400
    json_data = response.json()
    assert json_data["error"] == "INVALID_INPUT"
    assert "Validation failed at 'left_hand'" in json_data["message"]

def test_prediction_endpoint_success(client):
    """Tests /api/v1/predict with valid payload (mock single hand data)."""
    # 63 mock coordinates for left hand (wrist at 0.0, others normalized),
    # 63 padded coordinates (-1.0) for right hand
    mock_left_hand = [0.0] * 63
    mock_left_hand[0:3] = [0.0, 0.0, 0.0] # Wrist index 0
    mock_left_hand[15:18] = [0.1, 0.2, 0.3] # Landmark indices

    mock_right_hand = [-1.0] * 63

    payload = {
        "left_hand": mock_left_hand,
        "right_hand": mock_right_hand,
        "uses_two_hands": 0.0
    }

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert "confidence" in json_data
    assert "processing_time_ms" in json_data
    assert isinstance(json_data["processing_time_ms"], float)
    assert "top_predictions" in json_data
    assert len(json_data["top_predictions"]) == 3

    # Check that predictions are valid alphabet letters A-Z
    assert len(json_data["prediction"]) == 1
    assert json_data["prediction"].isupper()


    for item in json_data["top_predictions"]:
        assert "label" in item
        assert "confidence" in item
        assert len(item["label"]) == 1
        assert 0.0 <= item["confidence"] <= 100.0
