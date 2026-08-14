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

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_frame(use_two_hands: float = 0.0, valid_left: bool = True) -> list[float]:
    """Build one 127-feature frame.

    Frame layout: [uses_two_hands, left_63, right_63]
    Valid left-hand: wrist at origin (0,0,0) with small non-zero offsets
    for other landmarks so normalize_hand() can compute a non-zero scale.
    Absent hand: all coordinates set to -1.0 (sentinel).
    """
    if valid_left:
        # Wrist at (0,0,0), some landmarks with small non-zero offsets
        left = [0.0] * 63
        left[3:6] = [0.05, 0.10, 0.02]   # index finger base
        left[6:9] = [0.08, 0.18, 0.03]   # index mid
        left[9:12] = [0.06, 0.25, 0.01]  # index tip
    else:
        left = [-1.0] * 63

    right = [-1.0] * 63  # right hand always absent in single-hand tests
    return [use_two_hands] + left + right


def _make_sequence(num_valid_frames: int = 30) -> list[list[float]]:
    """Return a 30-frame sequence, the last `num_valid_frames` having a valid left hand."""
    sequence = []
    empty_frame = _make_frame(valid_left=False)
    valid_frame = _make_frame(valid_left=True)
    padding = 30 - num_valid_frames
    sequence.extend([empty_frame] * padding)
    sequence.extend([valid_frame] * num_valid_frames)
    assert len(sequence) == 30
    return sequence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Provides a FastAPI TestClient with lifespan events executed."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_check(client):
    """GET /api/v1/health returns 200 with expected fields."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert json_data["model_loaded"] is True
    assert json_data["version"] == "1.0.0"
    assert json_data["tensorflow"] == "2.x"


def test_root_portal(client):
    """GET / returns 200 with welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


def test_predict_empty_payload(client):
    """POST /api/v1/predict with empty body returns 400 INVALID_INPUT."""
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 400
    json_data = response.json()
    assert json_data["error"] == "INVALID_INPUT"
    assert "Validation failed" in json_data["message"]


def test_predict_wrong_sequence_length_short(client):
    """POST /api/v1/predict with fewer than 30 frames returns 400."""
    short_sequence = _make_sequence(num_valid_frames=10)[:10]  # only 10 frames
    payload = {"sequence": short_sequence, "mode": "numbers"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    json_data = response.json()
    assert json_data["error"] == "INVALID_INPUT"


def test_predict_wrong_frame_feature_size(client):
    """POST /api/v1/predict with wrong feature count per frame returns 400."""
    bad_frame = [0.0] * 10  # Should be 127
    bad_sequence = [bad_frame] * 30
    payload = {"sequence": bad_sequence, "mode": "numbers"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    json_data = response.json()
    assert json_data["error"] == "INVALID_INPUT"
    assert "Validation failed" in json_data["message"]


def test_predict_no_hand_detected(client):
    """POST /api/v1/predict with all frames having absent hands returns 422."""
    all_empty = _make_sequence(num_valid_frames=0)
    payload = {"sequence": all_empty, "mode": "numbers"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    json_data = response.json()
    # 422 is returned from HTTPException via the custom handler -> SERVER_ERROR code
    # but the message should mention hand detection
    assert "hand" in json_data.get("message", "").lower()


def test_predict_success_numbers(client):
    """POST /api/v1/predict with a valid 30×127 sequence returns 200 with correct structure."""
    payload = {"sequence": _make_sequence(num_valid_frames=30), "mode": "numbers"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200

    json_data = response.json()

    # Required response fields
    assert "prediction" in json_data
    assert "confidence" in json_data
    assert "processing_time_ms" in json_data
    assert "top_predictions" in json_data

    # Type checks
    assert isinstance(json_data["prediction"], str)
    assert isinstance(json_data["confidence"], float)
    assert isinstance(json_data["processing_time_ms"], float)
    assert json_data["processing_time_ms"] > 0.0

    # The numbers model returns digit/sign labels (strings like "0"–"25")
    assert len(json_data["prediction"]) >= 1

    # Confidence is a percentage 0–100
    assert 0.0 <= json_data["confidence"] <= 100.0

    # top_predictions: exactly 3 entries
    assert isinstance(json_data["top_predictions"], list)
    assert len(json_data["top_predictions"]) == 3

    for item in json_data["top_predictions"]:
        assert "label" in item
        assert "confidence" in item
        assert isinstance(item["label"], str)
        assert 0.0 <= item["confidence"] <= 100.0


def test_predict_success_only_last_valid_frame(client):
    """POST /api/v1/predict succeeds even if only the last frame has a hand."""
    # 29 empty frames followed by 1 valid frame
    payload = {"sequence": _make_sequence(num_valid_frames=1), "mode": "numbers"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_unknown_mode(client):
    """POST /api/v1/predict with an unrecognised mode returns 400."""
    payload = {"sequence": _make_sequence(), "mode": "invalid_mode"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
