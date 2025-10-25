import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from splitter.splitter import app  # Import app from splitter/splitter.py
  # Import your FastAPI app

client = TestClient(app)


# ----------------------------
# 1️⃣ Health endpoint test
# ----------------------------
def test_splitter_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ----------------------------
# 2️⃣ Split endpoint test
# ----------------------------
@pytest.fixture
def small_matrices():
    """Creates two small test matrices as in-memory .npy buffers."""
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    bufA, bufB = io.BytesIO(), io.BytesIO()
    np.save(bufA, A)
    np.save(bufB, B)
    bufA.seek(0)
    bufB.seek(0)
    return bufA, bufB


@patch("splitter.splitter.requests.post")
def test_splitter_split_endpoint(mock_post, small_matrices):
    """
    Unit test for /split endpoint.
    Mocks network requests to aggregator and worker services.
    """
    bufA, bufB = small_matrices

    # Mock all requests.post calls to return success
    class DummyResponse:
        def __init__(self, status_code=200):
            self.status_code = status_code
        def json(self):
            return {"message": "ok"}

    mock_post.return_value = DummyResponse(200)

    files = {
        "A_file": ("A.npy", bufA, "application/octet-stream"),
        "B_file": ("B.npy", bufB, "application/octet-stream")
    }
    data = {
        "worker_url": "http://worker:8001",
        "aggregator_url": "http://aggregator:8002",
        "block_size": "1"
    }

    response = client.post("/split", files=files, data=data)
    assert response.status_code == 200

    result = response.json()
    assert "job_id" in result
    assert "blocks_dispatched" in result
    assert "shape_A" in result
    assert "shape_B" in result
    assert isinstance(result["blocks_dispatched"], int)

    # Verify that the splitter attempted to contact both services
    mock_post.assert_called()
