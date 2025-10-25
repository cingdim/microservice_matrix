import io
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch
from worker.worker import app  # ✅ import from your worker file

client = TestClient(app)

def test_worker_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

@patch("worker.worker.requests.post")  # mock network POST to aggregator
def test_worker_multiply_endpoint(mock_post):
    # Dummy response for aggregator
    class DummyResponse:
        def __init__(self, status_code=200):
            self.status_code = status_code
        def raise_for_status(self): pass
        def json(self): return {"message": "ok"}
    mock_post.return_value = DummyResponse(200)

    # Build A and B test matrices
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    bufA, bufB = io.BytesIO(), io.BytesIO()
    np.save(bufA, A); bufA.seek(0)
    np.save(bufB, B); bufB.seek(0)

    files = {
        "A_file": ("A.npy", bufA, "application/octet-stream"),
        "B_file": ("B.npy", bufB, "application/octet-stream")
    }
    data = {
        "job_id": "test-job",
        "row_block": "0",
        "col_block": "0",
        "aggregator_url": "http://aggregator:8002"
    }

    resp = client.post("/multiply", data=data, files=files)
    assert resp.status_code == 200
    result = resp.json()
    assert "status" in result
    assert result["status"] == "done"
    assert "shape" in result
    assert result["shape"] == [2, 2]
    mock_post.assert_called_once()
