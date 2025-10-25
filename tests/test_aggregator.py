import io
import numpy as np
from fastapi.testclient import TestClient
from aggregator.aggregator import app  # ✅ import from your aggregator file

client = TestClient(app)

def test_aggregator_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_aggregator_init_and_submit():
    # 1️⃣ Initialize a job
    init_payload = {
        "job_id": "test-job",
        "blocks_expected": 1,
        "block_rows": 1,
        "block_cols": 1
    }
    r1 = client.post("/init_job", json=init_payload)
    assert r1.status_code == 200
    assert "Job test-job initialized" in r1.json()["message"]

    # 2️⃣ Submit a fake block
    fake_block = np.array([[19, 22], [43, 50]], dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, fake_block)
    buf.seek(0)

    files = {"file": ("block.npy", buf, "application/octet-stream")}
    form = {
        "job_id": "test-job",
        "row_block": "0",
        "col_block": "0",
        "worker_time_sec": "0.05"
    }

    r2 = client.post("/aggregate/submit_block", data=form, files=files)
    assert r2.status_code == 200
    assert "Stored block" in r2.json()["message"]

    # 3️⃣ Request final result
    r3 = client.get("/aggregate/final_result/test-job")
    assert r3.status_code == 200
    data = r3.json()
    assert data["message"] == "Aggregation complete"
    assert data["shape"] == [2, 2]
