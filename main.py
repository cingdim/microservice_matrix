import requests
import numpy as np
import time
import sys
import io

# Config
SPLITTER_URL = "http://localhost:8000"
AGGREGATOR_URL = "http://localhost:8002"


def create_matrix(n, identity=False):
    """Create a test matrix."""
    if identity:
        return np.eye(n, dtype=np.float64)
    else:
        return np.arange(1, n * n + 1, dtype=np.float64).reshape(n, n)


def run_pipeline(n=10, block_size=500):
    # Step 1. Create matrices
    A = create_matrix(n, identity=False)  # sequential numbers
    B = create_matrix(n, identity=True)   # identity matrix

    print(f"⚙️ Creating matrices A({n}x{n}) and B({n}x{n})")

    # Step 2. Save to memory buffers (no disk files needed)
    bufA, bufB = io.BytesIO(), io.BytesIO()
    np.save(bufA, A)
    np.save(bufB, B)
    bufA.seek(0)
    bufB.seek(0)

    # Step 3. Send to splitter as form-data
    print("🟢 Sending job to splitter...")
    files = {
        "A_file": ("A.npy", bufA, "application/octet-stream"),
        "B_file": ("B.npy", bufB, "application/octet-stream")
    }
    data = {
        "block_size": str(block_size),
        "worker_url": "http://worker:8001",
        "aggregator_url": "http://aggregator:8002"
    }

    resp = requests.post(f"{SPLITTER_URL}/split", files=files, data=data, timeout=120)
    resp.raise_for_status()
    job_info = resp.json()
    job_id = job_info["job_id"]

    print(f"✅ Splitter accepted job {job_id}, dispatched {job_info['blocks_dispatched']} blocks")
    print(f"Splitter time {job_info.get('time_sec', 0.0)} sec")

    # Step 4. Poll the aggregator
    result_url = f"{AGGREGATOR_URL}/aggregate/final_result/{job_id}"
    print("⏳ Waiting for aggregator to assemble result...")
    for _ in range(600):  # up to ~10 minutes
        time.sleep(1)
        r = requests.get(result_url, timeout=30)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        data = r.json()
        if data.get("shape") and data.get("message") == "Aggregation complete":
            print("🏁 Final result ready!")
            return data
    raise TimeoutError("Aggregator did not return a final result in time")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    block_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    result = run_pipeline(n=n, block_size=block_size)
    print(f"\n📌 Job ID: {result['job_id']}")
    print(f"📐 Result shape: {result['shape']}")

    print("\n⏱ Performance Summary")
    print(f"🕒 Aggregation time: {result.get('aggregation_time_sec', 0):.4f} sec")

    if "worker_time_total" in result:
        print(f"   Worker total time:   {result['worker_time_total']:.4f} sec")
        print(f"   Worker avg time:     {result['worker_time_avg']:.4f} sec")
        print(f"   Worker min time:     {result['worker_time_min']:.4f} sec")
        print(f"   Worker max time:     {result['worker_time_max']:.4f} sec")


