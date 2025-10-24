import requests
import numpy as np
import time
import sys
import io
from concurrent.futures import ProcessPoolExecutor, as_completed

SPLITTER_URL = "http://splitter:8000"
AGGREGATOR_URL = "http://aggregator:8002"

def create_matrix(n, identity=False):
    dtype = np.float32
    if identity:
        return np.eye(n, dtype=dtype)
    else:
        return np.arange(1, n * n + 1, dtype=dtype).reshape(n, n)

def run_pipeline(n=10, block_size=500, job_id=None):
    job_label = f"[Job-{job_id}]" if job_id is not None else ""
    print(f"{job_label} ⚙️ Creating matrices A({n}x{n}) and B({n}x{n})")

    A = create_matrix(n, identity=False)
    B = create_matrix(n, identity=True)

    bufA, bufB = io.BytesIO(), io.BytesIO()
    np.save(bufA, A)
    np.save(bufB, B)
    bufA.seek(0)
    bufB.seek(0)

    print(f"{job_label} 🟢 Sending job to splitter...")
    files = {
        "A_file": ("A.npy", bufA, "application/octet-stream"),
        "B_file": ("B.npy", bufB, "application/octet-stream")
    }
    data = {
        "block_size": str(block_size),
        "worker_url": "http://worker:8001",
        "aggregator_url": "http://aggregator:8002"
    }

    try:
        resp = requests.post(f"{SPLITTER_URL}/split", files=files, data=data, timeout=600)
        resp.raise_for_status()
    except Exception as e:
        print(f"{job_label} ❌ Splitter request failed: {e}")
        return None

    job_info = resp.json()
    job_uuid = job_info.get("job_id", "unknown")
    print(f"{job_label} ✅ Splitter accepted job {job_uuid}, dispatched {job_info.get('blocks_dispatched', '?')} blocks")

    # Poll aggregator
    result_url = f"{AGGREGATOR_URL}/aggregate/final_result/{job_uuid}"
    for _ in range(900):
        time.sleep(1)
        r = requests.get(result_url, timeout=30)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        data = r.json()
        if data.get("shape") and data.get("message") == "Aggregation complete":
            print(f"{job_label} 🏁 Final result ready! Shape: {data['shape']}")
            return data
    print(f"{job_label} ❌ Aggregator timeout.")
    return None

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    block_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    num_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    print(f"🚀 Launching {num_jobs} concurrent jobs of size {n}x{n} (block={block_size})")
    start = time.time()
    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(run_pipeline, n, block_size, i + 1) for i in range(num_jobs)]
        for future in as_completed(futures):
            result = future.result()
    end = time.time()
    print(f"\n✅ All jobs complete in {end - start:.2f} seconds")
