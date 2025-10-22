import requests
import numpy as np
import time
import sys

# Config: adjust if your Docker ports differ
SPLITTER_URL = "http://localhost:8000"
AGGREGATOR_URL = "http://localhost:8002"


def create_matrix(n, identity=False):
    """Create a test matrix."""
    if identity:
        return np.eye(n, dtype=np.float64).tolist()
    else:
        return np.arange(1, n * n + 1, dtype=np.float64).reshape(n, n).tolist()


def run_pipeline(n=10, identity=False):
    # Step 1. Create matrices
    A = create_matrix(n, identity=False)  # sequential numbers
    B = create_matrix(n, identity=True)   # identity matrix

    print(f"⚙️ Creating matrices A({n}x{n}) and B({n}x{n})")

    # Step 2. Send them to the splitter as JSON
    payload = {"A": A, "B": B}
    print("🟢 Sending job to splitter...")
    resp = requests.post(f"{SPLITTER_URL}/split", json=payload, timeout=120)
    resp.raise_for_status()
    job_info = resp.json()
    job_id = job_info["job_id"]
    print(f"✅ Splitter accepted job {job_id}, dispatched {job_info['blocks_dispatched']} blocks")
    print(f"Splitter time {resp.json().get('time_sec', 0.0)} sec")
    # Step 3. Poll the aggregator until complete
    result_url = f"{AGGREGATOR_URL}/aggregate/final_result/{job_id}"
    print("⏳ Waiting for aggregator to assemble result...")
    for _ in range(300):  # retry ~5 minutes
        time.sleep(1)
        r = requests.get(result_url, timeout=30)
        if r.status_code == 404:
            print("… Aggregator has not initialized job yet")
            continue
        r.raise_for_status()
        data = r.json()
        if data.get("shape") and data.get("message") == "Aggregation complete":
            print("🏁 Final result ready!")
            return data
        
        msg = data.get("message", "")
        if msg:
            print(f"… {msg}")

    raise TimeoutError("Aggregator did not return a final result in time")


if __name__ == "__main__":
    # Allow overrides from CLI
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    block_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    result = run_pipeline(n=n)

    print(f"\n📌 Job ID: {result['job_id']}")
    print(f"📐 Result shape: {result['shape']}")
    # print("result", result)
    # Print result if it's reasonably small
    print("\n⏱ Performance Summary")
    
    print(f"🕒 Aggregation time: {result.get('aggregation_time_sec', 0):.4f} sec")
     # Timing summary
    # Worker timings (from aggregator stats)
    if "worker_time_total" in result:
        print(f"   Worker total time:   {result['worker_time_total']:.4f} sec")
        print(f"   Worker avg time:     {result['worker_time_avg']:.4f} sec")
        print(f"   Worker min time:     {result['worker_time_min']:.4f} sec")
        print(f"   Worker max time:     {result['worker_time_max']:.4f} sec")
    # if result['shape'][0] <= 20:
    #     print("\n📊 Final Matrix:")
    #     for row in result.get("final_result", []):
    #         print(row)
    # else:
    #     print("⚠️ Matrix too large to print — only shape returned.")
