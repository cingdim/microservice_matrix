import requests
import numpy as np
import time
import uuid
import sys
import io
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- service URLs ---
SPLITTER_URL = "http://splitter:8000"
AGGREGATOR_URL = "http://aggregator:8002"

def create_matrix(n, identity=False):
    dtype = np.float32
    return np.eye(n, dtype=dtype) if identity else np.arange(1, n * n + 1, dtype=dtype).reshape(n, n)

def run_pipeline(n=10, block_size=500, job_id=None, splitter_url=SPLITTER_URL):
    job_label = f"[Job-{job_id[:8]}]" if job_id is not None else ""
    print(f"{job_label}  Creating matrices A({n}x{n}) and B({n}x{n})")

    # --- Create test matrices ---
    A = create_matrix(n, identity=False)
    B = create_matrix(n, identity=True)  # multiplying by identity → correctness check is easy

    bufA, bufB = io.BytesIO(), io.BytesIO()
    np.save(bufA, A)
    np.save(bufB, B)
    bufA.seek(0)
    bufB.seek(0)

    print(f"{job_label}  Sending job to splitter {splitter_url}...")
    files = {
        "A_file": ("A.npy", bufA, "application/octet-stream"),
        "B_file": ("B.npy", bufB, "application/octet-stream")
    }
    
    data = {
        "block_size": str(block_size),
        "worker_url": "http://worker:8001",
        "aggregator_url": AGGREGATOR_URL,
        "job_id": job_id
    }

    try:
        resp = requests.post(f"{SPLITTER_URL}/split", files=files, data=data, timeout=1200)
        resp.raise_for_status()
    except Exception as e:
        print(f"{job_label}  Splitter request failed: {e}")
        return None

    job_info = resp.json()
    print(f"{job_label}  Splitter accepted job {job_id[:8]}, dispatched {job_info.get('blocks_dispatched', '?')} blocks")

    # --- Poll aggregator for completion ---
    result_url = f"{AGGREGATOR_URL}/aggregate/final_result/{job_id}"
    print(f"{job_label}  Waiting for aggregator result...")

    for attempt in range(900):
        time.sleep(3)
        try:
            r = requests.get(result_url, timeout=1000)
            if r.status_code == 404:
                # Print progress every minute
                if attempt % 20 == 0 and attempt > 0:
                    print(f"{job_label}  Still waiting... ({attempt * 3}s elapsed)")
                continue
            
            data = r.json()
            if data.get("message") == "Aggregation complete":
                print(f"{job_label}  Final result ready! Shape: {data['shape']}")
                
                # OPTIMIZATION 1: Only verify correctness for small matrices
                if n <= 1000:  # Skip verification for large matrices
                    if isinstance(data.get("final_result"), list):
                        print(f"{job_label}  Verifying correctness...")
                        final = np.array(data["final_result"], dtype=np.float32)
                        expected = A @ B
                        if np.allclose(final, expected, rtol=1e-3, atol=1e-4):
                            print(f"{job_label}  Correct final matrix.")
                        else:
                            print(f"{job_label}  Incorrect matrix result!")
                    elif isinstance(data.get("final_result"), str):
                        print(f"{job_label}  Large matrix — correctness check skipped.")
                        print(f"{job_label} Summary: {data['final_result']}")
                        if "result_summary" in data:
                            print(f"{job_label} Stats: {data['result_summary']}")
                else:
                    #  For very large matrices, skip verification entirely
                    print(f"{job_label}  Job completed (verification skipped for n > 1000)")
                    if isinstance(data.get("final_result"), str):
                        print(f"{job_label} Result: {data['final_result']}")
                        if "result_summary" in data:
                            print(f"{job_label} Stats: {data['result_summary']}")
                
                # Print timing stats if available
                if "worker_time_total" in data:
                    print(f"{job_label}  Worker time: {data['worker_time_total']:.2f}s")
                if "aggregation_time_sec" in data:
                    print(f"{job_label}  Aggregation time: {data['aggregation_time_sec']:.2f}s")

                return data
        except Exception as e:
            if attempt % 20 == 0 and attempt > 0:
                print(f"{job_label}  Poll error (attempt {attempt}): {e}")
            continue
    
    print(f"{job_label}  Aggregator timeout after 45 minutes.")
    return None

if __name__ == "__main__":
    time.sleep(15)  # wait for containers to boot
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    block_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    num_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    print("="*70)
    print(f"   Distributed Matrix Multiplication")
    print(f"   Matrix size: {n}×{n}")
    print(f"   Block size: {block_size}×{block_size}")
    print(f"   Concurrent jobs: {num_jobs}")
    print("="*70)

    start = time.time()
    
    #  OPTIMIZATION 3: For single job, don't use ProcessPoolExecutor overhead
    if num_jobs == 1:
        print("\n Running single job (optimized mode)")
        job_id = str(uuid.uuid4())
        result = run_pipeline(n, block_size, job_id, SPLITTER_URL)
        results = [result] if result else []
    else:
        print(f"\n Running {num_jobs} concurrent jobs")
        with ProcessPoolExecutor(max_workers=num_jobs) as executor:
            # Each job gets a unique ID - no sharing
            futures = [
                executor.submit(run_pipeline, n, block_size, str(uuid.uuid4()), SPLITTER_URL)
                for i in range(num_jobs)
            ]
            
            results = []
            for f in as_completed(futures):
                result = f.result()
                if result:
                    results.append(result)
    
    end = time.time()
    if len(results) > 0:
        final_result = np.array(results[0]["final_result"])
        job_id = results[0]["job_id"]
        np.savetxt(f"/app/results/final_result_{job_id}.csv", final_result, delimiter=",")
        print(f"Saved final_result_{job_id}.csv")

    
    print("\n" + "="*70)
    print(f" COMPLETED: {len(results)}/{num_jobs} jobs successful")
    print(f"  Total wall time: {end - start:.2f}s")
    
    if len(results) > 0:
        # Calculate average times
        avg_worker_time = np.mean([r.get("worker_time_total", 0) for r in results if "worker_time_total" in r])
        avg_agg_time = np.mean([r.get("aggregation_time_sec", 0) for r in results if "aggregation_time_sec" in r])
        
        if avg_worker_time > 0:
            print(f" Avg worker time: {avg_worker_time:.2f}s")
        if avg_agg_time > 0:
            print(f" Avg aggregation time: {avg_agg_time:.2f}s")
    
    if len(results) < num_jobs:
        print(f"  Warning: {num_jobs - len(results)} jobs failed")
    
    print("="*70)