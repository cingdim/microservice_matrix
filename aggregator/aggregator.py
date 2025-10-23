from fastapi import FastAPI, Request, HTTPException, UploadFile, Form
from typing import Dict, Tuple
import numpy as np
import io
import time
app = FastAPI(title="Aggregator Microservice")

# Job storage
jobs: Dict[str, Dict] = {}

@app.post("/init_job")
async def init_job(request: Request):
    
    data = await request.json()
    job_id = data.get("job_id")
    blocks_expected = data.get("blocks_expected")
    block_rows = data.get("block_rows")
    block_cols = data.get("block_cols")

    if not job_id:
        raise HTTPException(status_code=400, detail="Missing job_id")

    jobs[job_id] = {
        "results": {},
        "blocks_expected": blocks_expected,
        "block_rows": block_rows,
        "block_cols": block_cols,
        "received": 0,
        "worker_times": []
    }

    print(f"✅ Initialized job {job_id} expecting {blocks_expected} blocks")
    return {"message": f"Job {job_id} initialized", "expected": blocks_expected}

@app.post("/aggregate/submit_block")
async def submit_block(
    job_id: str = Form(...),
    row_block: int = Form(...),
    col_block: int = Form(...),
    worker_time_sec: float = Form(0.0), 
    file: UploadFile = None
):
    if job_id not in jobs:
        raise HTTPException(status_code=400, detail="Unknown or missing job_id")

    # Load the npy block
    content = await file.read()
    buf = io.BytesIO(content)
    block_data = np.load(buf)

    # Store block
    job = jobs[job_id]
    job["results"][(row_block, col_block)] = block_data
    job["received"] += 1

    #track worker time
    job["worker_times"].append(worker_time_sec)

    print(f"✅ Received block ({row_block}, {col_block}) "
          f"for job {job_id} [{job['received']}/{job['blocks_expected']}] "
          f"(worker {worker_time_sec:.4f} sec)")
    return {"message": f"Stored block ({row_block},{col_block})", "job_id": job_id}

@app.get("/aggregate/final_result/{job_id}")
async def get_final_result(job_id: str):
    start_time = time.perf_counter() 
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    results = job["results"]

    if job["received"] < job["blocks_expected"]:
        return {"message": f"Not all blocks received yet ({job['received']}/{job['blocks_expected']})"}

    # Reconstruct with np.block (efficient stitching)
    row_blocks = []
    for r in range(job["block_rows"]):
        col_blocks = [results[(r, c)] for c in range(job["block_cols"])]
        row_blocks.append(np.hstack(col_blocks))
    final_result = np.vstack(row_blocks)

    shape = final_result.shape
    elapsed = time.perf_counter() - start_time
    print(f"✅ Done aggregator {elapsed:.4f} sec")
    print(f"🏁 Job {job_id} — Final result ready, shape {shape}")
    worker_times = job.get("worker_times", [])
    worker_summary = {}
    if worker_times:
        worker_summary = {
            "worker_time_total": float(np.sum(worker_times)),
            "worker_time_avg": float(np.mean(worker_times)),
            "worker_time_max": float(np.max(worker_times)),
            "worker_time_min": float(np.min(worker_times)),
        }
    # Option: don’t return entire matrix for huge jobs, just shape
    return {
        "message": "Aggregation complete",
        "job_id": job_id,
        "shape": shape,
        # ⚠️ CAREFUL: returning 100M numbers will blow up HTTP response
        # "final_result": final_result.tolist(),  # enable only for small jobs
        "aggregation_time_sec": elapsed,
        **worker_summary 
    }

@app.get("/aggregate/jobs")
def list_jobs():
    return {
        job_id: {
            "received": job["received"],
            "expected": job["blocks_expected"],
            "rows": job["block_rows"],
            "cols": job["block_cols"]
            
        } for job_id, job in jobs.items()
    }

@app.get("/health")
def health():
    return {"status": "ok"}
