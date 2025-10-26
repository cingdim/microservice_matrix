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

    if job_id in jobs:
        # Job already initialized - return existing info
        print(f"⚠️ Job {job_id} already initialized, returning existing configuration")
        return {
            "message": f"Job {job_id} already exists",
            "expected": jobs[job_id]["blocks_expected"]
        }

    jobs[job_id] = {
        "results": {},  # key: (i,j,k) -> value: A[i,k] × B[k,j]
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
    depth_block: int = Form(...),  # ✅ ADDED: k-index to prevent duplicates
    worker_time_sec: float = Form(0.0),
    file: UploadFile = None
):
    if job_id not in jobs:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown job_id: {job_id}. Job may not be initialized."
        )

    content = await file.read()
    buf = io.BytesIO(content)
    block_data = np.load(buf)

    job = jobs[job_id]
    key = (row_block, col_block, depth_block)  # ✅ 3D key: (i, j, k)

    # ✅ FIXED: Only store once per unique (i,j,k) combination
    if key in job["results"]:
        print(
            f"⚠️ Duplicate block ({row_block},{col_block},{depth_block}) "
            f"for job {job_id} - ignoring"
        )
        return {
            "message": f"Duplicate block ({row_block},{col_block},{depth_block}) ignored",
            "job_id": job_id
        }
    
    job["results"][key] = block_data
    job["received"] += 1
    job["worker_times"].append(worker_time_sec)

    print(
        f"✅ Received block ({row_block},{col_block},{depth_block}) for job {job_id} "
        f"[{job['received']}/{job['blocks_expected']}] (worker {worker_time_sec:.4f}s)"
    )
    
    return {
        "message": f"Stored block ({row_block},{col_block},{depth_block})",
        "job_id": job_id
    }


# In aggregator.py, modify get_final_result function:

@app.get("/aggregate/final_result/{job_id}")
async def get_final_result(job_id: str):
    start_time = time.perf_counter()
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    results = job["results"]

    if job["received"] < job["blocks_expected"]:
        return {
            "message": f"Not all blocks received yet ({job['received']}/{job['blocks_expected']})"
        }

    # Accumulate k-contributions properly
    final_blocks = {}
    for (i, j, k), partial_result in results.items():
        ij_key = (i, j)
        if ij_key not in final_blocks:
            final_blocks[ij_key] = np.zeros_like(partial_result)
        final_blocks[ij_key] += partial_result

    # Reconstruct full matrix
    row_blocks = []
    for r in range(job["block_rows"]):
        col_blocks = [final_blocks[(r, c)] for c in range(job["block_cols"])]
        row_blocks.append(np.hstack(col_blocks))
    
    final_result = np.vstack(row_blocks)
    shape = final_result.shape
    
    elapsed = time.perf_counter() - start_time
    print(f"✅ Aggregation complete in {elapsed:.4f}s")
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
    
    # ✅ NEW: Only return full matrix for small results
    total_elements = shape[0] * shape[1]
    if total_elements <= 10000:  # Only return if <= 100x100
        return {
            "message": "Aggregation complete",
            "job_id": job_id,
            "shape": shape,
            "final_result": final_result.tolist(),
            "aggregation_time_sec": elapsed,
            **worker_summary
        }
    else:
        # For large matrices, return shape and stats only
        return {
            "message": "Aggregation complete",
            "job_id": job_id,
            "shape": shape,
            "final_result": "Matrix too large to return via HTTP. Shape and stats provided.",
            "result_summary": {
                "min": float(np.min(final_result)),
                "max": float(np.max(final_result)),
                "mean": float(np.mean(final_result)),
                "std": float(np.std(final_result)),
            },
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