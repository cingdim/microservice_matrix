from fastapi import FastAPI, Request, HTTPException
from typing import Dict, Tuple, List
import numpy as np
import math

app = FastAPI(title="Aggregator Microservice")

# Job storage: keeps all jobs isolated
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
        "received": 0
    }

    print(f"✅ Initialized job {job_id} expecting {blocks_expected} blocks")
    return {"message": f"Job {job_id} initialized", "expected": blocks_expected}


@app.post("/aggregate/submit_block")
async def submit_block(request: Request):
    data = await request.json()
    job_id = data.get("job_id")
    row_block = data.get("row_block")
    col_block = data.get("col_block")
    block_data = data.get("data")

    if not job_id or job_id not in jobs:
        raise HTTPException(status_code=400, detail="Unknown or missing job_id")

    job = jobs[job_id]

    job["results"][(row_block, col_block)] = block_data
    job["received"] += 1

    print(f"✅ Received block ({row_block}, {col_block}) for job {job_id}")

    # Optional: if all blocks received, compute immediately
    if job["received"] == job["blocks_expected"]:
        print(f"🏁 Job {job_id} — all blocks received!")

    return {"message": f"Stored block ({row_block}, {col_block})", "job_id": job_id}


@app.get("/aggregate/final_result/{job_id}")
async def get_final_result(job_id: str):
    """
    Combines stored blocks for a specific job into a full matrix.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    results = job["results"]

    if len(results) == 0:
        return {"message": "No results yet for this job"}

    row_blocks = job["block_rows"]
    col_blocks = job["block_cols"]

    final_result = []
    for r in range(row_blocks):
        row_parts = [results.get((r, c)) for c in range(col_blocks)]
        if any(p is None for p in row_parts):
            return {"message": f"Missing blocks in row {r}"}

        row_combined = []
        for i in range(len(row_parts[0])):
            row_line = []
            for block in row_parts:
                row_line.extend(block[i])
            row_combined.append(row_line)

        final_result.extend(row_combined)

    shape = [len(final_result), len(final_result[0]) if final_result else 0]

    print(f"🏁 Job {job_id} — Final result ready, shape {shape}")
    return {
        "message": "Aggregation complete",
        "job_id": job_id,
        "shape": shape,
        "final_result": final_result
    }


@app.get("/aggregate/jobs")
def list_jobs():
    """
    Debug endpoint: lists all active jobs and how many blocks have been received.
    """
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
