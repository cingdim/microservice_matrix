from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import requests
import uuid
import math

app = FastAPI(title="Splitter Microservice")

# incoming request model
class SplitRequest(BaseModel):
    A: list
    B: list
    block_size: int = 0
    worker_url: str = "http://worker:8000"
    aggregator_url: str = "http://aggregator:8000"

@app.post("/split")
def split_and_dispatch(req: SplitRequest):
    try:
        # Convert lists to numpy arrays
        A = np.array(req.A)
        B = np.array(req.B)

        # Validate shapes
        if A.shape[1] != B.shape[0]:
            raise HTTPException(status_code=400, detail="Incompatible matrix dimensions")

        m, n = A.shape
        _, p = B.shape

        # Auto-select block size if not provided
        if req.block_size <= 0:
            req.block_size = max(50, min(n // 10, 1000))
            print(f"🧮 Auto-selected block size: {req.block_size}")

        b = req.block_size

        # Determine block counts
        row_blocks = math.ceil(m / b)
        col_blocks = math.ceil(p / b)
        depth_blocks = math.ceil(n / b)

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Notify aggregator to initialize job
        init_payload = {
            "job_id": job_id,
            "blocks_expected": row_blocks * col_blocks * depth_blocks,
            "block_rows": row_blocks,
            "block_cols": col_blocks
        }
        try:
            requests.post(f"{req.aggregator_url}/init_job", json=init_payload, timeout=10)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to contact aggregator: {e}")

        dispatched = 0

        # Iterate through blocks
        for i in range(row_blocks):
            for j in range(col_blocks):
                for k in range(depth_blocks):
                    A_block = A[i*b:(i+1)*b, k*b:(k+1)*b]
                    B_block = B[k*b:(k+1)*b, j*b:(j+1)*b]

                    payload = {
                        "job_id": job_id,
                        "row_block": i,
                        "col_block": j,
                        "A_block": A_block.tolist(),
                        "B_block": B_block.tolist(),
                        "aggregator_url": req.aggregator_url
                    }

                    try:
                        requests.post(f"{req.worker_url}/multiply", json=payload, timeout=15)
                        dispatched += 1
                    except Exception as e:
                        print(f"❌ Failed to send block ({i},{j},{k}): {e}")

        print(f"✅ Dispatched {dispatched} total blocks")
        return {
            "job_id": job_id,
            "blocks_dispatched": dispatched,
            "block_size": b,
            "shape_A": A.shape,
            "shape_B": B.shape
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
