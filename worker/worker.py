from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import requests


app = FastAPI(title="Worker Microservice")

#data woker expecting when it's called to multiply matrix blocks
class MultiplyRequest(BaseModel):
    job_id: str
    block_row: int
    block_col: int
    A_block: list
    B_block: list
    aggregator_url: str

@app.post("/multiply")
def multiply(req: MultiplyRequest):
    try:
        # Convert to numpy arrays
        A = np.array(req.A_block)
        B = np.array(req.B_block)

        # Perform multiplication
        C = np.dot(A, B)
        C_block = C.tolist()

        # Send the result to the aggregator
        payload = {
            "job_id": req.job_id,
            "block_row": req.block_row,
            "block_col": req.block_col,
            "C_block": C_block
        }

        response = requests.post(f"{req.aggregator_url}/submit_block", json=payload)
        response.raise_for_status()

        return {"status": "done", "job_id": req.job_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}