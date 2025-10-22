from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import requests


app = FastAPI(title="Worker Microservice")

#data woker expecting when it's called to multiply matrix blocks
class MultiplyRequest(BaseModel):
    job_id: str
    row_block: int
    col_block: int
    A_block: list
    B_block: list
    aggregator_url: str

@app.post("/multiply")
def multiply(req: MultiplyRequest):

   

    try:
        # Convert to numpy arrays
        A = np.array(req.A_block)
        B = np.array(req.B_block)

        if A.shape[1] != B.shape[0]:
            raise HTTPException(status_code=400, detail="Incompatible matrix dimensions for multiplication")
        
        # Perform multiplication
        C_block = np.dot(A, B).tolist()


        # Send the result to the aggregator
        payload = {
            "job_id": req.job_id,
            "row_block": req.row_block,
            "col_block": req.col_block,
            "data": C_block
        }

        response = requests.post(f"{req.aggregator_url}/submit_block", json=payload, timeout=5)
        response.raise_for_status()

        return {"status": "done", "job_id": req.job_id, "shape": [len(C_block), len(C_block[0])]}
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to send result to aggregator: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}