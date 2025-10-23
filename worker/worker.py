from fastapi import FastAPI, HTTPException, UploadFile, Form
import numpy as np
import io
import requests
import time
app = FastAPI(title="Worker Microservice")

@app.post("/multiply")
async def multiply(

    job_id: str = Form(...),
    row_block: int = Form(...),
    col_block: int = Form(...),
    aggregator_url: str = Form(...),
    A_file: UploadFile = None,
    B_file: UploadFile = None
):
    try:
        start_time = time.perf_counter()
        # Load A_block and B_block from uploaded npy files
        A_data = await A_file.read()
        B_data = await B_file.read()
        A = np.load(io.BytesIO(A_data))
        B = np.load(io.BytesIO(B_data))

        if A.shape[1] != B.shape[0]:
            raise HTTPException(status_code=400, detail="Incompatible matrix dimensions for multiplication")

        # Perform multiplication
        C_block = np.dot(A, B)
        compute_time = time.perf_counter() - start_time
        print(f"⏱ Worker multiplied block ({row_block}, {col_block}) in {compute_time:.4f} sec")

        # Serialize result back to npy
        buf = io.BytesIO()
        np.save(buf, C_block)
        buf.seek(0)

        files = {
            "file": (f"block_{row_block}_{col_block}.npy", buf, "application/octet-stream")
        }
        data = {
            "job_id": job_id,
            "row_block": row_block,
            "col_block": col_block,
            "worker_time_sec": compute_time
        }
       
        response = requests.post(f"{aggregator_url}/aggregate/submit_block", data=data, files=files, timeout=10)
        response.raise_for_status()

        return {"status": "done", "job_id": job_id, "shape": C_block.shape, "worker_time_sec": compute_time }  # ⏱ return timing

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to send result to aggregator: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}