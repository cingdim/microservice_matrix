from fastapi import FastAPI, UploadFile, Form, HTTPException
import numpy as np
import requests
import io
import time

app = FastAPI(title="Worker Microservice")

@app.post("/multiply")
async def multiply_blocks(
    A_file: UploadFile,
    B_file: UploadFile,
    job_id: str = Form(...),
    row_block: int = Form(...),
    col_block: int = Form(...),
    depth_block: int = Form(...),  # ✅ ADDED: Receive k-index
    aggregator_url: str = Form(...)
):
    """
    Compute partial result for C[row_block, col_block]:
    partial_result = A[row_block, depth_block] × B[depth_block, col_block]
    
    This represents ONE contribution to the final C[i,j] block.
    The aggregator will sum all depth_block contributions.
    """
    try:
        start_time = time.perf_counter()

        # Load matrix blocks
        A_content = await A_file.read()
        B_content = await B_file.read()
        
        A_block = np.load(io.BytesIO(A_content))
        B_block = np.load(io.BytesIO(B_content))

        # Validate dimensions
        if A_block.shape[1] != B_block.shape[0]:
            raise HTTPException(
                status_code=400,
                detail=f"Incompatible block dimensions: "
                       f"A{A_block.shape} × B{B_block.shape}"
            )

        # Perform block multiplication
        result_block = A_block @ B_block
        
        compute_time = time.perf_counter() - start_time

        print(f"✅ Computed block ({row_block},{col_block},{depth_block}) "
              f"for job {job_id} in {compute_time:.4f}s")

        # Send result to aggregator
        buf = io.BytesIO()
        np.save(buf, result_block, allow_pickle=False)
        buf.seek(0)

        files = {
            "file": (
                f"result_{row_block}_{col_block}_{depth_block}.npy",
                buf,
                "application/octet-stream"
            )
        }
        
        data = {
            "job_id": job_id,
            "row_block": row_block,
            "col_block": col_block,
            "depth_block": depth_block,  # ✅ ADDED: Pass k-index to aggregator
            "worker_time_sec": compute_time
        }

        resp = requests.post(
            f"{aggregator_url}/aggregate/submit_block",
            data=data,
            files=files,
            timeout=30
        )
        
        resp.raise_for_status()

        print(f"✅ Submitted block ({row_block},{col_block},{depth_block}) "
              f"to aggregator for job {job_id}")

        return {
            "message": "Block computed and submitted",
            "job_id": job_id,
            "block_position": (row_block, col_block, depth_block),
            "result_shape": result_block.shape,
            "compute_time_sec": compute_time
        }

    except Exception as e:
        print(f"❌ Worker error for block ({row_block},{col_block},{depth_block}): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}