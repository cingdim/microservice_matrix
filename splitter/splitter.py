from fastapi import FastAPI, HTTPException, UploadFile, Form
import numpy as np
import requests, uuid, math, io, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import os

app = FastAPI(title="Splitter Microservice")

@app.post("/split")
async def split_and_dispatch(
    A_file: UploadFile,
    B_file: UploadFile,
    worker_url: str = Form("http://worker:8001"),
    aggregator_url: str = Form("http://aggregator:8002"),
    block_size: int = Form(500),
    job_id: str = Form(None)
):
    try:
        start_time = time.perf_counter()

        # Unique temporary paths for each request
        tmp_dir = tempfile.mkdtemp(prefix="splitjob_")
        A_path = os.path.join(tmp_dir, f"A_{uuid.uuid4()}.npy")
        B_path = os.path.join(tmp_dir, f"B_{uuid.uuid4()}.npy")

        with open(A_path, "wb") as f:
            f.write(await A_file.read())
        with open(B_path, "wb") as f:
            f.write(await B_file.read())

        A = np.load(A_path, mmap_mode="r")
        B = np.load(B_path, mmap_mode="r")

        if A.shape[1] != B.shape[0]:
            raise HTTPException(
                status_code=400,
                detail=f"Incompatible matrix dimensions: A{A.shape} × B{B.shape}"
            )

        m, n = A.shape
        _, p = B.shape
        b = block_size

        row_blocks = math.ceil(m / b)
        col_blocks = math.ceil(p / b)
        depth_blocks = math.ceil(n / b)

        # Generate job_id if not provided
        job_id = job_id or str(uuid.uuid4())
        total_blocks = row_blocks * col_blocks * depth_blocks

        # Initialize job at aggregator
        init_payload = {
            "job_id": job_id,
            "blocks_expected": total_blocks,
            "block_rows": row_blocks,
            "block_cols": col_blocks
        }
        
        try:
            resp = requests.post(
                f"{aggregator_url}/init_job",
                json=init_payload,
                timeout=10
            )
            if resp.status_code == 200:
                print(f"✅ Initialized aggregator job {job_id}")
            else:
                print(f"⚠️ Aggregator returned {resp.status_code} (may already exist)")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Aggregator init failed: {e}")

        print(f"📦 Job {job_id}: dispatching {total_blocks} blocks "
              f"({row_blocks}×{col_blocks}×{depth_blocks})")

        # Define block sending task
        def send_block(i, j, k):
            """
            Compute partial result for C[i,j] from k-th depth slice:
            C[i,j] += A[i,k] × B[k,j]
            """
            try:
                A_block = A[i*b:(i+1)*b, k*b:(k+1)*b]
                B_block = B[k*b:(k+1)*b, j*b:(j+1)*b]

                bufA, bufB = io.BytesIO(), io.BytesIO()
                np.save(bufA, A_block, allow_pickle=False)
                np.save(bufB, B_block, allow_pickle=False)
                bufA.seek(0)
                bufB.seek(0)

                files = {
                    "A_file": (f"A_block_{i}_{k}.npy", bufA, "application/octet-stream"),
                    "B_file": (f"B_block_{k}_{j}.npy", bufB, "application/octet-stream")
                }
                
                data = {
                    "job_id": job_id,
                    "row_block": i,
                    "col_block": j,
                    "depth_block": k,  # ✅ ADDED: Include k-index
                    "aggregator_url": aggregator_url
                }

                resp = requests.post(
                    f"{worker_url}/multiply",
                    data=data,
                    files=files,
                    timeout=30
                )
                
                if resp.status_code == 200:
                    return True
                else:
                    print(f"❌ Worker returned {resp.status_code} for block ({i},{j},{k})")
                    return False

            except Exception as e:
                print(f"❌ Failed block ({i},{j},{k}): {e}")
                return False

        # Dispatch all blocks concurrently
        dispatched, failed = 0, 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(send_block, i, j, k)
                for i in range(row_blocks)
                for j in range(col_blocks)
                for k in range(depth_blocks)
            ]

            for idx, f in enumerate(as_completed(futures), start=1):
                result = f.result()
                if result:
                    dispatched += 1
                else:
                    failed += 1

                if idx % 100 == 0 or idx == total_blocks:
                    print(f"🚀 Progress: {idx}/{total_blocks} dispatched "
                          f"({dispatched} success, {failed} failed)")

        elapsed = time.perf_counter() - start_time
        print(f"✅ Splitter job {job_id} completed: {dispatched}/{total_blocks} "
              f"blocks in {elapsed:.2f}s")

        # Cleanup temp files
        try:
            os.remove(A_path)
            os.remove(B_path)
            os.rmdir(tmp_dir)
        except:
            pass

        return {
            "job_id": job_id,
            "blocks_dispatched": dispatched,
            "block_size": b,
            "shape_A": list(A.shape),
            "shape_B": list(B.shape),
            "time_sec": elapsed,
            "failed": failed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}