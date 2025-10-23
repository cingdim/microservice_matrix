from fastapi import FastAPI, HTTPException, UploadFile, Form
import numpy as np
import requests, uuid, math, io, time
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Splitter Microservice")

@app.post("/split")
async def split_and_dispatch(
    A_file: UploadFile,
    B_file: UploadFile,
    worker_url: str = Form("http://worker:8001"),
    aggregator_url: str = Form("http://aggregator:8002"),
    block_size: int = Form(500)
):
    try:
        start_time = time.perf_counter()

        # Save uploaded .npy files temporarily
        A_path, B_path = "A.npy", "B.npy"
        with open(A_path, "wb") as f:
            f.write(await A_file.read())
        with open(B_path, "wb") as f:
            f.write(await B_file.read())

        # Memory-map (zero-copy access)
        A = np.load(A_path, mmap_mode="r")
        B = np.load(B_path, mmap_mode="r")

        if A.shape[1] != B.shape[0]:
            raise HTTPException(status_code=400, detail="Incompatible matrix dimensions")

        m, n = A.shape
        _, p = B.shape
        b = block_size

        row_blocks = math.ceil(m / b)
        col_blocks = math.ceil(p / b)
        depth_blocks = math.ceil(n / b)

        job_id = str(uuid.uuid4())
        total_blocks = row_blocks * col_blocks * depth_blocks

        # Initialize job
        requests.post(f"{aggregator_url}/init_job", json={
            "job_id": job_id,
            "blocks_expected": total_blocks,
            "block_rows": row_blocks,
            "block_cols": col_blocks
        }, timeout=10)

        print(f"📦 Job {job_id}: dispatching {total_blocks} blocks "
              f"({row_blocks}x{col_blocks}x{depth_blocks})")

        def send_block(i, j, k):
            """Send one matrix block to a worker."""
            try:
                A_block = A[i*b:(i+1)*b, k*b:(k+1)*b]
                B_block = B[k*b:(k+1)*b, j*b:(j+1)*b]

                bufA, bufB = io.BytesIO(), io.BytesIO()
                np.save(bufA, A_block, allow_pickle=False)
                np.save(bufB, B_block, allow_pickle=False)
                bufA.seek(0); bufB.seek(0)

                files = {
                    "A_file": (f"A_block_{i}_{k}.npy", bufA, "application/octet-stream"),
                    "B_file": (f"B_block_{k}_{j}.npy", bufB, "application/octet-stream")
                }
                data = {
                    "job_id": job_id,
                    "row_block": i,
                    "col_block": j,
                    "aggregator_url": aggregator_url
                }
                requests.post(f"{worker_url}/multiply", data=data, files=files, timeout=30)
                return True
            except Exception as e:
                print(f"❌ Failed block ({i},{j},{k}): {e}")
                return False

        dispatched = 0
        failed = 0

        # Parallel dispatch with futures
        with ThreadPoolExecutor(max_workers=16) as executor:  # you can tweak concurrency here
            futures = [executor.submit(send_block, i, j, k)
                       for i in range(row_blocks)
                       for j in range(col_blocks)
                       for k in range(depth_blocks)]

            for idx, f in enumerate(as_completed(futures), start=1):
                result = f.result()
                if result:
                    dispatched += 1
                else:
                    failed += 1

                # progress log every 100 blocks
                if idx % 100 == 0 or idx == total_blocks:
                    print(f"🚀 Progress: {idx}/{total_blocks} dispatched "
                          f"({dispatched} success, {failed} failed)")

        elapsed = time.perf_counter() - start_time
        print(f"✅ Dispatched {dispatched}/{total_blocks} blocks in {elapsed:.2f}s")

        return {
            "job_id": job_id,
            "blocks_dispatched": dispatched,
            "block_size": b,
            "shape_A": A.shape,
            "shape_B": B.shape,
            "time_sec": elapsed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
