from typing import List, Dict, Tuple
from fastapi import FastAPI, Request

app = FastAPI()

# Temporary in-memory storage for results
results_storage: Dict[Tuple[int, int], List[List[int]]] = {}

@app.post("/aggregate/submit_block")
async def submit_block(request: Request):
    """
    Receives a partial matrix result block and stores it.
    Example payload:
    {
        "row_block": 0,
        "col_block": 1,
        "data": [[1, 2], [3, 4]]
    }
    """
    data = await request.json()
    row_block = data.get("row_block")
    col_block = data.get("col_block")
    block_data = data.get("data")

    if row_block is None or col_block is None or block_data is None:
        return {"error": "Missing row_block, col_block, or data"}

    results_storage[(row_block, col_block)] = block_data
    print(f"✅ Received block ({row_block}, {col_block}): {block_data}")


    return {"message": f"Block ({row_block}, {col_block}) received successfully"}

@app.get("/aggregate/final_result")
async def get_final_result():
    """
    Combines all received blocks into a final result matrix.
    Blocks are assumed to be keyed by (row_block, col_block).
    """
    if not results_storage:
        return {"message": "No results yet"}

   # Determine the block grid size
    row_blocks = sorted(set(k[0] for k in results_storage.keys()))
    col_blocks = sorted(set(k[1] for k in results_storage.keys()))

    final_result = []

    #combine row by row
    for r in row_blocks:
        #get all blokcs in this row
        row_parts = [results_storage.get((r, c)) for c in col_blocks]

        
        if None in row_parts:
            return {"error": f"Missing blocks in row {r}"}
        
        #horizontally stack the blocks in this row
        combined_rows=[]
        for i in range(len(row_parts[0])):  # for each row in the block
            combined_row=[]
            for block in row_parts:
                combined_row.extend(block[i])
            combined_rows.append(combined_row)
        final_result.extend(combined_rows)


    if not final_result or not isinstance(final_result[0], list):
        return {"error": "Invalid result structure"}

    print("Final aggregated result:")
    for row in final_result:
        print(row)

    return {
        "message": "Aggregation complete",
        "shape": [len(final_result), len(final_result[0])],
        "final_result": final_result
    }

@app.get("/health")
def health():
    return {"status": "ok"}
