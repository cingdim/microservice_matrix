from fastapi import FastAPI, Request

app = FastAPI()

# Store received blocks in memory
received_blocks = []

@app.post("/aggregate/submit_block")
async def submit_block(request: Request):
    data = await request.json()
    print(f"✅ Received block: {data}")
    received_blocks.append(data)
    return {"status": "received", "block": data}

@app.get("/aggregate/blocks")
def get_blocks():
    return {"received_blocks": received_blocks}

@app.get("/health")
def health():
    return {"status": "ok"}
