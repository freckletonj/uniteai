from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

class StreamState:
    def __init__(self):
        self.should_continue = True

stream_state = StreamState()
STREAM_TOK = '\n'

@app.get("/start_stream")
async def stream():
    stream_state.should_continue = True
    async def event_stream():
        i = 0
        while stream_state.should_continue:
            yield f"data: {i}" + STREAM_TOK
            i += 1
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/stop_stream")
async def end_stream():
    stream_state.should_continue = False
    return {"message": "Stream ended"}
