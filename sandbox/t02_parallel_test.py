import asyncio
import httpx
from contextlib import asynccontextmanager
import requests

# Here we are defining the main host
HOST = "http://localhost:8000"

# A context manager to handle SSE (Server Sent Events) connections.
@asynccontextmanager
async def sse_client(path: str):
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", f"{HOST}{path}") as response:
            yield response

async def listen_for_stream():
    async with sse_client("/start_stream") as response:
        async for line in response.aiter_lines():
            print(f"Stream Received: {line}")

async def stop_stream():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{HOST}/stop_stream")
        print(f"Stop Received: {response.json()}")

async def main():
    # Create a task to listen for streamed responses from the start_stream endpoint
    listen_task1 = asyncio.create_task(listen_for_stream())
    await asyncio.sleep(0.5)
    listen_task2 = asyncio.create_task(listen_for_stream())
    # Allow for the server to set up the stream
    await asyncio.sleep(2)
    # Create a task to send a request to stop the stream after some time
    stop_task = asyncio.create_task(stop_stream())
    # Wait for the tasks to complete
    await listen_task1
    await listen_task2
    await stop_task

    # Create a task to listen for streamed responses from the start_stream endpoint
    listen_task = asyncio.create_task(listen_for_stream())
    # Allow for the server to set up the stream
    await asyncio.sleep(2)
    # Create a task to send a request to stop the stream after some time
    stop_task = asyncio.create_task(stop_stream())
    # Wait for the tasks to complete
    await listen_task
    await stop_task


asyncio.run(main())
