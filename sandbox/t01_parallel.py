'''

I want to test if you can ascertain best practices when architecting a software solution in python.

Imagine you have a FastAPI server with 2 endpoints. One of the endpoints spins off a thread which has an infinite iterator/generator inside.
The other endpoint has the job of stopping the process running inside the first endpoint.
As an extra complication, the infinite iterator inside the first endpoint must acquire a global lock before it is allowed to do work.

So:

* one endpoint spins off a separate thread with an iterator inside. You must find a way to return this to the calling thread so that it can use FastAPIs StreamingResponse to stream the data to the client.

* That infinite generator can only be activated when it acquires a lock.

* That infinite generator can be halted if a global Event is set

* If it does get halted, the global Event must be cleared so that future calls will work once more.

Understood?

Please implement this in FastAPI. Feel free to make up an infinite generator that, say, yields an infinitely incrementing stream of numbers. This is just for demonstration.

It must have 2 endpoints. One to use a StreamingResponse which streams things out of a separate `threading.Thread`, but only once it acquires the lock.

The 2nd endpoint sets and Event which can terminate the infinite stream.

Please implement this difficult concurrency/parallelism problem.


AI Model + FastAPI
https://stackoverflow.com/questions/71613305/how-to-process-requests-from-multiiple-users-using-ml-model-and-fastapi?noredirect=1&lq=1

'''


# uvicorn sandbox.t01_parallel:app --port 8000

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from queue import Queue
import threading
import time
import asyncio

app = FastAPI()

# Define the global lock and event
lock = asyncio.Lock()
event = asyncio.Event()

# Define the queue for thread-safe data sharing
queue = Queue()

async def infinite_generator():
    i = 0
    # with lock:
    while not event.is_set():
        await print(f'gen. event={event.is_set()}')
        yield (f'{i} ')
        i += 1
        await asyncio.sleep(0.1)  # to simulate work
    print('INF GEN COMPLETED')

def empty_queue(q):
    while not q.empty():
        q.get()

@app.get("/start_stream")
async def start_stream():
    thread = threading.Thread(target=infinite_generator, daemon=True)
    thread.start()

    def stream():
        try:
            # while not event.is_set():
            while not queue.empty():
                print(f'stream. event={event.is_set()}')
                yield queue.get()
        except asyncio.CancelledError:
            print(f'CLIENT CANCELLED: {e}')
            event.set()
            # empty_queue(queue)
        except KeyboardInterrupt:
            print(f'KEYBOARD CANCELLED: {e}')
            event.set()
            # empty_queue(queue)
        print('STREAM DATA COMPLETED')
        # event.clear() # reset event

    return StreamingResponse(stream())

@app.get("/stop_stream")
async def stop_stream():
    print('Setting event')
    await event.set()
    return {"message": "Stream stopped"}


##########
# Test client closing connection

async def stream_data():
    ''' for testing when a client disconnects '''
    try:
        i=0
        while True:
            i+=1
            yield f"data: {i}\n\n"
            await asyncio.sleep(0.2)
    except asyncio.CancelledError:
        print("Client disconnected prematurely")

@app.get("/infinite")
def read_root():
    return StreamingResponse(stream_data())
