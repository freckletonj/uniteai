from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from queue import Queue
import multiprocessing as mp
import threading
import time
import queue

app = FastAPI()

# This Event will be used to signal the process to stop
stop_event = mp.Event()

def infinite_generator(q):
    ''' Generate incrementing numbers until stopped. Place them on a queue. '''
    i = 0
    while not stop_event.is_set():
        q.put(f'{i} ')  # put data onto the queue
        print(f'put {i} ')
        i += 1
        time.sleep(0.1)  # to simulate work
    print('INF GEN COMPLETED')
    q.put(None)  # sentinel value to signal completion

def stream_results(q):
    ''' Read items off a queue and yield them. '''
    while True:
        res = q.get()  # fetch result from the queue
        print(f'get {res}')
        if res is None:  # if sentinel value received, exit loop
            return
        yield res + '\n'

@app.get("/start_stream")
def test_endpoint():
    stop_event.clear()

    # THREAD
    q = queue.Queue()
    threading.Thread(target=infinite_generator, args=(q,)).start()

    # # MULTIPROCESSING
    # q = mp.Manager().Queue()
    # mp.Process(target=infinite_generator, args=(q,)).start()

    # Stream the results off the queue
    try:
        return StreamingResponse(stream_results(q))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stop_stream")
def terminate_endpoint():
    stop_event.set()
    return {"message": "Process termination signal sent"}
