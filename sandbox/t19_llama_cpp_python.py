'''

What features does llama-cpp-python have that we can make use of?

'''


import json
import argparse
from llama_cpp import Llama
from transformers import TextIteratorStreamer
import queue
import threading
import time

class QueueIterator:
    def __init__(self, q, stop_event):
        self.queue = q
        self.stop_event = stop_event

    def __iter__(self):
        return self

    def __next__(self):
        # Check if the stop event is set, and if so, stop the iteration
        if self.stop_event.is_set():
            raise StopIteration

        try:
            # Non-blocking get with a short timeout
            item = self.queue.get_nowait()
            return item
        except queue.Empty:
            # If the queue is empty, check again
            time.sleep(0.05)
            return self.__next__()

# Example usage
q = queue.Queue()
stop_event = threading.Event()

MODEL_PATH = '/home/mobius/_/models/zephyr-7b-beta.Q4_K_M.gguf'
llm = Llama(
    model_path=MODEL_PATH,
    verbose=False
)  # llama_cpp.llama.Llama

stream = llm(
    "Question: What are the names of the planets in the solar system? Answer: ",
    max_tokens=48,
    # stop=["Q:", "\n"],
    stream=True,
)

for output in stream:
    print(output['choices'][0]['text'], end='')
    # print(json.dumps(output, indent=2))
