import threading
import queue
import sys
import time
import contextlib

completion_lock = threading.Lock()
cancel_token = threading.Event()
completion_queue = queue.Queue()

@contextlib.contextmanager
def non_blocking_lock(lock):
    if not lock.acquire(blocking=False):
        raise BlockingIOError
    try:
        yield lock
    finally:
        lock.release()

def start_streaming():
    if completion_lock.acquire(blocking=False):  # Try to acquire the lock
        cancel_token.clear()  # Reset the cancellation token

        # Start the streaming operation in a new thread
        streaming_thread = threading.Thread(target=stream_completion)
        streaming_thread.start()
    else:
        print('Another completion operation is already in progress.')

def stop_streaming():
    cancel_token.set()  # Signal the streaming operation to cancel

    if completion_lock.acquire(blocking=False):
        completion_lock.release()  # Release the lock

def stream_completion():
    while not cancel_token.is_set():
        # Streaming from LLM Here
        completion_queue.put('.')
        time.sleep(0.1)

    # If we get here, the operation was cancelled or completed normally
    # In either case, release the lock
    if completion_lock.acquire(blocking=False):
        completion_lock.release()

def print_from_queue():
    while True:
        while not completion_queue.empty():
            x = completion_queue.get()
            print(x, end='.', flush=True)
        time.sleep(0.1)  # Prevents this loop from being too CPU-intensive

def main():
    threading.Thread(target=print_from_queue, daemon=True).start()

    while True:
        command = input('Enter command: ')
        if command == 'start':
            start_streaming()
        elif command == 'stop':
            stop_streaming()
        elif command == 'quit':
            break

if __name__ == '__main__':
    main()




# from pygls.server import LanguageServer
# from threading import Lock
# from pygls.types import (TextEdit, Position, CompletionItem)
# import threading
# import queue

# server = LanguageServer()

# completion_lock = Lock()
# cancel_token = threading.Event()
# completion_queue = queue.Queue()

# @server.feature('textDocument/completion', trigger_characters=['.'])
# def start_streaming(ls, params):
#     global completion_lock, cancel_token

#     if completion_lock.acquire(blocking=False):  # Try to acquire the lock
#         cancel_token.clear()  # Reset the cancellation token
#         initial = params['textDocument']['uri']

#         # Start the streaming operation in a new thread
#         streaming_thread = threading.Thread(target=stream_completion, args=(initial,))
#         streaming_thread.start()
#     else:
#         ls.show_message('Another completion operation is already in progress.', msg_type=1)

# @server.command('cancelStreaming')
# def stop_streaming(ls, params):
#     global cancel_token
#     cancel_token.set()  # Signal the streaming operation to cancel
#     completion_lock.release()  # Release the lock

# def stream_completion(initial: str):
#     # Streaming operation. Periodically checks for cancellation.
#     while not cancel_token.is_set():
#         # Streaming logic goes here
#         # When an edit is ready, add it to the queue
#         completion_queue.put(TextEdit(range=..., newText=...))

#     # If we get here, the operation was cancelled or completed normally
#     # In either case, release the lock
#     completion_lock.release()
#     return 'done'
