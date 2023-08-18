'''

Multiprocessing is hard to get right with CUDA, so this attempts a minimal
example.

'''
import torch
import torch.multiprocessing as mp
from thespian.actors import ActorSystem, Actor
import time
import speech_recognition as sr
import numpy as np
from threading import Thread

CONCURRENCY_MODEL = 'multiprocQueueBase'
# CONCURRENCY_MODEL = 'simpleSystemBase'

DEVICE = 'cuda'

# Define the actor that will receive the tensor and process it
class WorkerActor(Actor):
    def __init__(self):
        self.r = sr.Recognizer()
        self.mic = sr.Microphone()

    def recognize(self, audio):
        return self.r.recognize_whisper(
            audio,
            model='tiny',
            load_options=dict(
                device=DEVICE,
            ), language='english')

    def _warmup(self):
        ''' Warm up, intended for a separate thread. '''

        # Get some mic params
        with self.mic as source:
            self.sample_rate = source.SAMPLE_RATE
            self.sample_width = source.SAMPLE_WIDTH

        # Get model into memory
        empty_audio = sr.AudioData(np.zeros(10), sample_rate=1, sample_width=1)
        self.recognize(empty_audio)
        print(f'Warmed up. sample_rate={self.sample_rate}, sample_width={self.sample_width}')

    def warmup(self):
        '''Load whisper model into memory.'''
        print('Warming up transcription model in separate thread')

    def receiveMessage(self, message, sender):
        message = message.cuda()
        warmup_thread = Thread(target=self._warmup)
        warmup_thread.daemon = True
        warmup_thread.start()
        print(f"Processed tensor value: {message.item()}, {message.device}")


if __name__ == "__main__":
    use_cuda = False
    tensor = torch.tensor([42.0])
    if use_cuda:
        tensor = tensor.cuda()

    actor_system = ActorSystem(CONCURRENCY_MODEL, logDefs={'version': 1, 'logLevel': 'DEBUG'})

    worker = actor_system.createActor(WorkerActor)
    actor_system.tell(worker, tensor)

    # worker2 = actor_system.createActor(WorkerActor)
    # actor_system.tell(worker2, tensor)

    time.sleep(10)
    # Shutdown the actor system after a brief delay (this delay is for demo
    # purposes to let actor process the tensor before shutting down)
    actor_system.shutdown()


'''

Great! Let's address the findings:

### Step 1:

If setting `use_cuda=False` worked, this confirms our suspicion that the problem arises when trying to send CUDA tensors between processes.

### Step 2:

No error is a bit perplexing, but it's consistent with our findings that CUDA tensors may not be sent correctly.

### Step 3:

This is indeed a challenging scenario. Large ML models can't be sent over the queue easily, and even if you could, the serialization and deserialization overhead would likely be prohibitive.

One potential approach to tackle this is the following:

1. **Model Weights Transfer**: Instead of sending the entire model, send only the model's state (i.e., its weights). This would be lighter than the entire model, but might still be sizable.

2. **Initialize Models in Actor**: If feasible, consider instantiating the model architecture in the actor itself, and then loading the weights (sent from the main process) into that architecture. This way, you're sending the minimum amount of data required.

### Step 4:

Thespian logs might be going somewhere other than stdio. The logging settings of Thespian can be adjusted, and it might be worthwhile to check where it's saving its logs or if it's generating any.

### Another Alternative:

Given your need to work with large models and CUDA, you may want to consider an alternative approach:

**Using `torch.multiprocessing` directly**:

Instead of using the Thespian library, you could leverage PyTorch's own multiprocessing module, which is built with CUDA in mind.

1. You'd spawn processes using `torch.multiprocessing.spawn` or `mp.Process`.
2. Use `mp.Queue` or `mp.Pipe` for inter-process communication.
3. Ensure that any CUDA operations on shared data happen within the child processes to avoid IPC complications.

Here's a very simplified example:

```python
import torch.multiprocessing as mp
import torch

def worker(rank, model_weights_queue):
    # Assuming a predefined model architecture
    model = MyModel()
    model.load_state_dict(model_weights_queue.get())

    if torch.cuda.is_available():
        model = model.cuda()

    # Do whatever processing you want with the model

if __name__ == "__main__":
    mp.set_start_method('spawn')

    model = MyModel()
    if torch.cuda.is_available():
        model = model.cuda()

    model_weights = model.state_dict()

    model_weights_queue = mp.Queue()
    model_weights_queue.put(model_weights)

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=worker, args=(rank, model_weights_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

This way, the main process initializes the model, sends the weights to child processes, and each child process then handles the CUDA operations on its own.

This approach circumvents the tricky problem of sending CUDA tensors or models across processes and is more in line with PyTorch's recommended way of handling multiprocessing with CUDA.

'''
