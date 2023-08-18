'''

In the first example I struggled to recreate the bug, so I'm copying and paring down transcription.py here.

ANSWER:

This isn't implemented yet, but I think the answer is:

1. set_start_method('spawn') on the main process

2. instantiate a cuda context on the main process

3. instantiate models on the main process

4. send the models to processes via torch.multiprocessing.Queue

5. There are rules around when you can delete the model off the main process: https://pytorch.org/docs/stable/multiprocessing.html


'''

import torch.multiprocessing as mp

mp.set_start_method('forkserver', force=True)
mp.set_start_method('spawn', force=True)
mp.set_start_method('forkserver', force=False)
mp.set_start_method('spawn', force=False)


from thespian.actors import ActorSystem, Actor
from threading import Thread, Event
import numpy as np
import speech_recognition as sr
import time
import torch

CONCURRENCY_MODEL = 'multiprocQueueBase'
# CONCURRENCY_MODEL = 'simpleSystemBase'



# device = 'cuda'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


##################################################
# SpeechRecognition

class SpeechRecognition:
    def __init__(self,
                 model_type,

                 # whisper
                 model_path,
                 model_size,
                 volume_threshold):
        self.model_type = model_type
        self.model_path = model_path
        self.model_size = model_size

        # Recognizer
        self.r = sr.Recognizer()
        self.mic = sr.Microphone()

    def recognize(self, audio):
        print(f'MODEL_TYPE: {self.model_type}')
        print(f'AUDIO: {audio}')
        return self.r.recognize_whisper(
            audio,
            model=self.model_size,
            load_options=dict(
                device=device,
                download_root=self.model_path
            ), language='english')

    def _warmup(self):
        ''' Warm up, intended for a separate thread. '''

        # Get model into memory
        empty_audio = sr.AudioData(np.zeros(10), sample_rate=1, sample_width=1)
        self.recognize(empty_audio)
        print(f'Warmed up')

    def warmup(self):
        '''Load whisper model into memory.'''
        print('Warming up transcription model in separate thread')
        warmup_thread = Thread(target=self._warmup)
        warmup_thread.daemon = True
        warmup_thread.start()


##################################################
# Actor

class TranscriptionActor(Actor):
    def __init__(self):
        self.is_running = Event()
        self.speech_recognition = None  # set during initialization

        # set during set_config/start
        self.model_type = 'whisper'
        self.model_path = None
        self.model_size = 'tiny'
        self.volume_threshold = None
        self.stop_fn = lambda: None

    def receiveMessage(self, msg, sender):
        self.speech_recognition = SpeechRecognition(
            self.model_type,

            # whisper
            self.model_path,
            self.model_size,
            self.volume_threshold
        )

        # load the model into GPU
        self.speech_recognition.warmup()


##################################################

actor_system = ActorSystem(CONCURRENCY_MODEL, logDefs={'version': 1, 'logLevel': 'DEBUG'})
worker = actor_system.createActor(TranscriptionActor)
actor_system.tell(worker, None)
