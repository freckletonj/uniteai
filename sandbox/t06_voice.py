'''

Live streaming mic audio through whisper

deps
  sudo apt install portaudio19-dev

pip deps
  torch
  transformers
  datasets
  soundfile
  librosa
  pyaudio

  SpeechRecognition
  openai-whisper

'''

import speech_recognition as sr
from threading import Thread
from queue import Queue
import os
import re
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = '/home/josh/_/models/whisper-large-v2'
MODEL_SIZE = 'small' # tiny, base, small, medium, large-v2

r = sr.Recognizer()
r.energy_threshold = 2000
r.dynamic_energy_threshold = False # adjust `energy_threshold` if True, based on ambient noise

audio_queue = Queue()

def filter_alphanum(x:str)->str:
    ''' keep only alphanum, not even spaces. '''
    return re.sub(r'\W+', '', x)

filter_list = [
    filter_alphanum(x)
    for x in
    [
        '',
        'bye',
        'you',
        'thank you',
        'thanks for watching',
    ]
]

def filter_out(x:str) -> bool:
    # replace all non-alphanumerics
    x = filter_alphanum(x)
    if len(x) < 4:
        return True
    return x.strip().lower() in filter_list

def color(x:str, color:str):
    colors = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m"
    }
    # default is uncolored
    if color not in colors:
        return x
    return colors[color] + x + colors["reset"]


##################################################
# Transcription Thread

def recognize_worker():
    # this runs in a background thread
    while True:
        audio = audio_queue.get()
        if audio is None: break  # stop processing if the main thread is done
        try:
            x = r.recognize_whisper(audio,
                                    model=MODEL_SIZE,
                                    load_options=dict(
                                        device=device,
                                        download_root=MODEL_PATH,
                                    ),
                                    language='english').strip()
            if filter_out(x):
                continue
            x = color(x, 'red')
            print(f"I think you said: <{x}>. {r.energy_threshold}" )
        except sr.UnknownValueError:
            print("ERROR: could not understand audio")
        audio_queue.task_done()

# start a new thread to recognize audio, while this thread focuses on listening
recognize_thread = Thread(target=recognize_worker)
recognize_thread.daemon = True
recognize_thread.start()


##################################################
# Listening Thread

with sr.Microphone() as source:
    try:
        i = 0
        while True:  # repeatedly listen for phrases and put the resulting audio on the audio processing job queue
            i += 1
            print(color('listening...', 'blue'), end='')
            x = r.listen(source)
            print(color(f'done listening, item #{i}', 'blue'))
            audio_queue.put(x)
    except KeyboardInterrupt:
        pass

audio_queue.join()  # block until all current audio processing jobs are done
audio_queue.put(None)  # tell the recognize_thread to stop
recognize_thread.join()  # wait for the recognize_thread to actually stop
