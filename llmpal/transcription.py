'''

Transcribe Speech.

'''


from typing import List
import pygls
from pygls.server import LanguageServer
from lsprotocol.types import (
    ApplyWorkspaceEditParams,
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Command,
    Position,
    Range,
    TextDocumentIdentifier,
    VersionedTextDocumentIdentifier,
    TextEdit,
    WorkspaceEdit,
    DidChangeTextDocumentParams,
)
import sys
import logging
from pygls.protocol import default_converter
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import openai
import yaml
from threading import Thread, Lock, Event
from queue import Queue, Empty
import speech_recognition as sr
import re
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple
import re
import itertools

from llmpal.common import ThreadSafeCounter


##########
# Util

def filter_alphanum(x: str) -> str:
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


def filter_out(x: str) -> bool:
    x = filter_alphanum(x)
    if len(x) < 4:
        return True
    return x.strip().lower() in filter_list

def remove_regex(ls, tags, doc, uri, version):
    removals = itertools.chain.from_iterable(
        [find_pattern_in_document(doc, tag)for tag in tags]
    )
    removals = [
        (Position(i, s),
         Position(i, e),
         ''  # remove original tags
         )
        for i, s, e in removals
    ]
    edit = workspace_edits(uri,
                           version,
                           removals
                           )
    print(f'{edit} \nWORKSPACE EDITS')
    params = ApplyWorkspaceEditParams(edit=edit)
    futu = ls.lsp.send_request("workspace/applyEdit", params)
    # res = futu.result()
    # print(f'RESTULT: {res}')


##########
# Speech Recognition setup

class SpeechRecognition:
    def __init__(self,
                 model_path,
                 model_size,
                 volume_threshold):

        self.model_path = model_path
        self.model_size = model_size

        # Recognizer
        self.r = sr.Recognizer()
        self.r.energy_threshold = volume_threshold
        self.r.dynamic_energy_threshold = False
        self.audio_queue = Queue()

        # Keep track of the iteration when a thread was started. That way, if
        # it had a blocking operation (like `r.listen`) that should have been
        # terminated, but couldn't because the thread was blocked, well, now we
        # can deprecate that thread.
        self.transcription_counter = ThreadSafeCounter()

    def _warmup(self):
        ''' Warm up, intended for a separate thread. '''
        empty_audio = sr.AudioData(np.zeros(10), sample_rate=1, sample_width=1)
        self.r.recognize_whisper(empty_audio,
                                 model=self.model_size,
                                 load_options=dict(
                                     device='cuda:0',
                                     download_root=self.model_path
                                 ), language='english')
        logging.info('Warmed up Whisper')

    def warmup(self):
        '''Load whisper model into memory.'''
        logging.info('Warming up whisper in separate thread')
        warmup_thread = Thread(target=self._warmup)
        warmup_thread.daemon = True
        warmup_thread.start()

    def listen_worker(self, current_i, ls):
        print('STARTING L WORKER')
        with sr.Microphone() as source:
            try:
                while not ls.stop_transcription.is_set():
                    print('LISTEN AUDIO')
                    # blocks, and prevents while-loop from terminating when
                    # required
                    audio = self.r.listen(source)

                    # check if thread was deprecated while r.listen blocked
                    if self.transcription_counter.get() > current_i:
                        break
                    self.audio_queue.put(audio, block=False)
            except KeyboardInterrupt:
                pass
        print('DONE LISTENING')

    def transcribe_worker(self, current_i, ls, uri, version):
        running_transcription = ""  # keep, for calculating offsets from marker
        while not ls.stop_transcription.is_set():
            try:
                # non-blocking, to more frequently allow the
                # `stop_transcription` signal to end this thread.
                audio = self.audio_queue.get(False)
            except Empty:
                time.sleep(0.2)
                continue

            # break out if this thread has been deprecated
            if self.transcription_counter.get() > current_i:
                break

            try:
                x = self.r.recognize_whisper(audio,
                                             model=self.model_size,
                                             load_options=dict(
                                                 device='cuda:0',
                                                 download_root=self.model_path
                                             ), language='english').strip()

                print(f'TRANSCRIPTION: {x}')
                if filter_out(x):
                    continue

                # Add space to respect next loop of transcription
                running_transcription += x + ' '
                job = Job(
                    uri,
                    f':START_TRANSCRIPTION:{current_i}:',
                    f':END_TRANSCRIPTION:{current_i}:',
                    f'\n{running_transcription}\n'
                )
                ls.edit_jobs['transcription'].put(job)
            except sr.UnknownValueError:
                print("ERROR: could not understand audio")
            self.audio_queue.task_done()
        print('DONE TRANSCRIBING')

    def terminate_transcription(self, ls):
        ls.stop_transcription.set()
        ls.is_transcription_running.clear()

        # drain audio queue
        try:
            while True:
                self.audio_queue.get(False)
        except Empty:
            pass


@server.command('command.transcribeStream')
def transcribe_stream(ls: LanguageServer, args):
    if ls.is_transcription_running.is_set():
        terminate_transcription(ls)
        time.sleep(0.1)  # allow threads to clean up (TODO: messy)

    ls.is_transcription_running.set()
    ls.stop_transcription.clear()

    # remember the current iteration when this thread was started.
    current_i = transcription_counter.increment()

    # Prepare args
    text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
    uri = text_document.uri
    doc = ls.workspace.get_document(uri)
    version = doc.version

    # Insert new Tags
    cur = converter.structure(args[1], Position)
    tags = '\n'.join([
        f':START_TRANSCRIPTION:{current_i}:',
        f':END_TRANSCRIPTION:{current_i}:',
    ])
    edit = workspace_edit(uri, version, cur, cur, tags)
    params = ApplyWorkspaceEditParams(edit=edit)
    ls.lsp.send_request("workspace/applyEdit", params)

    listen_thread = Thread(target=listen_worker, args=(current_i, ls))
    listen_thread.daemon = True
    listen_thread.start()

    transcribe_thread = Thread(target=transcribe_worker,
                               args=(current_i, ls, uri, version))
    transcribe_thread.daemon = True
    transcribe_thread.start()

    return {'status': 'success'}

@server.thread()
@server.command('command.stopTranscribeStream')
def stop_transcribe_stream(ls: LanguageServer, args):
    start_tag = r':START_TRANSCRIPTION:\d+:'
    end_tag   = r':END_TRANSCRIPTION:\d+:'

    text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
    uri = text_document.uri
    doc = ls.workspace.get_document(uri)

    terminate_transcription(ls)

    version = doc.version
    remove_regex(ls, [start_tag, end_tag], doc.source, uri, version)

    return {'status': 'success'}
