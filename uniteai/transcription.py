'''

Speech-to-text


'''

from thespian.actors import Actor
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
import argparse

from uniteai.common import ThreadSafeCounter, mk_logger, find_block, get_nested
from uniteai.edit import BlockJob, cleanup_block, init_block


START_TAG = ':START_TRANSCRIPTION:'
END_TAG = ':END_TRANSCRIPTION:'
NAME = 'transcription'

# A custom logger for just this feature. You can tune the log level to turn
# on/off just this feature's logs.
log = mk_logger(NAME, logging.DEBUG)


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
        self.r.energy_threshold = volume_threshold
        self.r.dynamic_energy_threshold = False
        self.audio_queue = Queue()

        # Keep track of the iteration when a thread was started. That way, if
        # it had a blocking operation (like `r.listen`) that should have been
        # terminated, but couldn't because the thread was blocked, well, now we
        # can deprecate that thread.
        self.transcription_counter = ThreadSafeCounter()

    def recognize(self, audio):
        log.debug(f'MODEL_TYPE: {self.model_type}')
        log.debug(f'AUDIO: {audio}')
        if self.model_type == 'vosk':
            return self.r.recognize_whisper(
                audio,
                load_options=dict(
                    device='cuda:0',
                    download_root=self.model_path
                ), language='en')

        if self.model_type == 'whisper':
            return self.r.recognize_whisper(
                audio,
                model=self.model_size,
                load_options=dict(
                    device='cuda:0',
                    download_root=self.model_path
                ), language='english')

    def _warmup(self):
        ''' Warm up, intended for a separate thread. '''
        empty_audio = sr.AudioData(np.zeros(10), sample_rate=1, sample_width=1)
        self.recognize(empty_audio)
        logging.info('Warmed up transcription model')

        # TODO: Transcription needs to be tuned better to deal with ambient
        # noise, and appropriate volume levels
        #
        logging.info('Adjusting thresholds for ambient noise')
        with sr.Microphone() as source:
            self.r.adjust_for_ambient_noise(source)


    def warmup(self):
        '''Load whisper model into memory.'''
        logging.info('Warming up whisper in separate thread')
        warmup_thread = Thread(target=self._warmup)
        warmup_thread.daemon = True
        warmup_thread.start()

    def listen(self, should_stop):
        def callback(r, audio):
            log.debug('LISTENING CALLBACK called')
            self.audio_queue.put(audio, block=False)
        stop_listening_fn = self.r.listen_in_background(
            sr.Microphone(),
            callback
        )
        return stop_listening_fn

    def transcription_worker(self, uri, edits, should_stop,
                             transcription_worker_is_running):
        transcription_worker_is_running.set()
        running_transcription = ""
        while not should_stop.is_set():
            try:
                # non-blocking, to more frequently allow the
                # `stop_transcription` signal to end this thread.
                audio = self.audio_queue.get(False)
            except Empty:
                time.sleep(0.2)
                continue

            try:
                x = self.recognize(audio)
                if not x:
                    continue

                x = x.strip()
                log.debug(f'TRANSCRIPTION: {x}')
                if filter_out(x):
                    continue

                # Add space to respect next loop of transcription
                running_transcription += x + ' '
                job = BlockJob(
                    uri=uri,
                    start_tag=START_TAG,
                    end_tag=END_TAG,
                    text=f'\n{running_transcription}\n',
                    strict=False,
                )
                edits.add_job(NAME, job)
            except sr.UnknownValueError:
                log.debug("ERROR: could not understand audio")
            self.audio_queue.task_done()

        cleanup_block(NAME, [START_TAG, END_TAG], uri, edits)
        transcription_worker_is_running.clear()
        log.debug('DONE TRANSCRIBING')


##################################################
# Actor

class TranscriptionActor(Actor):
    def __init__(self):
        self.transcription_worker_is_running = Event()
        self.should_stop = Event()
        self.tags = [START_TAG, END_TAG]
        self.speech_recognition = None  # set during initialization
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.transcription_thread_future = None

        # set during set_config/start
        self.model_path = None
        self.model_size = None
        self.volume_threshold = None
        self.stop_listening_fn = lambda x,y: None

    def receiveMessage(self, msg, sender):
        command = msg.get('command')
        edits = msg.get('edits')
        tw_set = self.transcription_worker_is_running.is_set()
        log.debug(f'''
%%%%%%%%%%
ACTOR RECV: {msg["command"]}
ACTOR STATE:
transcription_worker is running={tw_set}
should_stop: {self.should_stop.is_set()}
transcription_thread_future: {self.transcription_thread_future}

EDITS STATE:
job_thread alive: {edits.job_thread.is_alive() if edits and edits.job_thread else "NOT STARTED"}
%%%%%%%%%%
''')

        ##########
        # Start
        if command == 'start':
            uri = msg.get('uri')
            cursor_pos = msg.get('cursor_pos')
            doc = msg.get('doc')

            # check if block already exists
            start_ixs, end_ixs = find_block(START_TAG,
                                            END_TAG,
                                            doc)
            # make block
            if not (start_ixs and end_ixs):
                init_block(NAME, self.tags, uri, cursor_pos, edits)

            self.start(uri, cursor_pos, edits)

        ##########
        # Stop
        elif command == 'stop':
            self.stop()

        ##########
        # Set Config
        elif command == 'set_config':
            config = msg.get('config')
            self.model_type = config.transcription_model_type
            self.volume_threshold = config.transcription_volume_threshold
            self.model_path = config.transcription_model_path

            # Vosk
            if self.model_type == 'vosk':
                pass

            # Whisper
            if self.model_type == 'whisper':
                self.model_size = config.transcription_model_size

        elif command == 'initialize':
            log.debug(f'INIT TYPE: {self.model_type}')
            self.speech_recognition = SpeechRecognition(
                self.model_type,

                # whisper
                self.model_path,
                self.model_size,
                self.volume_threshold
            )

            # load the model into GPU
            self.speech_recognition.warmup()

    def start(self, uri, cursor_pos, edits):
        tw_set = self.transcription_worker_is_running.is_set()
        if tw_set:
            log.info(f'WARN: ON_START_BUT_RUNNING. '
                     f'transcription_worker is running={tw_set}')
            return
        log.debug('ACTOR START')
        self.should_stop.clear()

        # Audio Listener
        self.stop_listening_fn = self.speech_recognition.listen(self.should_stop)

        # Transcriber
        self.transcription_thread_future = self.executor.submit(
            self.speech_recognition.transcription_worker,
            uri, edits, self.should_stop, self.transcription_worker_is_running)

        log.debug('START CAN RETURN')

    def stop(self):
        log.debug('ACTOR STOP')
        tw_set = self.transcription_worker_is_running.is_set()
        if not tw_set:
            log.info('WARN: ON_STOP_BUT_STOPPED'
                     f'transcription_worker is running={tw_set}')
            return False

        self.should_stop.set()
        self.stop_listening_fn(wait_for_stop=False)

        if self.transcription_thread_future:
            log.debug('Waiting for audio `transcription_thread_future` to terminate')
            self.transcription_thread_future.result()  # block, wait to finish
            self.transcription_thread_future = None  # reset

        self.should_stop.clear()
        self.stop_listening_fn = lambda x,y: None
        log.debug('FINALLY STOPPED')


##########
# Util

def filter_alphanum(x: str) -> str:
    ''' keep only alphanum, not even spaces nor punctuation. '''
    return re.sub(r'\W+', '', x)


filter_list = [
    filter_alphanum(x)  # removes spaces
    for x in
    [  # quirks of Whisper
        '',
        'bye',
        'you',
        'thank you',
        'thanks for watching',
    ]
]


def filter_out(x: str) -> bool:
    x = filter_alphanum(x)
    # if len(x) < 4:  # weed out short utterances
    #     return True
    return x.strip().lower() in filter_list


def code_action_transcribe(params: CodeActionParams):
    '''Trigger a ChatGPT response. A code action calls a command, which is set
    up below to `tell` the actor to start streaming a response. '''
    text_document = params.text_document
    range = params.range  # lsp spec only provides `Range`
    cursor_pos = range.end
    return CodeAction(
        title='Transcribe',
        kind=CodeActionKind.Refactor,
        command=Command(
            title='Transcribe',
            command='command.transcribe',
            # Note: these arguments get jsonified, not passed as python objs
            arguments=[text_document, cursor_pos]
        )
    )


##################################################

def configure(config_yaml):
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcription_model_type', default=get_nested(config_yaml, ['transcription', 'model_type']))
    parser.add_argument('--transcription_model_path', default=get_nested(config_yaml, ['transcription', 'model_path']))
    parser.add_argument('--transcription_volume_threshold', default=get_nested(config_yaml, ['transcription', 'volume_threshold']))

    # whisper
    parser.add_argument('--transcription_model_size', default=get_nested(config_yaml, ['transcription', 'model_size']))

    # bc this is only concerned with transcription params, do not error if extra
    # params are sent via cli.
    args, _ = parser.parse_known_args()
    return args


def initialize(config: argparse.Namespace, server):
    # Actor
    server.add_actor(NAME, TranscriptionActor)
    server.tell_actor(NAME, {
        'command': 'set_config',
        'config': config
    })
    server.tell_actor(NAME, {
        'command': 'initialize'
    })

    # CodeActions
    server.add_code_action(code_action_transcribe)

    # Modify Server
    @server.thread()
    @server.command('command.transcribe')
    def transcribe_stream(ls: LanguageServer, args):
        if len(args) != 2:
            log.error(f'command.transcribe: Wrong arguments, received: {args}')
        # Prepare args
        text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
        uri = text_document.uri
        doc = ls.workspace.get_document(uri)
        cursor_pos = ls.converter.structure(args[1], Position)
        actor_args = {
            'command': 'start',
            'uri': uri,
            'doc': doc.source,
            'edits': ls.edits,
            'cursor_pos': cursor_pos,
        }
        ls.tell_actor(NAME, actor_args)
        return {'status': 'success'}
