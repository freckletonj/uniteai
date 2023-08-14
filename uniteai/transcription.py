'''

Speech-to-text


'''

from thespian.actors import Actor
from pygls.server import LanguageServer
from lsprotocol.types import (
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Command,
    Position,
    TextDocumentIdentifier,
)
import logging
from threading import Thread, Event
from queue import Queue, Empty
import speech_recognition as sr
import re
import numpy as np
import time
import argparse
import threading
from functools import partial
import torch

from uniteai.common import mk_logger, find_block, get_nested
from uniteai.edit import BlockJob, cleanup_block, init_block

device = 'cuda' if torch.cuda.is_available() else 'cpu'

START_TAG = ':START_TRANSCRIPTION:'
END_TAG = ':END_TRANSCRIPTION:'
NAME = 'transcription'

# A custom logger for just this feature. You can tune the log level to turn
# on/off just this feature's logs.
log_level = logging.WARN
log = mk_logger(NAME, log_level)


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
        self.sample_rate = None
        self.sample_width = None
        self.r.energy_threshold = volume_threshold
        self.r.dynamic_energy_threshold = False

    def recognize(self, audio):
        log.debug(f'MODEL_TYPE: {self.model_type}')
        log.debug(f'AUDIO: {audio}')
        if self.model_type == 'vosk':
            return self.r.recognize_whisper(
                audio,
                load_options=dict(
                    device=device,
                    download_root=self.model_path
                ), language='en')

        if self.model_type == 'whisper':
            return self.r.recognize_whisper(
                audio,
                model=self.model_size,
                load_options=dict(
                    device=device,
                    download_root=self.model_path
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
        log.info(f'Warmed up. sample_rate={self.sample_rate}, sample_width={self.sample_width}')

    def warmup(self):
        '''Load whisper model into memory.'''
        log.info('Warming up transcription model in separate thread')
        warmup_thread = Thread(target=self._warmup)
        warmup_thread.daemon = True
        warmup_thread.start()

    def listen_(self,
                queue: Queue,
                should_stop: Event):
        with sr.Microphone() as s:
            while not should_stop.is_set():
                buf = s.stream.read(s.CHUNK)
                queue.put(buf)

    def transcription_(self,
                       audio_queue,
                       transcription_callback,
                       finished_callback,
                       should_stop):
        audios = []
        while not should_stop.is_set():
            try:
                # non-blocking, to more frequently allow the
                # `stop_transcription` signal to end this thread.
                buffer = audio_queue.get(False)

                # TODO: can we more intelligently separate silence from speech?
                # energy = audioop.rms(buffer, self.sample_width)
                audios.append(buffer)
                try:
                    while True:
                        buffer = audio_queue.get(False)
                        audios.append(buffer)
                except Empty:
                    pass

                log.debug(f'len audio: {len(audios)}')

            except Empty:
                time.sleep(0.2)
                continue

            try:
                audio = sr.audio.AudioData(
                    b''.join(audios),
                    self.sample_rate,
                    self.sample_width
                )
                # Debug audio: The audio gets sliced up regularly, how does it
                #              sound when stitched back?
                if log_level == logging.DEBUG:
                    with open("debug_transcription.wav", "wb") as output_file:
                        output_file.write(audio.get_wav_data())

                # breakout if needed
                if should_stop.is_set():
                    break

                # Speech-to-text
                x = self.recognize(audio)

                # Nothing recognized
                if not x:
                    continue

                x = x.strip()
                if filter_out(x):
                    continue

                # breakout if needed
                if should_stop.is_set():
                    break

                transcription_callback(x)

            except sr.UnknownValueError:
                log.debug("ERROR: could not understand audio")
            audio_queue.task_done()

        finished_callback()
        log.debug('DONE TRANSCRIBING')

    def go(self,
           transcription_callback,
           finished_callback):
        audio_queue = Queue()
        should_stop = Event()

        # Listener Thread
        l_thread = threading.Thread(
            target=self.listen_,
            args=(audio_queue, should_stop))
        l_thread.daemon = True
        l_thread.start()

        # Transcription Thread
        t_thread = threading.Thread(
            target=self.transcription_,
            args=(audio_queue,
                  transcription_callback,
                  finished_callback,
                  should_stop))
        t_thread.daemon = True
        t_thread.start()

        def stop_fn():
            log.debug('stop_fn called')
            should_stop.set()
            l_thread.join()
            t_thread.join()

        return stop_fn


##################################################
# Actor

class TranscriptionActor(Actor):
    def __init__(self):
        self.is_running = Event()
        self.tags = [START_TAG, END_TAG]
        self.speech_recognition = None  # set during initialization

        # set during set_config/start
        self.model_path = None
        self.model_size = None
        self.volume_threshold = None
        self.stop_fn = lambda: None

    def receiveMessage(self, msg, sender):
        command = msg.get('command')
        edits = msg.get('edits')
        tw_set = self.is_running.is_set()
        log.debug(f'''
%%%%%%%%%%
ACTOR RECV: {msg["command"]}
ACTOR STATE:
transcription_worker is running={tw_set}

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

    def transcription_callback(self, edits, uri, text):
        # Add space to respect next loop of transcription
        log.debug(f'TRANSCRIBED: {text}')
        job = BlockJob(
            uri=uri,
            start_tag=START_TAG,
            end_tag=END_TAG,
            text=f'\n{text}\n',
            strict=False,
        )
        edits.add_job(NAME, job)

    def finished_callback(self, edits, uri):
        log.debug(f'FINISHED CALLBACK: {uri}')
        cleanup_block(NAME, [START_TAG, END_TAG], uri, edits)

    def start(self, uri, cursor_pos, edits):
        if self.is_running.is_set():
            log.info('WARN: ON_START_BUT_RUNNING.')
            return False
        self.stop_fn = self.speech_recognition.go(
            partial(self.transcription_callback, edits, uri),
            partial(self.finished_callback, edits, uri))
        self.is_running.set()
        log.debug('START CAN RETURN')

    def stop(self):
        log.debug('ACTOR STOP')
        if not self.is_running.is_set():
            log.info('WARN: ON_STOP_BUT_STOPPED')
            return False
        self.stop_fn()
        self.is_running.clear()
        self.stop_fn = lambda: None
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

    # bc this is only concerned with transcription params, do not error if
    # extra params are sent via cli.
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
