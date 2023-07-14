'''

Speech-to-text


'''

from thespian.actors import Actor
from typing import List
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
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock, Event
from queue import Queue, Empty
import speech_recognition as sr
import re
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple
import argparse

from uniteai.common import ThreadSafeCounter, mk_logger, find_block, get_nested
from uniteai.edit import BlockJob, cleanup_block, init_block
import math
import audioop
import collections
import threading


START_TAG = ':START_TRANSCRIPTION:'
END_TAG = ':END_TRANSCRIPTION:'
NAME = 'transcription'

# A custom logger for just this feature. You can tune the log level to turn
# on/off just this feature's logs.
log_level = logging.DEBUG
log = mk_logger(NAME, log_level)


##################################################

def listen(r: sr.Recognizer, source, external_queue=None):
    assert isinstance(source, sr.AudioSource), "Source must be an audio source"
    assert source.stream is not None, "Audio source must be entered before listening, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"

    seconds_per_buffer = float(source.CHUNK) / source.SAMPLE_RATE
    pause_buffer_count = int(math.ceil(r.pause_threshold / seconds_per_buffer))  # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
    phrase_buffer_count = int(math.ceil(r.phrase_threshold / seconds_per_buffer))  # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
    non_speaking_buffer_count = int(math.ceil(r.non_speaking_duration / seconds_per_buffer))  # maximum number of buffers of non-speaking audio to retain before and after a phrase

    # read audio input for phrases until there is a phrase that is long enough
    elapsed_time = 0  # number of seconds of audio read
    buffer = b""  # an empty buffer means that the stream has ended and there is no data left to read
    while True:
        frames = collections.deque()

        ##########
        # Wait for phrase start, store audio until then
        while True:
            # handle waiting too long for phrase by raising an exception
            elapsed_time += seconds_per_buffer

            buffer = source.stream.read(source.CHUNK)
            if len(buffer) == 0:
                break  # reached end of the stream
            frames.append(buffer)
            if len(frames) > non_speaking_buffer_count:  # ensure we only keep the needed amount of non-speaking buffers
                frames.popleft()

            # detect whether speaking has started on audio input
            energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # energy of the audio signal
            if energy > r.energy_threshold:
                break

            # dynamically adjust the energy threshold using asymmetric weighted average
            if r.dynamic_energy_threshold:
                damping = r.dynamic_energy_adjustment_damping ** seconds_per_buffer  # account for different chunk sizes and rates
                target_energy = energy * r.dynamic_energy_ratio
                r.energy_threshold = r.energy_threshold * damping + target_energy * (1 - damping)

        ##########
        # Keep the audio prefix
        for b in frames:
            external_queue.put((b, source.SAMPLE_RATE, source.SAMPLE_WIDTH))

        ##########
        # Phrase has started, keep reading audio input until the phrase ends

        pause_count, phrase_count = 0, 0
        while True:
            buffer = source.stream.read(source.CHUNK)
            if len(buffer) == 0:
                break  # reached end of the stream

            # TODO: this is hacky, for sending back rate/width
            external_queue.put((buffer,
                                source.SAMPLE_RATE,
                                source.SAMPLE_WIDTH))
            phrase_count += 1

            # check if speaking has stopped for longer than the pause threshold on the audio input
            energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # unit energy of the audio signal within the buffer
            if energy > r.energy_threshold:
                pause_count = 0
            else:
                pause_count += 1
            if pause_count > pause_buffer_count:  # end of the phrase
                break

        # check how long the detected phrase is, and retry listening if the phrase is too short
        phrase_count -= pause_count  # exclude the buffers for the pause before the phrase
        if phrase_count >= phrase_buffer_count or len(buffer) == 0: break  # phrase is long enough or we've reached the end of the stream, so stop listening


def listen_in_background(r: sr.Recognizer, source, callback, external_queue=None):
    assert isinstance(source, sr.AudioSource), "Source must be an audio source"
    running = [True]

    def threaded_listen():
        with source as s:
            while running[0]:
                try:  # listen for 1 second, then check again if the stop function has been called
                    audio = listen(r, s, external_queue)
                except sr.exceptions.WaitTimeoutError:  # listening timed out, just try again
                    pass
                else:
                    if running[0]:
                        callback(r, audio)

    def stopper(wait_for_stop=True):
        running[0] = False
        if wait_for_stop:
            listener_thread.join()  # block until the background thread is done, which can take around 1 second

    listener_thread = threading.Thread(target=threaded_listen)
    listener_thread.daemon = True
    listener_thread.start()
    return stopper

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
        logging.info('Warming up transcription model in separate thread')
        warmup_thread = Thread(target=self._warmup)
        warmup_thread.daemon = True
        warmup_thread.start()

    def listen(self, should_stop):
        def callback(r, audio):
            log.debug('LISTENING CALLBACK called')
            # self.audio_queue.put(audio, block=False)
        stop_listening_fn = listen_in_background(
            self.r,
            self.mic,
            callback,
            external_queue=self.audio_queue,
            # phrase_time_limit=1.0
        )
        return stop_listening_fn

    def transcription_worker(self, uri, edits, should_stop,
                             transcription_worker_is_running):
        transcription_worker_is_running.set()
        audios = []
        sample_rate = None
        sample_width = None
        while not should_stop.is_set():
            try:
                # non-blocking, to more frequently allow the
                # `stop_transcription` signal to end this thread.
                buffer, sample_rate, sample_width = self.audio_queue.get(False)
                audios.append(buffer)
                try:
                    while True:
                        buffer, sample_rate, sample_width = self.audio_queue.get(False)
                        audios.append(buffer)
                except Empty:
                    pass

                log.warn(f'len audio: {len(audios)}')
                # if not sample_rate:  # for future reference
                #     sample_rate = sample_rate
                #     sample_width = sample_width
            except Empty:
                time.sleep(0.2)
                continue

            try:
                audio = sr.audio.AudioData(
                    b''.join(audios),
                    sample_rate,
                    sample_width
                )
                # Debug audio: The audio gets sliced up regularly, how does it
                #              sound when stitched back?
                if log_level == logging.DEBUG:
                    with open("debug_transcription.wav", "wb") as output_file:
                        output_file.write(audio.get_wav_data())
                x = self.recognize(audio)
                if not x:
                    continue

                x = x.strip()
                log.debug(f'TRANSCRIPTION: {x}')
                if filter_out(x):
                    continue

                # Add space to respect next loop of transcription
                job = BlockJob(
                    uri=uri,
                    start_tag=START_TAG,
                    end_tag=END_TAG,
                    text=f'\n{x}\n',
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
        self.stop_listening_fn = lambda x, y: None

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
        self.stop_listening_fn = lambda x, y: None
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
