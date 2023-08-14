'''

Text-to-Speech

pip install spacy
python -m spacy download en_core_web_sm

'''

import re
import os
from lsprotocol.types import (
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Command,
    Range,
    TextDocumentIdentifier,
    WorkspaceEdit,
)
from threading import Event, Thread
from thespian.actors import Actor
import argparse
import logging
import importlib

from uniteai.common import extract_range, mk_logger
from uniteai.server import Server

os.environ["SUNO_USE_SMALL_MODELS"] = "True"  # must do before `import bark`


# Temperatures
#   1.0 = more diverse
#   0.0 = more conservative
TEXT_TEMP = 0.6
WAVEFORM_TEMP = 0.8

# START_TAG = ':START_DOCUMENT_CHAT:'
# END_TAG = ':END_DOCUMENT_CHAT:'
NAME = 'text_to_speech'

# A custom logger for just this feature. You can tune the log level to turn
# on/off just this feature's logs.
log = mk_logger(NAME, logging.DEBUG)


def prep_prompts(text, desired_length=200, max_length=300):
    """Split text it into chunks of a desired length trying to keep sentences
    intact.

    From tortoise-tts"""
    # normalize text, remove redundant whitespace and convert non-ascii quotes
    # to ascii
    text = re.sub(r'\n\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[“”]', '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the
                # desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # no full sentences, seek back until we are not in the middle
                # of a word and split there
                while c not in '!?.\n ' and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in '!?\n' or (c == '.' and peek(1) in '\n ')):
            # seek forward if we have consecutive boundary markers but still
            # within the max length
            while pos < len(text) - 1 and len(current) < max_length and peek(1) in '!?.':
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in '\n ':
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r'^[\s\.,;:!?]*$', s)]

    return rv

# def prep_prompts(text_prompt: str):
#     ''' Bark prompts should ideally be around 14seconds once translated. Here's
#     a heuristic for that. '''

#     # split into sentences (nltk)
#     sentences = nltk.sent_tokenize(text_prompt)  # import separately since it's slow to load

#     # 2 sentences = 14 seconds
#     return [sentences[i] + " " + sentences[i + 1]
#             for i in range(0, len(sentences), 2)]


class TextToSpeechActor(Actor):
    def __init__(self):
        self.should_stop = Event()
        self.is_running = False
        self.audio_history = None  # bark can mimic this history

    def receiveMessage(self, msg, sender):
        command = msg.get('command')
        doc = msg.get('doc')
        edits = msg.get('edits')

        log.debug(f'''
%%%%%%%%%%
ACTOR RECV: {msg["command"]}
ACTOR STATE:
locked: {self.should_stop.is_set()}
%%%%%%%%%%''')

        ##########
        #
        if command == 'save':
            prompt = msg.get('prompt')
            self.save(prompt)

        ##########
        #
        if command == 'play':
            prompt = msg.get('prompt')
            self.play(prompt)

        ##########
        # Stop
        elif command == 'stop':
            self.stop()

        ##########
        # Set Config
        elif command == 'initialize':
            log.info('Warming up TTS model in separate thread')
            warmup_thread = Thread(target=self._warmup)
            warmup_thread.daemon = True
            warmup_thread.start()

    def _warmup(self):
        ''' Warm up, intended for a separate thread. '''
        # global nltk
        global bark
        global sounddevice

        # nltk = importlib.import_module('nltk')  # loads slowly
        # nltk.download('punkt')

        bark = importlib.import_module('bark')
        sounddevice = importlib.import_module('sounddevice')

        bark.preload_models()
        log.info('Warmed up.')

    def save(self, prompt):
        prompt = prompt.strip()
        if re.findall(r'.+_speaker_.+', prompt):
            self.audio_history = prompt
            return True

        log.debug(f'SAVE: {str(self.audio_history)[:100]}')
        self.should_stop.clear()

        history, audio_array = bark.generate_audio(
            prompt,
            text_temp=TEXT_TEMP,
            waveform_temp=WAVEFORM_TEMP,
            output_full=True,
            silent=True  # no progress bar
        )
        self.audio_history = history
        # self.audio_history = prompt
        sounddevice.play(audio_array, samplerate=bark.SAMPLE_RATE)

    def play(self, prompt):
        log.debug(f'PLAY: {str(self.audio_history)[:100]}')
        self.should_stop.clear()
        sentences = prep_prompts(prompt)
        for sentence in sentences:
            # break early
            if self.should_stop.is_set():
                return False
            audio_array = bark.generate_audio(
                sentence.strip(),
                history_prompt=self.audio_history,
                text_temp=TEXT_TEMP,
                waveform_temp=WAVEFORM_TEMP,
                silent=True  # no progress bar
            )
            # break early
            if self.should_stop.is_set():
                return False
            sounddevice.wait()  # wait for previous gen
            sounddevice.play(audio_array, samplerate=bark.SAMPLE_RATE)

    def stop(self):
        log.debug('ACTOR STOP')
        if not self.is_running:
            log.info('WARN: ON_STOP_BUT_STOPPED')
        self.should_stop.set()
        log.debug('FINALLY STOPPED')


def code_action_text_to_speech_save(params: CodeActionParams):
    ''' A code_action triggers a command, which sends a message to the Actor,
    to handle it. '''
    text_document = params.text_document
    # position of the highlighted region in the client's editor
    range = params.range
    return CodeAction(
        title='TextToSpeech Save History',
        kind=CodeActionKind.Refactor,
        command=Command(
            title='TextToSpeech Save History',
            command='command.textToSpeechSave',
            # Note: these arguments get jsonified, not passed as python objs
            arguments=[text_document, range]
        )
    )

def code_action_text_to_speech_play(params: CodeActionParams):
    ''' A code_action triggers a command, which sends a message to the Actor,
    to handle it. '''
    text_document = params.text_document
    # position of the highlighted region in the client's editor
    range = params.range
    return CodeAction(
        title='TextToSpeech Play',
        kind=CodeActionKind.Refactor,
        command=Command(
            title='TextToSpeech Play',
            command='command.textToSpeechPlay',
            # Note: these arguments get jsonified, not passed as python objs
            arguments=[text_document, range]
        )
    )


##################################################
# Setup
#
# NOTE: In `.uniteai.yml`, just add `uniteai.text_to_speech` under `modules`,
#       and this will automatically get built into the server at runtime.
#

def configure(config_yaml):
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    return args


def initialize(config, server):

    # Actor
    server.add_actor(NAME, TextToSpeechActor)

    # Initialize configuration in Actor
    server.tell_actor(NAME, {
        'command': 'initialize',
        'config': config,
    })

    # CodeActions
    server.add_code_action(code_action_text_to_speech_save)
    server.add_code_action(code_action_text_to_speech_play)

    # Modify Server
    @server.thread()
    @server.command('command.textToSpeechSave')
    def text_to_speech_save(ls: Server, args):
        if len(args) != 2:
            log.error(f'command.textToSpeechSave: Wrong arguments, received: {args}')
        text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
        range = ls.converter.structure(args[1], Range)
        uri = text_document.uri
        doc = ls.workspace.get_document(uri)
        doc_source = doc.source

        # Extract the highlighted region
        prompt = extract_range(doc_source, range)

        # Send a message to start the stream
        actor_args = {
            'command': 'save',
            'uri': uri,
            'prompt': prompt,
            'edits': ls.edits,
            'doc': doc_source,
        }
        ls.tell_actor(NAME, actor_args)

        # Return null-edit immediately (the rest will stream)
        return WorkspaceEdit()

    # Modify Server
    @server.thread()
    @server.command('command.textToSpeechPlay')
    def text_to_speech_play(ls: Server, args):
        if len(args) != 2:
            log.error(f'command.textToSpeechPlay: Wrong arguments, received: {args}')
        text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
        range = ls.converter.structure(args[1], Range)
        uri = text_document.uri
        doc = ls.workspace.get_document(uri)
        doc_source = doc.source

        # Extract the highlighted region
        prompt = extract_range(doc_source, range)

        # Send a message to start the stream
        actor_args = {
            'command': 'play',
            'uri': uri,
            'prompt': prompt,
            'edits': ls.edits,
            'doc': doc_source,
        }
        ls.tell_actor(NAME, actor_args)

        # Return null-edit immediately (the rest will stream)
        return WorkspaceEdit()
