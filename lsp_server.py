'''

An LSP server that connects to the LLM server for doing the brainy stuff.

USAGE:
    python lsp_server.py

'''

from typing import List
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
    TextEdit,
    WorkspaceEdit,
    DidChangeTextDocumentParams,
)
import sys
from threading import Event
import logging
from pygls.protocol import default_converter
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import openai
import yaml

from threading import Thread
from queue import Queue
import speech_recognition as sr
import re


##################################################
# Initialization

##########
# OpenAI Secrets

with open(".secret.yml", 'r') as file:
    secrets = yaml.safe_load(file)
openai.api_key = secrets["OPENAI_API_KEY"]

##########
# Configuration

with open("config.yml", 'r') as file:
    config = yaml.safe_load(file)

    # OpenAI
    OPENAI_COMPLETION_ENGINE = config['openai_completion_engine']
    OPENAI_CHAT_ENGINE = config['openai_chat_engine']
    OPENAI_MAX_LENGTH = config['openai_max_length']

    # Local LLM
    LOCAL_MAX_LENGTH = config['local_max_length']

    LSP_PORT = config['lsp_port']
    LLM_PORT = config['llm_port']
    LLM_URI = config['llm_uri']
    TOP_K = config['top_k']

    # TODO move to config
    TRANSCRIPTION_MODEL_SIZE = 'tiny'
    TRANSCRIPTION_MODEL_PATH = '/home/josh/_/models/whisper-large-v2'

# A sentinel to be used by LSP clients when calling commands. If a CodeAction is
# called (from a dropdown in the text editor), params are defined in the python
# function here. If a Command is called (IE a key sequence), the editor provides
# the values. But if we want the client to defer to the LSP server's config, use
# this sentinel.
FROM_CONFIG = 'FROM_CONFIG'
FROM_CONFIG_CHAT = 'FROM_CONFIG_CHAT'
FROM_CONFIG_COMPLETION = 'FROM_CONFIG_COMPLETION'
##########
# Logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    # level=logging.INFO,
    # level=logging.WARN,
)


##################################################
# Endpoints


class Server(LanguageServer):
    def __init__(self, name, version):
        super().__init__(name, version)
        self.stop_stream = Event()
        self.stop_transcription = Event()

        # A Queue for LLM Jobs. By throwing long running tasks on here, LSP
        # endpoints can return immediately.
        self.executor = ThreadPoolExecutor(max_workers=1)


server = Server("llm-lsp", "0.1.0")


##################################################
# Util

def workspace_edit(uri: str,
                   new_start: Position,
                   new_text: str) -> WorkspaceEdit:
    ''' Build a `WorkspaceEdit` for pygls to send to LSP client. '''
    text_edit = TextEdit(range=Range(start=new_start, end=new_start),
                         new_text=new_text)
    text_document_edit = {
        'textDocument': TextDocumentIdentifier(uri=uri),
        'edits': [text_edit],
    }
    return WorkspaceEdit(document_changes=[text_document_edit])


def extract_range(doc: str, range: Range) -> str:
    '''Extract the highlighted region of the text coming from the client.'''
    lines = doc.split("\n")
    start = range.start
    end = range.end
    if start.line == end.line:
        return lines[start.line][start.character:end.character]
    else:
        start_line_text = lines[start.line][start.character:]
        middle_lines_text = lines[start.line + 1:end.line]
        end_line_text = lines[end.line][:end.character]
        return "\n".join([start_line_text] +
                         middle_lines_text +
                         [end_line_text])


##################################################
# Marker Test
#
# This is just a test to get markers saved and recallable on a per-buffer basis.

# Global variable to store the marker
emacs_marker = {}

@server.command('command.markerSet')
def marker_set(ls: LanguageServer, args: dict):
    global emacs_marker
    doc = args[0]
    params = args[1] # params= [{'emacsMarker': {'line': 3, 'character': 0}}]

    text_document = converter.structure(doc, TextDocumentIdentifier)
    uri = text_document.uri

    # Extract the EmacsMarker
    marker = params['emacsMarker']

    if marker:
        # Each marker will be a dictionary with 'line' and 'character' keys.
        emacs_marker[uri] = Position(marker['line'], marker['character'])

        print(f'SET: Set marker at line: {emacs_marker[uri].line} and character: {emacs_marker[uri].character}')

    else:
        print('No marker received.')

    return {'status': 'success'}

@server.command('command.markerGet')
def marker_get(ls: LanguageServer, args: dict):
    global emacs_marker
    doc = args[0]
    params = args[1] # params= [{'emacsMarker': {'line': 3, 'character': 0}}]

    text_document = converter.structure(doc, TextDocumentIdentifier)
    uri = text_document.uri

    if emacs_marker:
        print(f'GET: Marker is at line: {emacs_marker[uri].line} and character: {emacs_marker[uri].character}')
    else:
        print('No marker set.')

    return {'status': 'success'}


##################################################
# Voice Transcription

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


##########

# Speech Recognition setup
r = sr.Recognizer()
r.energy_threshold = 2000
r.dynamic_energy_threshold = False

audio_queue = Queue()

# Global flag to control when the transcription thread should stop
stop_transcribing = False


##########
# Listening Thread

def listen_worker(ls):
    print('STARTING L WORKER')
    with sr.Microphone() as source:
        try:
            while not ls.stop_transcription.is_set():
                print('LISTEN AUDIO')
                audio_queue.put(r.listen(source))
        except KeyboardInterrupt:
            pass
    print('DONE LISTENING')


##########
# Transcription Thread

def transcribe_worker(ls, args):
    print('STARTING T WORKER')
    text_document = converter.structure(args[0], TextDocumentIdentifier)
    uri = text_document.uri
    doc = ls.workspace.get_document(uri).source

    while not ls.stop_transcription.is_set():
        audio = audio_queue.get()
        print('GOT AUDIO')
        if audio is None: break
        try:
            x = r.recognize_whisper(audio,
                                    model=TRANSCRIPTION_MODEL_SIZE,
                                    load_options=dict(
                                        device='cuda:0',
                                        download_root=TRANSCRIPTION_MODEL_PATH
                                    ), language='english').strip()

            print(f'TRANSCRIPTION: {x}')
            if filter_out(x):
                continue
            server.lsp.send_request(
                "workspace/applyEdit",
                ApplyWorkspaceEditParams(
                    edit=WorkspaceEdit(
                        changes={
                            uri: [
                                TextEdit(
                                    range=Range(start=Position(line=0, character=0),
                                                end=Position(line=0, character=0)),
                                    new_text=x
                                )
                            ]
                        }
                    )
                ))
        except sr.UnknownValueError:
            print("ERROR: could not understand audio")
        audio_queue.task_done()
    print('DONE TRANSCRIBING')

@server.thread()
@server.command('command.transcribeStream')
def transcribe_stream(ls: LanguageServer, args):
    ls.stop_transcription.clear()

    listen_thread = Thread(target=listen_worker, args=(ls,))
    listen_thread.daemon = True
    listen_thread.start()

    transcribe_thread = Thread(target=transcribe_worker, args=(ls, args))
    transcribe_thread.daemon = True
    transcribe_thread.start()

    return



@server.thread()
@server.command('command.stopTranscribeStream')
def stop_transcribe_stream(ls: LanguageServer, args):
    ls.stop_transcription.set()

    # drain audio queue
    while True:
        if audio_queue.empty():
            break
        _ = audio_queue.get()

    # stop audio thread
    audio_queue.put(None)


##################################################
# Local LLM

# For converting jsonified things into proper instances. This is necessary when
# a CodeAction summons a Command, and passes raw JSON to it.
converter = default_converter()

@server.thread()  # multi-threading unblocks emacs
@server.command('command.localLlmStream')
def local_llm_stream(ls: Server, args):
    # print(f'ARGS: {args}')
    # print(f'TYPE: {type(args[0])}')

    # free the event so this call can run
    ls.stop_stream.clear()

    text_document = converter.structure(args[0], TextDocumentIdentifier)
    range = converter.structure(args[1], Range)
    uri = text_document.uri
    doc = ls.workspace.get_document(uri).source

    # Extract the highlighted region
    input_text = extract_range(doc, range)

    def stream(ls):
        try:
            request_data = {
                "text": input_text,
                "max_length": LOCAL_MAX_LENGTH,
                "do_sample": True,
                "top_k": TOP_K,
                "num_return_sequences": 1
            }
            # Stream response from LLM Server
            response = requests.post(f"{LLM_URI}/local_llm_stream",
                                     json=request_data,
                                     stream=True)
            if response.status_code != 200:
                raise Exception(f"POST request to {LLM_URI} failed with status code {response.status_code}")

            # Initialize Cursor as current end position
            cur = Position(line=range.end.line + 1,
                           character=range.end.character)

            # Print "RESPONSE:"
            response_text = '\n\nLOCAL_RESPONSE:\n'
            edit = workspace_edit(uri, cur, response_text)
            params = ApplyWorkspaceEditParams(edit=edit)
            ls.lsp.send_request("workspace/applyEdit", params)
            cur = Position(line=cur.line + 3, character=0)

            # Stream the results to LSP Client
            for line in response.iter_lines():
                response_data = json.loads(line)
                new_text = response_data["generated_text"]
                # ignore empty strings
                if len(new_text) == 0:
                    continue
                edit = workspace_edit(uri, cur, new_text)
                params = ApplyWorkspaceEditParams(edit=edit)
                ls.lsp.send_request("workspace/applyEdit", params)

                # Update the current position, accounting for newlines
                newlines = new_text.count('\n')
                last_line = new_text.rsplit('\n', 1)[-1] if '\n' in new_text else new_text
                last_line_length = len(last_line)
                cur = Position(line=cur.line + newlines,
                               character=last_line_length if newlines else cur.character + len(new_text))

                # For breaking out early
                if ls.stop_stream.is_set():
                    return
        except Exception as e:
            print(f'EXCEPTION: {e}')

    # stream in a thread so this function can return immediately
    ls.executor.submit(stream, ls)

    # Return an empty edit. The actual edits will be carried out by Commands in
    # the `stream` function.
    return WorkspaceEdit()


@server.command('command.localLlmStreamStop')
def local_llm_stream_stop(ls: Server, *args):
    ls.stop_stream.set()
    requests.post(f"{LLM_URI}/local_llm_stream_stop", json='{}')


##################################################
# OpenAI

COMPLETION_ENGINES = [
    "text-davinci-003",
    "text-davinci-002",
    "ada",
    "babbage",
    "curie",
    "davinci",
]

CHAT_ENGINES = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-4",
]


def openai_autocomplete(engine, text, max_length):
    print(f"AND THE ENGINE IS: {engine}")
    if engine in COMPLETION_ENGINES:
        response = openai.Completion.create(
          engine=engine,
          prompt=text,
          max_tokens=max_length,
          stream=True
        )
        for message in response:
            generated_text = message['choices'][0]['text']
            yield generated_text
    elif engine in CHAT_ENGINES:
        response = openai.ChatCompletion.create(
          model=engine,
          messages=[{"role": "user", "content": text}],
          stream=True
        )
        for message in response:
            # different json structure than completion endpoint
            delta = message['choices'][0]['delta']
            if 'content' in delta:
                generated_text = delta['content']
                yield generated_text


@server.thread()
@server.command('command.openaiAutocompleteStream')
def openai_autocomplete_stream(ls: Server, args):
    # free the event so this call can run
    ls.stop_stream.clear()

    text_document = converter.structure(args[0], TextDocumentIdentifier)
    range = converter.structure(args[1], Range)

    # Check for sentinel values to allow LSP client to defer arguments to
    # server's configuration.

    # Engine
    if args[2] == FROM_CONFIG_CHAT:
        engine = OPENAI_CHAT_ENGINE
    elif args[2] == FROM_CONFIG_COMPLETION:
        engine = OPENAI_COMPLETION_ENGINE
    else:
        engine = args[2]

    # Max Length
    if args[3] == FROM_CONFIG:
        max_length = OPENAI_MAX_LENGTH
    else:
        max_length = args[3]

    # Get the document
    uri = text_document.uri
    doc = ls.workspace.get_document(uri).source

    # Extract the highlighted region
    input_text = extract_range(doc, range)

    def stream(ls, engine, text, max_length):
        # Initialize Cursor as current end position
        cur = Position(line=range.end.line + 1, character=range.end.character)

        # Print "RESPONSE:"
        response_text = '\n\nOPENAI_RESPONSE:\n'
        edit = workspace_edit(uri, cur, response_text)
        params = ApplyWorkspaceEditParams(edit=edit)
        ls.lsp.send_request("workspace/applyEdit", params)
        cur = Position(line=cur.line + 3, character=0)

        # Call the openai_autocomplete function with request and stream parameters
        for new_text in openai_autocomplete(engine, text, max_length):
            edit = workspace_edit(uri, cur, new_text)
            params = ApplyWorkspaceEditParams(edit=edit)
            ls.lsp.send_request("workspace/applyEdit", params)

            # Update the current position, accounting for newlines
            newlines = new_text.count('\n')
            last_line = new_text.rsplit('\n', 1)[-1] if '\n' in new_text else new_text
            last_line_length = len(last_line)
            cur = Position(line=cur.line + newlines,
                           character=last_line_length if newlines else cur.character + len(new_text))

            # For breaking out early
            if ls.stop_stream.is_set():
                break

    ls.executor.submit(stream, ls, engine, input_text, max_length)
    return WorkspaceEdit()


@server.command('command.openaiAutocompleteStreamStop')
def openai_autocomplete_stream_stop(ls: Server, *args):
    ls.stop_stream.set()


##################################################
# Go

@server.feature('workspace/didChangeConfiguration')
def workspace_did_change_configuration(ls: Server, *args):
    return []


@server.feature("textDocument/codeAction")
def code_action(params: CodeActionParams) -> List[CodeAction]:
    text_document = params.text_document
    range = params.range
    return [
        # OpenAI GPT
        CodeAction(
            title='OpenAI GPT',
            kind=CodeActionKind.Refactor,
            command=Command(
                title='Local LLM',
                command='command.openaiAutocompleteStream',
                # Note: these arguments get jsonified, not passed as python objs
                arguments=[text_document, range, OPENAI_COMPLETION_ENGINE, OPENAI_MAX_LENGTH]
            )
        ),

        # OpenAI ChatGPT
        CodeAction(
            title='OpenAI ChatGPT',
            kind=CodeActionKind.Refactor,
            command=Command(
                title='Local LLM',
                command='command.openaiAutocompleteStream',
                # Note: these arguments get jsonified, not passed as python objs
                arguments=[text_document, range, OPENAI_CHAT_ENGINE, OPENAI_MAX_LENGTH]
            )
        ),

        # LLM
        CodeAction(
            title='Local LLM',
            kind=CodeActionKind.Refactor,
            command=Command(
                title='Local LLM',
                command='command.localLlmStream',
                # Note: these arguments get jsonified, not passed directly
                arguments=[text_document, range]
            )
        ),

        # Stop LLM
        CodeAction(
            title='Stop Local LLM',
            kind=CodeActionKind.Refactor,
            command=Command(
                title='Stop Streaming',
                command='command.localLlmStreamStop'
            )
        ),
    ]


if __name__ == '__main__':
    print(f'Starting LSP on port: {LSP_PORT}')
    server.start_tcp(host='localhost', port=LSP_PORT)
