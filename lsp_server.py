'''

An LSP server that connects to the LLM server for doing the brainy stuff.

USAGE:
    python lsp_server.py

TODO:
  - collect regions
  - send regions to handler
  - handler sends edits to job queue


AutoAssistant
  - init with streaming function
  - called with initial prompt
  - adds edit jobs to a specific block (by tags)
  - on exit, cleans up the block tags


'''


import sys
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
    # VersionedTextDocumentIdentifier,
    WorkspaceEdit,
)
from pygls.protocol import default_converter
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import openai
import yaml
from threading import Event
from thespian.actors import Actor, ActorSystem
import logging

import llmpal.edit as edit
from llmpal.edit import Edits, BlockJob, InsertJob, DeleteJob
from llmpal.common import extract_range, find_block, workspace_edit, mk_logger


##########
# Logging
#
# Change the basicConfig level to be the generous DEBUG, quiet the libraries,
# and make custom loggers with their own debug levels. This helps especially
# with debugging weird concurrency quirks, ie allowing different processes to
# report things as needed, in a way that can be easily turned on and off.

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,  # Globally allow any level. Other loggers can narrow.
)

# Tune this libs loggers as needed
log_local_llm = mk_logger('local_llm', logging.WARN)
log_openai = mk_logger('openai', logging.WARN)

# Quiet the libs a little
logging.getLogger('pygls.feature_manager').setLevel(logging.WARN)
logging.getLogger('pygls.protocol').setLevel(logging.WARN)
logging.getLogger('Thespian').setLevel(logging.WARN)


# NOTE: Thespian has a domineering logging methodology. To change their default
#       formatter, see: `thespian.system.simpleSystemBase`.
#
#       Also, https://github.com/thespianpy/Thespian/issues/73


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

    # Transcription
    TRANSCRIPTION_MODEL_SIZE = config['transcription_model_size']
    TRANSCRIPTION_MODEL_PATH = config['transcription_model_path']
    TRANSCRIPTION_ENERGY_THRESHOLD = config['transcription_energy_threshold']


# A sentinel to be used by LSP clients when calling commands. If a CodeAction
# is called (from a dropdown in the text editor), params are defined in the
# python function here. If a Command is called (IE a key sequence bound to a
# function), the editor provides the values. But if we want the client to defer
# to the LSP server's config, use this sentinel. See usage example in `docs/`.
FROM_CONFIG = 'FROM_CONFIG'
FROM_CONFIG_CHAT = 'FROM_CONFIG_CHAT'
FROM_CONFIG_COMPLETION = 'FROM_CONFIG_COMPLETION'


##################################################
# Local LLM

local_llm_start_tag = ':START_LOCAL:'
local_llm_end_tag = ':END_LOCAL:'
local_llm_name = 'local_llm'


class LLMStreamActor(Actor):
    def __init__(self):
        log_local_llm.debug('ACTOR INIT')
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.current_future = None
        self.should_stop = Event()
        self.tags = [local_llm_start_tag, local_llm_end_tag]

    def receiveMessage(self, msg, sender):
        if isinstance(msg, dict):
            command = msg.get('command')
            doc = msg.get('doc')

            # check if block already exists
            start_ixs, end_ixs = find_block(local_llm_start_tag,
                                            local_llm_end_tag,
                                            doc)

            edits = msg.get('edits')
            log_local_llm.debug(
                f'%%%%%%%%%%'
                f'ACTOR RECV: {msg["command"]}'
                f'ACTOR STATE:'
                f'found block: {start_ixs}  |  {end_ixs}'
                f'is_running: {self.is_running}'
                f'locked: {self.should_stop.is_set()}'
                f'future: {self.current_future}'
                f''
                f'EDITS STATE:'
                f'job_thread alive: {edits.job_thread.is_alive()}'
                f'%%%%%%%%%%'
            )
            if command == 'start':
                uri = msg.get('uri')
                range = msg.get('range')
                prompt = msg.get('prompt')
                edits = msg.get('edits')

                if not (start_ixs and end_ixs):
                    actor_init(local_llm_name, self.tags, uri, range, edits)

                self.start(uri, range, prompt, edits)

            elif command == 'stop':
                uri = msg.get('uri')
                edits = msg.get('edits')
                self.stop()

    def start(self, uri, range, prompt, edits):
        if self.is_running:
            log_local_llm.info('WARN: ON_START_BUT_RUNNING')
            return
        log_local_llm.debug('ACTOR START')

        self.is_running = True
        self.should_stop.clear()

        def f(uri_, prompt_, should_stop_, edits_):
            ''' Compose the streaming fn with some cleanup. '''
            local_llm_stream_fn(uri_, prompt_, should_stop_, edits_)

            # Cleanup
            log_local_llm.debug('CLEANING UP')
            actor_cleanup(local_llm_name, self.tags, uri_, edits_)
            self.is_running = False
            self.current_future = None
            self.should_stop.clear()

        self.current_future = self.executor.submit(
            f, uri, prompt, self.should_stop, edits
        )
        log_local_llm.debug('START CAN RETURN')

    def stop(self):
        log_local_llm.debug('ACTOR STOP')
        if not self.is_running:
            log_local_llm.info('WARN: ON_STOP_BUT_STOPPED')

        self.should_stop.set()

        if self.current_future:
            self.current_future.result()  # block, wait to finish
            self.current_future = None
        log_local_llm.debug('FINALLY STOPPED')


##################################################
# OpenAI

openai_start_tag = ':START_OPENAI:'
openai_end_tag = ':END_OPENAI:'
openai_name = 'openai'


class OpenAIStreamActor(Actor):
    def __init__(self):
        log_openai.debug('ACTOR INIT')
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.current_future = None
        self.should_stop = Event()
        self.tags = [openai_start_tag, openai_end_tag]

    def receiveMessage(self, msg, sender):
        if isinstance(msg, dict):
            command = msg.get('command')
            doc = msg.get('doc')

            # check if block already exists
            start_ixs, end_ixs = find_block(openai_start_tag,
                                            openai_end_tag,
                                            doc)

            edits = msg.get('edits')
            log_openai.debug(
                f'%%%%%%%%%%'
                f'ACTOR RECV: {msg["command"]}'
                f'ACTOR STATE:'
                f'found block: {start_ixs}  |  {end_ixs}'
                f'is_running: {self.is_running}'
                f'locked: {self.should_stop.is_set()}'
                f'future: {self.current_future}'
                f''
                f'EDITS STATE:'
                f'job_thread alive: {edits.job_thread.is_alive()}'
                f'%%%%%%%%%%'
            )
            if command == 'start':
                uri = msg.get('uri')
                range = msg.get('range')
                prompt = msg.get('prompt')
                engine = msg.get('engine')
                max_length = msg.get('max_length')
                edits = msg.get('edits')

                if not (start_ixs and end_ixs):
                    actor_init(openai_name, self.tags, uri, range, edits)

                self.start(uri, range, prompt, engine, max_length, edits)

            elif command == 'stop':
                uri = msg.get('uri')
                edits = msg.get('edits')
                self.stop()

    def start(self, uri, range, prompt, engine, max_length, edits):
        if self.is_running:
            log_openai.info('WARN: ON_START_BUT_RUNNING')
            return
        log_openai.debug('ACTOR START')

        self.is_running = True
        self.should_stop.clear()

        def f(uri_, prompt_, engine_, max_length_, should_stop_, edits_):
            ''' Compose the streaming fn with some cleanup. '''
            openai_stream_fn(uri_, prompt_, engine_, max_length_,
                             should_stop_, edits_)

            # Cleanup
            log_openai.debug('CLEANING UP')
            actor_cleanup(openai_name, self.tags, uri_, edits_)
            self.is_running = False
            self.current_future = None
            self.should_stop.clear()

        self.current_future = self.executor.submit(
            f, uri, prompt, engine, max_length, self.should_stop, edits
        )
        log_openai.debug('START CAN RETURN')

    def stop(self):
        log_openai.debug('ACTOR STOP')
        if not self.is_running:
            log_openai.info('WARN: ON_STOP_BUT_STOPPED')

        self.should_stop.set()

        if self.current_future:
            self.current_future.result()  # block, wait to finish
            self.current_future = None
        log_openai.debug('FINALLY STOPPED')



##################################################
# Server

class Server(LanguageServer):
    def __init__(self, name, version):
        super().__init__(name, version)
        self.stop_stream = Event()
        self.stop_transcription = Event()
        self.is_transcription_running = Event()

        # A ThreadPool for tasks. By throwing long running tasks on here, LSP
        # endpoints can return immediately.
        self.executor = ThreadPoolExecutor(max_workers=3)

        # For converting jsonified things into proper instances. This is
        # necessary when a CodeAction summons a Command, and passes raw JSON to
        # it.
        self.converter = default_converter()

        # Edit-jobs
        edit_job_delay = 0.2
        self.edits = Edits(lambda job: edit._attempt_edit_job(self, job),
                           edit_job_delay)
        # self.edits.create_job_queue('transcription')
        self.edits.create_job_queue(local_llm_name)
        self.edits.create_job_queue(openai_name)
        self.edits.start()

        # Actor System
        self.actor_system = ActorSystem()

        # Local LLM
        self.local_llm_actor = self.actor_system.createActor(
            LLMStreamActor,
            globalName="LLMStreamActor"
        )

        # OpenAI
        self.openai_actor = self.actor_system.createActor(
            OpenAIStreamActor,
            globalName="OpenAIStreamActor"
        )


server = Server("llm-lsp", "0.1.0")


##################################################
# Local LLM

def actor_init(edit_name, tags, uri, range, edits):
    ''' Insert new tags, demarcating block. '''
    tags = '\n'.join(tags)
    job = InsertJob(
        uri=uri,
        text=tags,
        line=range.end.line+1,
        column=0,
        strict=True,
    )
    edits.add_job(edit_name, job)


def actor_cleanup(edit_name, tags, uri, edits):
    ''' Delete tags that demarcated block. '''
    edits.add_job(edit_name, DeleteJob(
        uri=uri,
        regexs=tags,
        strict=True,
    ))


def local_llm_stream_fn(uri, prompt, stop_event, edits):
    log_local_llm.debug('START: LOCAL_LLM_STREAM_FN')
    try:
        request_data = {
            "text": prompt,
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
            raise Exception(
                f"POST request to {LLM_URI} failed with status code "
                f"{response.status_code}"
            )

        # Stream the results to LSP Client
        running_text = ''
        for line in response.iter_lines():
            # For breaking out early
            if stop_event.is_set():
                log_local_llm.debug('STREAM_FN received STOP EVENT')
                return
            response_data = json.loads(line)
            new_text = response_data["generated_text"]
            log_local_llm.debug(f'NEW: {new_text}')
            # ignore empty strings
            if len(new_text) == 0:
                continue

            running_text += new_text
            job = BlockJob(
                uri=uri,
                start_tag=local_llm_start_tag,
                end_tag=local_llm_end_tag,
                text=f'\n{running_text}\n',
                strict=False,
            )
            edits.add_job(local_llm_name, job)

        # Streaming is done, and those added jobs were all non-strict. Let's
        # make sure to have one final strict job. Streaming jobs are ok to be
        # dropped, but we need to make sure it does finalize, eg before a
        # strict delete-tags job is added.
        job = BlockJob(
            uri=uri,
            start_tag=local_llm_start_tag,
            end_tag=local_llm_end_tag,
            text=f'\n{running_text}\n',
            strict=True,
        )
        edits.add_job(local_llm_name, job)

    except Exception as e:
        log_local_llm.error(f'Error: Local LLM, {e}')


@server.thread()
@server.command('command.localLlmStream')
def local_llm_stream(ls: Server, args):
    text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
    range = ls.converter.structure(args[1], Range)
    uri = text_document.uri
    doc = ls.workspace.get_document(uri)
    doc_source = doc.source

    # Extract the highlighted region
    prompt = extract_range(doc_source, range)

    # Send a message to start the stream
    actor_args = {
        'command': 'start',
        'uri': uri,
        'range': range,
        'prompt': prompt,
        'edits': ls.edits,
        'doc': doc_source,
    }
    ls.actor_system.tell(ls.local_llm_actor, actor_args)


@server.thread()
@server.command('command.localLlmStreamStop')
def local_llm_stream_stop(ls: Server, args):
    text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
    uri = text_document.uri
    doc = ls.workspace.get_document(uri)
    doc_source = doc.source

    # Send a message to stop the stream
    actor_args = {
        'command': 'stop',
        'uri': uri,
        'edits': ls.edits,
        'doc': doc_source,
    }
    ls.actor_system.tell(ls.local_llm_actor, actor_args)
    ls.actor_system.tell(ls.openai_actor, actor_args)


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
    ''' Stream responses from OpenAI's API as a generator. '''
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


def openai_stream_fn(uri, prompt, engine, max_length, stop_event, edits):
    log_openai.debug('START: OPENAI_STREAM_FN')
    try:
        # Stream the results to LSP Client
        running_text = ''
        for new_text in openai_autocomplete(engine, prompt, max_length):
            # For breaking out early
            if stop_event.is_set():
                log_openai.debug('STREAM_FN received STOP EVENT')
                return
            log_openai.debug(f'NEW: {new_text}')
            # ignore empty strings
            if len(new_text) == 0:
                continue

            running_text += new_text
            job = BlockJob(
                uri=uri,
                start_tag=openai_start_tag,
                end_tag=openai_end_tag,
                text=f'\n{running_text}\n',
                strict=False,
            )
            edits.add_job(openai_name, job)

        # Streaming is done, and those added jobs were all non-strict. Let's
        # make sure to have one final strict job. Streaming jobs are ok to be
        # dropped, but we need to make sure it does finalize, eg before a
        # strict delete-tags job is added.
        job = BlockJob(
            uri=uri,
            start_tag=openai_start_tag,
            end_tag=openai_end_tag,
            text=f'\n{running_text}\n',
            strict=True,
        )
        edits.add_job(local_llm_name, job)

    except Exception as e:
        log_openai.error(f'Error: Local LLM, {e}')


# def stream(ls, engine, text, max_length):
#     # Initialize Cursor as current end position
#     cur = Position(line=range.end.line + 1, character=range.end.character)

#     response_text = '\n\nOPENAI_RESPONSE:\n'
#     edit = workspace_edit(uri, cur, response_text)
#     params = ApplyWorkspaceEditParams(edit=edit)
#     ls.lsp.send_request("workspace/applyEdit", params)
#     cur = Position(line=cur.line + 3, character=0)

#     # Call the openai_autocomplete function with request and stream
#     # parameters
#     for new_text in openai_autocomplete(engine, text, max_length):
#         edit = workspace_edit(uri, cur, new_text)
#         params = ApplyWorkspaceEditParams(edit=edit)
#         ls.lsp.send_request("workspace/applyEdit", params)

#         # Update the current position, accounting for newlines
#         newlines = new_text.count('\n')
#         last_line = (new_text.rsplit('\n', 1)[-1]
#                      if '\n' in new_text
#                      else new_text)
#         last_line_length = len(last_line)
#         cur = Position(line=cur.line + newlines,
#                        character=(last_line_length
#                                   if newlines
#                                   else cur.character + len(new_text)))

#         # For breaking out early
#         if ls.stop_stream.is_set():
#             break

@server.thread()
@server.command('command.openaiAutocompleteStream')
def openai_autocomplete_stream(ls: Server, args):
    # free the event so this call can run
    ls.stop_stream.clear()

    text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
    range = ls.converter.structure(args[1], Range)

    # Determine engine, by checking for sentinel values to allow LSP client to
    # defer arguments to server's configuration.
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

    text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
    range = ls.converter.structure(args[1], Range)
    uri = text_document.uri
    doc = ls.workspace.get_document(uri)
    doc_source = doc.source

    # Extract the highlighted region
    prompt = extract_range(doc_source, range)

    # Send a message to start the stream
    actor_args = {
        'command': 'start',
        'uri': uri,
        'range': range,
        'prompt': prompt,
        'engine': engine,
        'max_length': max_length,
        'edits': ls.edits,
        'doc': doc_source,
    }
    ls.actor_system.tell(ls.openai_actor, actor_args)

    # # Get the document
    # uri = text_document.uri
    # doc = ls.workspace.get_document(uri).source

    # # Extract the highlighted region
    # input_text = extract_range(doc, range)

    # ls.executor.submit(openai_stream, ls, engine, input_text, max_length)
    return WorkspaceEdit()


@server.command('command.openaiAutocompleteStreamStop')
def openai_autocomplete_stream_stop(ls: Server, *args):
    # ls.stop_stream.set()
    text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
    uri = text_document.uri
    doc = ls.workspace.get_document(uri)
    doc_source = doc.source

    # Send a message to stop the stream
    actor_args = {
        'command': 'stop',
        'uri': uri,
        'edits': ls.edits,
        'doc': doc_source,
    }
    ls.actor_system.tell(ls.openai_actor, actor_args)



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
                # Note: these arguments get jsonified, not passed as python
                #       objs
                arguments=[text_document,
                           range,
                           OPENAI_COMPLETION_ENGINE,
                           OPENAI_MAX_LENGTH]
            )
        ),

        # OpenAI ChatGPT
        CodeAction(
            title='OpenAI ChatGPT',
            kind=CodeActionKind.Refactor,
            command=Command(
                title='Local LLM',
                command='command.openaiAutocompleteStream',
                # Note: these arguments get jsonified, not passed as python
                #       objs
                arguments=[text_document,
                           range,
                           OPENAI_CHAT_ENGINE,
                           OPENAI_MAX_LENGTH]
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
