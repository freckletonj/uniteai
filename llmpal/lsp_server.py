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
from llmpal.edits import Edits


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


##########
# Logging

logging.basicConfig(
    stream=sys.stdout,
    # level=logging.DEBUG,
    level=logging.INFO,
    # level=logging.WARN,
)


##################################################
# Endpoints


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
        self.edits = Edits(self)
        self.edits.create_job_queue('transcription')
        self.edits.create_job_queue('local')
        self.edits.create_job_queue('api')
        self.edits.start()


server = Server("llm-lsp", "0.1.0")


##################################################
# Local LLM


local_counter = ThreadSafeCounter()

@server.thread()  # multi-threading unblocks editor
@server.command('command.localLlmStream')
def local_llm_stream(ls: Server, args):
    # free the event so this call can run
    ls.stop_stream.clear()

    current_i = local_counter.increment()
    start_tag = f':START_LOCAL:{current_i}:'
    end_tag = f':END_LOCAL:{current_i}:'

    text_document = converter.structure(args[0], TextDocumentIdentifier)
    range = converter.structure(args[1], Range)
    uri = text_document.uri
    doc = ls.workspace.get_document(uri)
    version = doc.version
    doc_source = doc.source

    # Extract the highlighted region
    input_text = extract_range(doc_source, range)

    # Insert new Tags
    cur = Position(line=range.end.line + 1,
                   character=0)
    tags = '\n'.join([
        start_tag,
        end_tag
    ])
    edit = workspace_edit(uri, version, cur, cur, tags)
    params = ApplyWorkspaceEditParams(edit=edit)
    ls.lsp.send_request("workspace/applyEdit", params)

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

            # Stream the results to LSP Client
            running_text = ''
            for line in response.iter_lines():
                # For breaking out early
                if ls.stop_stream.is_set():
                    return
                response_data = json.loads(line)
                new_text = response_data["generated_text"]
                # ignore empty strings
                if len(new_text) == 0:
                    continue

                running_text += new_text
                job = Job(
                    uri,
                    start_tag,
                    end_tag,
                    f'\n{running_text}\n',
                )
                ls.edit_jobs['local'].put(job)


        except Exception as e:
            print(f'EXCEPTION: {e}')

    # stream in a thread so this function can return immediately
    ls.executor.submit(stream, ls)

    # Return an empty edit. The actual edits will be carried out by Commands in
    # the `stream` function.
    return WorkspaceEdit()
#
# @server.thread()
@server.command('command.localLlmStreamStop')
def local_llm_stream_stop(ls: Server, args):
    start_tag = r':START_LOCAL:\d+:'
    end_tag   = r':END_LOCAL:\d+:'

    print('STOP STREAMN')
    ls.stop_stream.set()

    print('POST')
    requests.post(f"{LLM_URI}/local_llm_stream_stop", json='{}')

    # Remove tags
    text_document = converter.structure(args[0], TextDocumentIdentifier)
    uri = text_document.uri
    doc = ls.workspace.get_document(uri)
    version = doc.version
    remove_regex(ls, [start_tag, end_tag], doc.source, uri, version)
    return {'status': 'success'}


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
    warmup_thread.join()
