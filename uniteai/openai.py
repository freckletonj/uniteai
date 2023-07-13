'''

Connection to OpenAI API.

'''


from lsprotocol.types import (
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Command,
    Range,
    TextDocumentIdentifier,
    WorkspaceEdit,
)
from concurrent.futures import ThreadPoolExecutor
import openai
from threading import Event
from thespian.actors import Actor
import argparse
import logging

from uniteai.edit import init_block, cleanup_block, BlockJob
from uniteai.common import extract_range, find_block, mk_logger, get_nested
from uniteai.server import Server


##################################################
# OpenAI

START_TAG = ':START_OPENAI:'
END_TAG = ':END_OPENAI:'
NAME = 'openai'
log = mk_logger(NAME, logging.WARN)

# A sentinel to be used by LSP clients when calling commands. If a CodeAction
# is called (from a dropdown in the text editor), params are defined in the
# python function here. If a Command is called (IE a key sequence bound to a
# function), the editor provides the values. But if we want the client to defer
# to the LSP server's config, use this sentinel. See usage example in `docs/`.
FROM_CONFIG = 'FROM_CONFIG'
FROM_CONFIG_CHAT = 'FROM_CONFIG_CHAT'
FROM_CONFIG_COMPLETION = 'FROM_CONFIG_COMPLETION'


class OpenAIActor(Actor):
    def __init__(self):
        log.debug('ACTOR INIT')
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.current_future = None
        self.should_stop = Event()
        self.tags = [START_TAG, END_TAG]

    def receiveMessage(self, msg, sender):
        command = msg.get('command')
        doc = msg.get('doc')
        edits = msg.get('edits')
        log.debug(f'''
%%%%%%%%%%
ACTOR RECV: {msg["command"]}
ACTOR STATE:
is_running: {self.is_running}
should_stop: {self.should_stop.is_set()}
current_future: {self.current_future}

EDITS STATE:
job_thread alive: {edits.job_thread.is_alive() if edits and edits.job_thread else "NOT STARTED"}
%%%%%%%%%%''')
        if command == 'start':
            uri = msg.get('uri')
            range = msg.get('range')
            prompt = msg.get('prompt')
            engine = msg.get('engine')
            max_length = msg.get('max_length')
            edits = msg.get('edits')

            # check if block already exists
            start_ixs, end_ixs = find_block(START_TAG,
                                            END_TAG,
                                            doc)

            if not (start_ixs and end_ixs):
                init_block(NAME, self.tags, uri, range, edits)

            self.start(uri, range, prompt, engine, max_length, edits)

        elif command == 'stop':
            self.stop()

    def start(self, uri, range, prompt, engine, max_length, edits):
        if self.is_running:
            log.info('WARN: ON_START_BUT_RUNNING')
            return
        log.debug('ACTOR START')

        self.is_running = True
        self.should_stop.clear()

        def f(uri_, prompt_, engine_, max_length_, should_stop_, edits_):
            ''' Compose the streaming fn with some cleanup. '''
            openai_stream_fn(uri_, prompt_, engine_, max_length_,
                             should_stop_, edits_)

            # Cleanup
            log.debug('CLEANING UP')
            cleanup_block(NAME, self.tags, uri_, edits_)
            self.is_running = False
            self.current_future = None
            self.should_stop.clear()

        self.current_future = self.executor.submit(
            f, uri, prompt, engine, max_length, self.should_stop, edits
        )
        log.debug('START CAN RETURN')

    def stop(self):
        log.debug('ACTOR STOP')
        if not self.is_running:
            log.info('WARN: ON_STOP_BUT_STOPPED')

        self.should_stop.set()

        if self.current_future:
            self.current_future.result()  # block, wait to finish
            self.current_future = None
        log.debug('FINALLY STOPPED')


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
    log.debug(f'START: OPENAI_STREAM_FN, max_length={max_length}')
    try:
        # Stream the results to LSP Client
        running_text = ''
        for new_text in openai_autocomplete(engine, prompt, max_length):
            # For breaking out early
            if stop_event.is_set():
                log.debug('STREAM_FN received STOP EVENT')
                break
            log.debug(f'NEW: {new_text}')
            # ignore empty strings
            if len(new_text) == 0:
                continue

            running_text += new_text
            job = BlockJob(
                uri=uri,
                start_tag=START_TAG,
                end_tag=END_TAG,
                text=f'\n{running_text}\n',
                strict=False,
            )
            edits.add_job(NAME, job)

        # Streaming is done, and those added jobs were all non-strict. Let's
        # make sure to have one final strict job. Streaming jobs are ok to be
        # dropped, but we need to make sure it does finalize, eg before a
        # strict delete-tags job is added.
        job = BlockJob(
            uri=uri,
            start_tag=START_TAG,
            end_tag=END_TAG,
            text=f'\n{running_text}\n',
            strict=True,
        )
        edits.add_job(NAME, job)
        log.debug('STREAM COMPLETE')
    except Exception as e:
        log.error(f'Error: OpenAI, {e}')


def code_action_gpt(engine, max_length, params: CodeActionParams):
    '''Trigger a GPT Autocompletion response. A code action calls a command,
    which is set up below to `tell` the actor to start streaming a response.'''
    text_document = params.text_document
    range = params.range
    return CodeAction(
        title='OpenAI GPT',
        kind=CodeActionKind.Refactor,
        command=Command(
            title='OpenAI GPT',
            command='command.openaiAutocompleteStream',
            # Note: these arguments get jsonified, not passed as python objs
            arguments=[text_document, range, engine, max_length]
        )
    )


def code_action_chat_gpt(engine, max_length, params: CodeActionParams):
    '''Trigger a ChatGPT response. A code action calls a command, which is set
    up below to `tell` the actor to start streaming a response. '''
    text_document = params.text_document
    range = params.range
    return CodeAction(
        title='OpenAI ChatGPT',
        kind=CodeActionKind.Refactor,
        command=Command(
            title='OpenAI ChatGPT',
            command='command.openaiAutocompleteStream',
            # Note: these arguments get jsonified, not passed as python objs
            arguments=[text_document, range, engine, max_length]
        )
    )


##################################################
# Setup

def configure(config_yaml):
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_completion_engine', default=get_nested(config_yaml, ['openai', 'completion_engine']))
    parser.add_argument('--openai_chat_engine', default=get_nested(config_yaml, ['openai', 'chat_engine']))
    parser.add_argument('--openai_max_length', default=get_nested(config_yaml, ['openai', 'max_length']))
    parser.add_argument('--openai_api_key', default=get_nested(config_yaml, ['openai', 'api_key']))

    # bc this is only concerned with openai params, do not error if extra params
    # are sent via cli.
    args, _ = parser.parse_known_args()
    return args




def initialize(config, server):
    # Config
    openai_chat_engine = config.openai_chat_engine
    openai_completion_engine = config.openai_completion_engine
    openai_max_length = config.openai_max_length
    openai.api_key = config.openai_api_key  # make library aware of api key

    # Actor
    server.add_actor(NAME, OpenAIActor)

    # CodeActions
    server.add_code_action(
        lambda params:
        code_action_gpt(openai_completion_engine, openai_max_length, params))
    server.add_code_action(
        lambda params:
        code_action_chat_gpt(openai_chat_engine, openai_max_length, params))

    # Modify Server
    @server.thread()
    @server.command('command.openaiAutocompleteStream')
    def openai_autocomplete_stream(ls: Server, args):
        if len(args) != 4:
            log.error(f'command.openaiAutocompleteStream: Wrong arguments, received: {args}')
        text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
        range = ls.converter.structure(args[1], Range)
        uri = text_document.uri
        doc = ls.workspace.get_document(uri)
        doc_source = doc.source

        # Determine engine, by checking for sentinel values to allow LSP client
        # to defer arguments to server's configuration.
        if args[2] == FROM_CONFIG_CHAT:
            engine = openai_chat_engine
        elif args[2] == FROM_CONFIG_COMPLETION:
            engine = openai_completion_engine
        else:
            engine = args[2]

        # Max Length
        if args[3] == FROM_CONFIG:
            max_length = openai_max_length
        else:
            max_length = args[3]

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
        ls.tell_actor(NAME, actor_args)

        # Return null-edit immediately (the rest will stream)
        return WorkspaceEdit()
