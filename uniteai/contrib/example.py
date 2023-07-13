'''

Start building your feature off this example!

This example inserts auto-incrementing numbers.

'''


from lsprotocol.types import (
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Command,
    Position,
    TextDocumentIdentifier,
    WorkspaceEdit,
)
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from thespian.actors import Actor
import argparse
import logging
import time

from uniteai.edit import init_block, cleanup_block, BlockJob
from uniteai.common import find_block, mk_logger, get_nested
from uniteai.server import Server


START_TAG = ':START_EXAMPLE:'
END_TAG = ':END_EXAMPLE:'
NAME = 'example'

# A custom logger for just this feature. You can tune the log level to turn
# on/off just this feature's logs.
log = mk_logger(NAME, logging.WARN)


class ExampleActor(Actor):
    def __init__(self):
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
locked: {self.should_stop.is_set()}
future: {self.current_future}

EDITS STATE:
job_thread alive: {edits.job_thread.is_alive() if edits and edits.job_thread else "NOT STARTED"}
%%%%%%%%%%''')

        ##########
        # Start
        if command == 'start':
            uri = msg.get('uri')
            cursor_pos = msg.get('cursor_pos')
            engine = msg.get('engine')
            max_length = msg.get('max_length')
            edits = msg.get('edits')

            # check if block already exists
            start_ixs, end_ixs = find_block(START_TAG,
                                            END_TAG,
                                            doc)

            if not (start_ixs and end_ixs):
                init_block(NAME, self.tags, uri, cursor_pos, edits)

            self.start(uri, cursor_pos, engine, max_length, edits)

        ##########
        # Stop
        elif command == 'stop':
            self.stop()

        ##########
        # Set Config
        elif command == 'set_config':
            config = msg['config']
            self.start_digit = config.example_start_digit
            self.end_digit = config.example_end_digit
            self.delay = config.example_delay

    def start(self, uri, cursor_pos, engine, max_length, edits):
        if self.is_running:
            log.info('WARN: ON_START_BUT_RUNNING')
            return
        log.debug('ACTOR START')

        self.is_running = True
        self.should_stop.clear()
        self.current_future = self.executor.submit(
            self.stream_fn, uri, self.should_stop, edits
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

    def stream_fn(self, uri, stop_event, edits):
        log.debug('START: EXAMPLE_STREAM_FN')
        try:
            # Stream the results to LSP Client
            running_text = ''
            for x in range(self.start_digit, self.end_digit+1):
                # For breaking out early
                if stop_event.is_set():
                    log.debug('STREAM_FN received STOP EVENT')
                    break

                running_text += f'{x} '
                job = BlockJob(
                    uri=uri,
                    start_tag=START_TAG,
                    end_tag=END_TAG,
                    text=f'\n{running_text}\n',
                    # "non-strict" means that if this job doesn't get applied
                    # successfully, it can be dropped. This is useful when
                    # streaming.
                    strict=False,
                )
                edits.add_job(NAME, job)
                time.sleep(self.delay)

            # Streaming is done, and those added jobs were all
            # non-strict. Let's make sure to have one final strict
            # job. Streaming jobs are ok to be dropped, but we need to make
            # sure it does finalize, eg before a strict delete-tags job is
            # added.
            job = BlockJob(
                uri=uri,
                start_tag=START_TAG,
                end_tag=END_TAG,
                text=f'\n{running_text}\n',
                strict=True,
            )
            edits.add_job(NAME, job)

        except Exception as e:
            log.error(f'Error: ExapleActor, {e}')

        # Cleanup
        log.debug('CLEANING UP')
        cleanup_block(NAME, self.tags, uri, edits)
        self.is_running = False
        self.current_future = None
        self.should_stop.clear()


def code_action_example(start_digit: int,
                        end_digit: int,
                        delay: int,
                        params: CodeActionParams):
    ''' A code_action triggers a command, which sends a message to the Actor,
    to handle it. '''
    text_document = params.text_document
    # position of the highlighted region in the client's editor
    range = params.range  # lsp spec only provides `Range`
    cursor_pos = range.end
    return CodeAction(
        title='Example Counter',
        kind=CodeActionKind.Refactor,
        command=Command(
            title='Example Counter',
            command='command.exampleCounter',
            # Note: these arguments get jsonified, not passed as python objs
            arguments=[text_document, cursor_pos, start_digit, end_digit, delay]
        )
    )


##################################################
# Setup
#
# NOTE: In `.uniteai.yml`, just add `uniteai.example` under `modules`, and this
#       will automatically get built into the server at runtime.
#

def configure(config_yaml):
    parser = argparse.ArgumentParser()
    parser.add_argument('--example_start_digit', default=get_nested(config_yaml, ['example', 'start_digit']))
    parser.add_argument('--example_end_digit', default=get_nested(config_yaml, ['example', 'end_digit']))
    parser.add_argument('--example_delay', default=get_nested(config_yaml, ['example', 'delay']))

    # These get picked up as `config` in `initialize`

    # bc this is only concerned with example params, do not error if extra
    # params are sent via cli.
    args, _ = parser.parse_known_args()
    return args


def initialize(config, server):
    # Config
    start_digit = config.example_start_digit
    end_digit = config.example_end_digit
    delay = config.example_delay

    # Actor
    server.add_actor(NAME, ExampleActor)

    # Initialize configuration in Actor
    server.tell_actor(NAME, {
        'command': 'set_config',
        'config': config,
    })

    # CodeActions
    server.add_code_action(
        lambda params:
        code_action_example(start_digit, end_digit, delay, params))

    # Modify Server
    @server.thread()
    @server.command('command.exampleCounter')
    def example_counter(ls: Server, args):
        if len(args) != 2:
            log.error(f'command.exampleCounter: Wrong arguments, received: {args}')
        text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
        cursor_pos = ls.converter.structure(args[1], Position)
        uri = text_document.uri
        doc = ls.workspace.get_document(uri)
        doc_source = doc.source

        # Send a message to start the stream
        actor_args = {
            'command': 'start',
            'uri': uri,
            'cursor_pos': cursor_pos,
            'start_digit': start_digit,
            'end_digit': end_digit,
            'delay': delay,
            'edits': ls.edits,
            'doc': doc_source,
        }
        ls.tell_actor(NAME, actor_args)

        # Return null-edit immediately (the rest will stream)
        return WorkspaceEdit()
