'''

The Pygls LSP Server, and functions for adding features into it.

'''

from typing import List, Callable
from pygls.server import LanguageServer
from lsprotocol.types import (
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Command,
    TextDocumentIdentifier,

    TEXT_DOCUMENT_DID_OPEN,
    DidOpenTextDocumentParams,
    Diagnostic,
    Range,
    Position,
)
from pygls.protocol import default_converter
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from thespian.actors import ActorSystem
import uniteai.edit as edit
import logging
from uniteai.common import mk_logger

NAME = 'server.py'
log = mk_logger(NAME, logging.DEBUG)

##################################################
# Server

class Server(LanguageServer):
    def __init__(self, name, version):
        super().__init__(name, version)
        self.code_actions = []

        # A ThreadPool for tasks. By throwing long running tasks on here, LSP
        # endpoints can return immediately.
        self.executor = ThreadPoolExecutor(max_workers=3)

        # For converting jsonified things into proper instances. This is
        # necessary when a CodeAction summons a Command, and passes raw JSON to
        # it.
        self.converter = default_converter()

        # Edit-jobs
        edit_job_delay = 0.2
        self.edits = edit.Edits(
            lambda job: edit._attempt_edit_job(self, job), edit_job_delay)
        self.edits.start()

        # Actor System
        self.actor_system = ActorSystem()
        self.actors = {}  # each feature (eg Local LLM) gets its own actor)

    def add_actor(self, name, cls):
        ''' Adds an Actor, and an Edits job queue '''
        self.edits.create_job_queue(name)
        actor = self.actor_system.createActor(
            cls,
            globalName=name
        )
        self.actors[name] = actor
        return actor

    def add_code_action(self, f: Callable[[CodeActionParams], CodeAction]):
        self.code_actions.append(f)

    def tell_actor(self, name, msg):
        self.actor_system.tell(self.actors[name], msg)


def initialize():
    '''
    A Barebones pygls LSP Server.
    '''
    server = Server("uniteai", "0.1.0")

    @server.feature('workspace/didChangeConfiguration')
    def workspace_did_change_configuration(ls: Server, args):
        ''' There's a warning without this. '''
        log.debug(f'workspace/didChangeConfiguration: args={args}')
        return []

    @server.feature(TEXT_DOCUMENT_DID_OPEN)
    async def did_open(ls: Server, params: DidOpenTextDocumentParams):
        """Text document did open notification. It appears this does not
        overwrite the default `didOpen` handler in pygls, so we can add extra
        `didOpen` logic here."""
        log.debug(f'Document did open: uri={params.text_document.uri}')

    @server.feature("textDocument/codeAction")
    def code_action(params: CodeActionParams) -> List[CodeAction]:
        ''' Code Actions are eg a list of options you can trigger in your
        client. Included modules can add to `server.code_actions` via
        `server.add_code_action`.  '''
        return [action(params) for action in server.code_actions]

    @server.thread()
    @server.command('command.stop')
    def stop(ls: Server, args):
        '''
        Tell ALL actors to stop.
        '''
        if len(args) != 1:
            log.error(f'command.stop: Wrong arguments, received: {args}')
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
        for actor in ls.actors:
            ls.tell_actor(actor, actor_args)

    # Add `command.stop` as a "Code Action" too (accessible from the dropdown
    # menu, eg `M-'`.
    server.add_code_action(code_action_stop)
    return server


def code_action_stop(params: CodeActionParams):
    text_document = params.text_document
    return CodeAction(
                title='Stop Streaming Things',
                kind=CodeActionKind.Refactor,
                command=Command(
                    title='Stop Streaming Things',
                    command='command.stop',
                    arguments=[text_document]
                )
            )
