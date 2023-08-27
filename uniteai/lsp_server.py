'''

An LSP server that connects to the LLM server and APIs for doing the brainy
stuff.

'''

import logging
import uniteai.config as config
from typing import List, Callable
from pygls.server import LanguageServer
from lsprotocol.types import (
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Command,
    TextDocumentIdentifier,
    TEXT_DOCUMENT_DID_OPEN,
    INITIALIZED,
    WINDOW_SHOW_MESSAGE_REQUEST,
    DidOpenTextDocumentParams,
)
from pygls.protocol import default_converter
from concurrent.futures import ThreadPoolExecutor
from thespian.actors import ActorSystem
import uniteai.edit as edit
from uniteai.common import mk_logger
import importlib
import time
import asyncio
from functools import partial
import sys

NAME = 'lsp_server'
log = mk_logger(NAME, logging.INFO)


##################################################
# Actor Concurrency Model
#
# NOTE: If you try to load a CUDA model on a separate process you get this
#       error, and I haven't debugged yet:
#
#       RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use
#       CUDA with multiprocessing, you must use the 'spawn' start method

CONCURRENCY_MODEL = 'simpleSystemBase'  # multi-threading (easier debug)

# CONCURRENCY_MODEL = 'multiprocQueueBase'  # multi-processing (faster)


##################################################
# Logging
#
# Change the basicConfig level to be the generous DEBUG, quiet the libraries,
# and allow custom loggers in modules to have their own debug levels. This
# helps especially with debugging weird concurrency quirks, ie allowing
# different processes to report things as needed, in a way that can be easily
# turned on and off per object of your attention.
#
# NOTE: Thespian has a domineering logging methodology. To customize their
#       formatter, see: `thespian.system.simpleSystemBase`.
#
#       Also, https://github.com/thespianpy/Thespian/issues/73
#
# NOTE: VSCode thinks the server is erroring if it logs to stdout from the main
#       thread, but spawned threads don't seem to have the same effect.

logging.basicConfig(
    filename='log_file.log',  # TODO: feature loggers still seem to report on stdout
    # stream=sys.stdout,
    level=logging.DEBUG,
)

# Quiet the libs a little
logging.getLogger('pygls.feature_manager').setLevel(logging.WARN)
logging.getLogger('pygls.protocol').setLevel(logging.WARN)
logging.getLogger('Thespian').setLevel(logging.WARN)
logging.getLogger('asyncio').setLevel(logging.WARN)


##################################################
# Module Loading
#
# All features on opt-in only, and this loads them up (based on .uniteai.yml
# config)

def load_module(module_name, config_yaml, server):
    ''' Load individual feature modules. This is useful for loading just the
    modules referenced in the config.

    It loads in 3 steps:

    1. collect config

    2. fast initialization (before connecting to client)

    3. post initialization (after connecting to client) '''
    logging.info(f'Loading module: {module_name}')

    # COLLECT CONFIGURATION (eg yml + cli args)
    module = importlib.import_module(module_name)
    if hasattr(module, 'configure'):
        logging.info(f'Configuring module: {module_name}')
        args = module.configure(config_yaml)
    else:
        logging.warn(f'No `configure` fn found for: {module_name}')

    # INITIALIZE. This happens before client connection.
    if hasattr(module, 'initialize'):
        module.initialize(args, server)
        logging.info(f'Initializing module: {module_name}')
    else:
        logging.warn(f'No `initialize` fn found for: {module_name}')

    # POST-INITIALIZATION. These will be called post client conection.
    if hasattr(module, 'post_initialization'):
        server.post_initialization.append(
            partial(module.post_initialization, args)
        )
    else:
        logging.warn(f'No `post_initialization` fn found for: {module_name}')


##################################################
# Server

class Server(LanguageServer):
    def __init__(self, name, version):
        super().__init__(name, version)
        self.code_actions = []

        # Post client-connection. Functions in this list will be called. This is
        # useful for resource-intensive setup that shouldn't block the client
        # booting up.
        self.post_initialization = []

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
        self.actor_system = ActorSystem(CONCURRENCY_MODEL)
        self.actors = {}  # each feature (eg Local LLM) gets its own actor)

    def load_modules(self, args, config_yaml):
        for module_name in args.modules:
            start_time = time.time()  # keep track of initialization time
            load_module(module_name, config_yaml, self)
            end_time = time.time()
            log.info(f'Loading module {module_name} took {end_time - start_time:.2f} seconds.')

    def add_actor(self, name, cls):
        ''' Adds an Actor, and an Edits job queue for that Actor. '''
        self.edits.create_job_queue(name)
        actor = self.actor_system.createActor(
            cls,
            globalName=name
        )
        self.actors[name] = actor
        return actor

    def add_code_action(self, f: Callable[[CodeActionParams], CodeAction]):
        ''' Code Actions are an LSP term, and your client will show you
        available Actions. '''
        self.code_actions.append(f)

    def tell_actor(self, name, msg):
        ''' A non-blocking signal to control an Actor. '''
        self.actor_system.tell(self.actors[name], msg)


def create_action(title, cmd):
    ''' A Helper for adding Commands as CodeActions. '''
    def code_action_fn(params: CodeActionParams):
        text_document = params.text_document
        range = params.range
        return CodeAction(
                    title=title,
                    kind=CodeActionKind.Refactor,
                    command=Command(
                        title=title,
                        command=cmd,
                        arguments=[text_document, range]
                    )
                )
    return code_action_fn


def initialize(args, config_yaml):
    '''
    A Barebones pygls LSP Server.
    '''
    server = Server("uniteai", "0.0.0")

    @server.feature(INITIALIZED)
    def on_initialized(ls, initialized_params):
        log.info('Initialized. Calling `post_initialization` functions.')
        for f in ls.post_initialization:
            f(ls)

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
        log.debug(f'Document opened: uri={params.text_document.uri}')

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
    server.add_code_action(create_action('Stop Streaming Things', 'command.stop'))
    return server


##################################################
# Main

def main():
    # First pass at configuration. Further passes will pick up config
    # per-feature.
    args, config_yaml, parser = config.fetch_config()
    server = initialize(args, config_yaml)

    # Server features must be registered to the server before connecting. IE you
    # cannot add `@server.feature(...)` after the server has been
    # started. Capabilities *can* be dynamically registered, but since clients
    # don't need to implement that we should avoid it by just registering ahead
    # of time. This means that you have to be careful to not do
    # resource-intensive things before `server.start_io()` or the boot time will
    # be slow and clients will timeout.
    server.load_modules(args, config_yaml)

    if args.tcp:
        logging.info(f'Starting LSP on port {args.lsp_port}')
        server.start_tcp(host='localhost', port=args.lsp_port)
    elif args.stdio:
        logging.info('Starting on STDIO')
        server.start_io()

if __name__ == '__main__':
    main()
