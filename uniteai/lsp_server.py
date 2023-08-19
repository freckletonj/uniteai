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

    WINDOW_WORK_DONE_PROGRESS_CREATE,
    TELEMETRY_EVENT,
    WorkDoneProgressCreateParams,
    WorkDoneProgressBegin,
    WorkDoneProgressEnd,
    ProgressParams,
    MessageType,

    MessageActionItem,
    Diagnostic,
    Position,
    Range,
    DiagnosticSeverity,
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    MessageType,
    ShowMessageRequestParams,
    WorkDoneProgressBegin,
    WorkDoneProgressEnd,
    WorkDoneProgressCreateParams,

    WorkDoneProgressEnd,
    WorkDoneProgressBegin,
    WorkDoneProgressParams,
    WorkDoneProgressReport,
    WorkDoneProgressOptions,
    WorkDoneProgressCancelParams,
    WorkDoneProgressCreateParams,
    WindowWorkDoneProgressCreateRequest,
    WindowWorkDoneProgressCreateResponse,
    WindowWorkDoneProgressCancelNotification,
    TextDocumentPositionParams,

    CompletionItem,
    CompletionItemKind,
    InsertTextFormat,
    Position,
    Range,
    TextEdit,
    CompletionList,
    CompletionContext,
    CompletionTriggerKind,
    CompletionParams,

    Hover,
    CompletionOptions,
    CompletionItem,
    CompletionList,
    CompletionItemKind,
    SignatureHelp,
    SignatureInformation,
    ParameterInformation,
    Location,
    Position,
    DocumentHighlight,
    DocumentHighlightKind,
    CodeLens,
    Command,
    DocumentSymbol,
    SymbolKind,
    MarkupContent,
    MarkupKind,

    COMPLETION_ITEM_RESOLVE,
    TEXT_DOCUMENT_COMPLETION,
    TEXT_DOCUMENT_SIGNATURE_HELP,
    TEXT_DOCUMENT_DEFINITION,
    TEXT_DOCUMENT_REFERENCES,
    TEXT_DOCUMENT_DOCUMENT_HIGHLIGHT,
    TEXT_DOCUMENT_CODE_LENS,
    TEXT_DOCUMENT_DOCUMENT_SYMBOL,
    TEXT_DOCUMENT_HOVER,
)
from pygls.protocol import default_converter
from concurrent.futures import ThreadPoolExecutor
from thespian.actors import ActorSystem
import uniteai.edit as edit
from uniteai.common import mk_logger
import importlib
import time
import asyncio

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
    ''' Useful for loading just the modules referenced in the config. '''
    logging.info(f'Loading module: {module_name}')

    # Collect configuration (eg yml + cli args)
    module = importlib.import_module(module_name)
    if hasattr(module, 'configure'):
        logging.info(f'Configuring module: {module_name}')
        args = module.configure(config_yaml)
    else:
        logging.warn(f'No `configure` fn found for: {module_name}')

    # Initialize
    if hasattr(module, 'initialize'):
        module.initialize(args, server)
        logging.info(f'Initializing module: {module_name}')
    else:
        logging.warn(f'No `initialize` fn found for: {module_name}')


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
    def on_initialized(initialized_params):
        log.info('Initialized. Loading optional modules specified in config.')
        server.load_modules(args, config_yaml)

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


    ##########
    # DEMOS

    async def clear_diagnostic_after_delay(uri, delay=3):
        await asyncio.sleep(delay)
        server.publish_diagnostics(uri, [])

    @server.command("showDiagnostic")
    async def show_diagnostic(ls, args):
        ''' Make a diagnostic of the highlighted region, and then clear it after
        3 seconds. '''
        text_doc = ls.converter.structure(args[0], TextDocumentIdentifier)
        range = ls.converter.structure(args[1], Range)

        uri = text_doc.uri
        message = "Sample diagnostic message"
        diagnostic = Diagnostic(
            range=range,
            message=message,
            severity=DiagnosticSeverity.Warning
        )
        server.publish_diagnostics(uri, [diagnostic])
        asyncio.create_task(clear_diagnostic_after_delay(uri))
        return []

    @server.command("showPopup")
    def show_popup(ls, text_doc):
        ''' This shows up in the minibuffer in emacs. '''
        ls.show_message("This is a popup message", MessageType.Info)

    @server.command("showMsgRequest")
    async def show_msg_request(ls, text_doc):
        params = ShowMessageRequestParams(
            type=MessageType.Info,
            message="Is UniteAI the bomb dot com?",
            actions=[
                MessageActionItem(title="Yes"),
                MessageActionItem(title="Of course")
            ]
        )
        response = ls.lsp.send_request(WINDOW_SHOW_MESSAGE_REQUEST, params)
        response = await asyncio.wrap_future(response)
        if response:
            ls.show_message(f"You chose: {response.title}", MessageType.Info)

    async def send_report(ls):
        # Step 1: Create progress
        token = "sampleToken"
        create_params = WorkDoneProgressCreateParams(token=token)
        response_future = ls.lsp.send_request(WINDOW_WORK_DONE_PROGRESS_CREATE, create_params)
        await asyncio.wrap_future(response_future)

        # Step 2: Begin Progress
        progress_begin_params = ProgressParams(
            token=token,
            value=WorkDoneProgressBegin(
                title="Analyzing begins",
                cancellable=True,
                message="Starting Message"
            )
        )
        ls.send_notification("$/progress", progress_begin_params)

        # Simulated task with reports
        for percentage in range(0, 101, 10):
            await asyncio.sleep(0.5)  # Simulate some work
            progress_report_params = ProgressParams(
                token=token,
                value=WorkDoneProgressReport(
                    percentage=percentage,
                    message=f'Task {percentage}% complete'
                )
            )
            ls.send_notification("$/progress", progress_report_params)

        # Step 4: End Progress
        progress_end_params = ProgressParams(
            token=token,
            value=WorkDoneProgressEnd(
                message="Analysis complete"
            )
        )
        ls.send_notification("$/progress", progress_end_params)

    @server.command("reportProgress")
    async def report_progress(ls, args):
        asyncio.create_task(send_report(ls))
        return []

    @server.command("logSomething")
    async def report_progress(ls, args):
        '''This is to help debug features that trigger commands, eg code lenses. '''
        log.info(f'logSomething log: {str(args)}')
        return []


    ##########


    @server.feature(TEXT_DOCUMENT_COMPLETION, CompletionOptions(trigger_characters=['.']))
    async def provide_completions(params: CompletionParams):
        '''
        NOTES:
        * These are not only triggered on the `trigger_character`. It's just that the `trigger_character` also works.

        '''
        # Extract the line text till the position of the autocomplete trigger
        position = params.position
        uri = params.text_document.uri
        doc = server.workspace.get_document(uri)
        line_text = doc.lines[position.line][:position.character]

        if '.' in line_text:  # If the trigger character was used
            return await triggered_completions()

        return general_completions(line_text)

    def general_completions(starting_text):
        ''' Return general completions based on starting_text. '''
        # Filtering based on existing text
        possible_completions = [
            CompletionItem(
                label='my_variable_1',
                kind=CompletionItemKind.Variable,
                detail='A sample variable',
                sort_text='aaa',  # Controlling order
                commit_characters=["."]
            ),
            CompletionItem(
                label='my_variable_2',
                kind=CompletionItemKind.Variable,
                detail='A sample variable',
                sort_text='aaa',  # Controlling order
                commit_characters=["."]
            ),
            CompletionItem(
                label='a whole sentence can be completed on, but the inserted text changed',
                kind=CompletionItemKind.Variable,
                detail='A long natural language example',
                insert_text='hah tricked you!',
                sort_text='aab'
            ),
            CompletionItem(
                label='my_function',
                kind=CompletionItemKind.Function,
                detail='A sample function',
                insert_text='my_function(',  # Auto-insert parentheses
                sort_text='aab'
            ),
            CompletionItem(
                label='''
def multiline(x):
    a = 123
    b = 456
    c = 789
    return a + b + c
'''.strip(),
                insert_text_format=InsertTextFormat.PlainText,

                # Yassnippet: a tabbable completion
                # insert_text_format=InsertTextFormat.Snippet,
            ),
            CompletionItem(
                label='my_class',
                kind=CompletionItemKind.Class,
                detail='A sample class',
                insert_text='MyClass',

                # This will also add an import statement (or arbitrary edit)
                additional_text_edits=[
                    TextEdit(
                        Range(Position(0, 0), Position(0, 0)),
                        "import MyClass\n"
                    )
                ],
                sort_text='zzz'
            ),
        ]

        # Filtering completions to only include items that start with starting_text.
        #
        # The client does it's own filtering, but it also does a fuzzy match
        # thing that means if you've typed only a few chars, many of the
        # possible matches *will* match, but probably shouldn't
        filtered_completions = [item for item in possible_completions if item.label.startswith(starting_text)]

        return CompletionList(
            # `is_incomplete` means: "I could have given you more completions,
            #     but didn't want to (IE too resource intensive or too many
            #     completions possible. Keep typing to narrow the list."
            is_incomplete=False,
            items=filtered_completions
        )

    async def triggered_completions():
        return CompletionList(
            is_incomplete=True,
            items=[
                CompletionItem(
                    label="main",
                    filter_text="mn",  # demonstrate filter_text
                    sort_text='aaa',
                    documentation=None,
                ),
                CompletionItem(
                    "foo",
                    sort_text='aab',
                    documentation=None,
                ),
                # additional items will be fetched in the next completion call...
            ]
        )

    # Signature Help
    @server.feature(TEXT_DOCUMENT_SIGNATURE_HELP)
    def provide_signature_help(ls, params):
        ''' The signature help shows up, I think, just in the minibuffer in emacs. '''
        signature = SignatureInformation(
            label='myFunctionWithSig(arg1: string, arg2: int)',
            documentation='A sample function',
            parameters=[
                ParameterInformation(label='arg1', documentation='A string argument'),
                ParameterInformation(label='arg2', documentation='An integer argument'),
            ]
        )
        return SignatureHelp(signatures=[signature])

    # Goto Definition
    @server.feature(TEXT_DOCUMENT_DEFINITION)
    def goto_definition(ls, params):
        return [Location(uri='file:///path/to/definition/file.py',
                         range=Range(Position(line=10, character=5), Position(line=10, character=15)))]

    # Find References
    @server.feature(TEXT_DOCUMENT_REFERENCES)
    def find_references(ls, params):
        ''' Perhaps you could use this to get cos-sim of embeddings with a doc,
        and allow you to jump to locations in that doc. '''
        return [
            Location(uri='file:///path/to/reference1/file.py', range=Range(Position(line=5, character=5), Position(line=5, character=15))),
            Location(uri='file:///path/to/reference2/file.py', range=Range(Position(line=8, character=7), Position(line=8, character=17))),
        ]

    # Document Highlights
    @server.feature(TEXT_DOCUMENT_DOCUMENT_HIGHLIGHT)
    def provide_highlights(ls, params):
        ''' NOTE: You must trigger your client to show highlights '''
        return [
            DocumentHighlight(
                range=Range(Position(line=2, character=0), Position(line=6, character=0)),
                kind=DocumentHighlightKind.Text
            )
        ]

    # Code Lens
    @server.feature(TEXT_DOCUMENT_CODE_LENS)
    def provide_code_lens(ls, params):
        ''' Shows an inline hyperlink that if clicked will trigger a command '''
        return [
            CodeLens(
                range=Range(Position(line=10, character=5), Position(line=10, character=15)),
                command=Command(title='My Code Lens', command='logSomething', arguments=['an argument', 'wow']),
            )
        ]

    # Document Symbols
    @server.feature(TEXT_DOCUMENT_DOCUMENT_SYMBOL)
    def provide_symbols(ls, params):
        ''' Symbols are like function or variable names in a document. '''
        return [
            DocumentSymbol(
                name='mySymbol1',
                kind=SymbolKind.Function,
                range=Range(Position(line=10, character=5), Position(line=20, character=5)),
                selection_range=Range(Position(line=10, character=5), Position(line=10, character=15)),
            ),
            DocumentSymbol(
                name='mySymbol2',
                kind=SymbolKind.Function,
                range=Range(Position(line=21, character=5), Position(line=22, character=5)),
                selection_range=Range(Position(line=21, character=5), Position(line=22, character=15)),
            )
        ]

    @server.feature(TEXT_DOCUMENT_HOVER)
    def hover(ls, params: TextDocumentPositionParams):
        """Return a hover response."""
        # You'll typically inspect the params to determine the symbol that the user is hovering over
        # and fetch appropriate documentation or other relevant details.
        message = MarkupContent(
            kind=MarkupKind.Markdown, # alt: MarkupKind.PlainText
            value="**This is hover documentation!**\n\nExample content for the hover feature."
        )
        range = Range(
            start=Position(line=params.position.line, character=params.position.character),
            end=Position(line=params.position.line, character=params.position.character + 1)
        )
        return Hover(contents=message, range=range)

    # Code Actions
    server.add_code_action(create_action("Show Diagnostic Message", "showDiagnostic"))
    server.add_code_action(create_action("Show Log Message", "showLog"))
    server.add_code_action(create_action("Show Popup Message", "showPopup"))
    server.add_code_action(create_action("Show Message Request", "showMsgRequest"))
    server.add_code_action(create_action("Report Progress", "reportProgress"))


    # Add `command.stop` as a "Code Action" too (accessible from the dropdown
    # menu, eg `M-'`.
    server.add_code_action(create_action('Stop Streaming Things', 'command.stop'))
    return server


##################################################
# Main

def main():
    # First pass at configuration. Further passes will pick up config
    # per-feature.
    args, config_yaml, parser = config.get_args()
    server = initialize(args, config_yaml)

    if args.tcp:
        logging.info(f'Starting LSP on port {args.lsp_port}')
        server.start_tcp(host='localhost', port=args.lsp_port)
    elif args.stdio:
        logging.info('Starting on STDIO')
        server.start_io()


if __name__ == '__main__':
    main()
