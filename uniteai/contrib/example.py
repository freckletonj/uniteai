'''

Start building your feature off this example!

This example inserts auto-incrementing numbers.

'''

import asyncio
import lsprotocol.types as lsp
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from thespian.actors import Actor
import argparse
import logging
import time

from uniteai.edit import init_block, cleanup_block, BlockJob
from uniteai.common import find_block, mk_logger, get_nested

START_TAG = ':START_EXAMPLE:'
END_TAG = ':END_EXAMPLE:'
NAME = 'example'

# A custom logger for just this feature. You can tune the log level to turn
# on/off just this feature's logs.
log = mk_logger(NAME, logging.DEBUG)


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


def code_action_example(params: lsp.CodeActionParams):
    ''' A code_action triggers a command, which sends a message to the Actor,
    to handle it. '''
    text_document = params.text_document
    # position of the highlighted region in the client's editor
    range = params.range  # lsp spec only provides `Range`
    cursor_pos = range.end
    return lsp.CodeAction(
        title='Example Counter',
        kind=lsp.CodeActionKind.Refactor,
        command=lsp.Command(
            title='Example Counter',
            command='command.exampleCounter',
            # Note: these arguments get jsonified, not passed as python objs
            arguments=[text_document, cursor_pos]
        )
    )


##################################################
# Setup
#
# NOTE: In `.uniteai.yml`, just add `uniteai.example` under `modules`, and this
#       example feature will automatically get built into the server at
#       runtime.
#

def create_action(title, cmd):
    ''' A Helper for adding Commands as CodeActions. '''
    def code_action_fn(params: lsp.CodeActionParams):
        text_document = params.text_document
        range = params.range
        return lsp.CodeAction(
                    title=title,
                    kind=lsp.CodeActionKind.Refactor,
                    command=lsp.Command(
                        title=title,
                        command=cmd,
                        arguments=[text_document, range]
                    )
                )
    return code_action_fn


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

    # CodeActions
    server.add_code_action(code_action_example)

    # Modify Server
    @server.thread()
    @server.command('command.exampleCounter')
    def example_counter(ls, args):
        if len(args) != 2:
            log.error(f'command.exampleCounter: Wrong arguments, received: {args}')
        text_document = ls.converter.structure(args[0], lsp.TextDocumentIdentifier)
        cursor_pos = ls.converter.structure(args[1], lsp.Position)

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
        return lsp.WorkspaceEdit()

    ##############################
    # Example LSP Features

    @server.feature(lsp.TEXT_DOCUMENT_COMPLETION, lsp.CompletionOptions(trigger_characters=['.']))
    async def provide_completions(params: lsp.CompletionParams):
        '''
        NOTES:
        * These are not only triggered on the `trigger_character`. It's just that the `trigger_character` also works.
        '''
        log.info('COMLETIONS STARTED')
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
            lsp.CompletionItem(
                label='my_variable_1',
                kind=lsp.CompletionItemKind.Variable,
                detail='A sample variable',
                sort_text='aaa',  # Controlling order
                commit_characters=["."]
            ),
            lsp.CompletionItem(
                label='my_variable_2',
                kind=lsp.CompletionItemKind.Variable,
                detail='A sample variable',
                sort_text='aaa',  # Controlling order
                commit_characters=["."]
            ),
            lsp.CompletionItem(
                label='a whole sentence can be completed on, but the inserted text changed',
                kind=lsp.CompletionItemKind.Variable,
                detail='A long natural language example',
                insert_text='hah tricked you!',
                sort_text='aab'
            ),
            lsp.CompletionItem(
                label='my_function',
                kind=lsp.CompletionItemKind.Function,
                detail='A sample function',
                insert_text='my_function(',  # Auto-insert parentheses
                sort_text='aab'
            ),
            lsp.CompletionItem(
                label='''
def multiline(x):
    a = 123
    b = 456
    c = 789
    return a + b + c
'''.strip(),
                insert_text_format=lsp.InsertTextFormat.PlainText,

                # Yassnippet: a tabbable completion
                # insert_text_format=InsertTextFormat.Snippet,
            ),
            lsp.CompletionItem(
                label='my_class',
                kind=lsp.CompletionItemKind.Class,
                detail='A sample class',
                insert_text='MyClass',

                # This will also add an import statement (or arbitrary edit)
                additional_text_edits=[
                    lsp.TextEdit(
                        lsp.Range(lsp.Position(0, 0), lsp.Position(0, 0)),
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
        # possible matches *will* match, but probably shouldn't.
        filtered_completions = [item for item in possible_completions if item.label.startswith(starting_text)]

        return lsp.CompletionList(
            # `is_incomplete` means: "I could have given you more completions,
            #     but didn't want to (IE too resource intensive or too many
            #     completions possible. Keep typing to narrow the list."
            is_incomplete=False,
            items=filtered_completions
        )

    async def triggered_completions():
        return lsp.CompletionList(
            is_incomplete=True,
            items=[
                lsp.CompletionItem(
                    label="main",
                    filter_text="mn",  # demonstrate filter_text
                    sort_text='aaa',
                    documentation=None,
                ),
                lsp.CompletionItem(
                    "foo",
                    sort_text='aab',
                    documentation=None,
                ),
                # additional items will be fetched in the next completion call...
            ]
        )

    # Signature Help
    @server.feature(lsp.TEXT_DOCUMENT_SIGNATURE_HELP)
    def provide_signature_help(ls, params):
        ''' The signature help shows up, I think, just in the minibuffer in emacs. '''
        signature = lsp.SignatureInformation(
            label='myFunctionWithSig(arg1: string, arg2: int)',
            documentation='A sample function',
            parameters=[
                lsp.ParameterInformation(label='arg1', documentation='A string argument'),
                lsp.ParameterInformation(label='arg2', documentation='An integer argument'),
            ]
        )
        return lsp.SignatureHelp(signatures=[signature])

    # Goto Definition
    @server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
    def goto_definition(ls, params):
        return [lsp.Location(
            uri='file:///path/to/definition/file.py',
            range=lsp.Range(lsp.Position(line=10, character=5),
                            lsp.Position(line=10, character=15)))
                ]

    # Find References
    @server.feature(lsp.TEXT_DOCUMENT_REFERENCES)
    def find_references(ls, params):
        ''' Perhaps you could use this to get cos-sim of embeddings with a doc,
        and allow you to jump to locations in that doc. '''
        return [
            lsp.Location(uri='file:///path/to/reference1/file.py',
                         range=lsp.Range(
                             lsp.Position(line=5, character=5),
                             lsp.Position(line=5, character=15))),
            lsp.Location(uri='file:///path/to/reference2/file.py',
                         range=lsp.Range(
                             lsp.Position(line=8, character=7),
                             lsp.Position(line=8, character=17)))]

    # Document Highlights
    @server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_HIGHLIGHT)
    def provide_highlights(ls, params):
        ''' NOTE: You must trigger your client to show highlights '''
        return [
            lsp.DocumentHighlight(
                range=lsp.Range(lsp.Position(line=2, character=0), lsp.Position(line=6, character=0)),
                kind=lsp.DocumentHighlightKind.Text
            )
        ]

    # Code Lens
    @server.feature(lsp.TEXT_DOCUMENT_CODE_LENS)
    def provide_code_lens(ls, params):
        ''' Shows an inline hyperlink that if clicked will trigger a command '''
        return [
            lsp.CodeLens(
                range=lsp.Range(lsp.Position(line=10, character=5), lsp.Position(line=10, character=15)),
                command=lsp.Command(title='My Code Lens', command='logSomething', arguments=['an argument', 'wow']),
            )
        ]

    # Document Symbols
    @server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
    def provide_symbols(ls, params):
        ''' Symbols are like function or variable names in a document. '''
        return [
            lsp.DocumentSymbol(
                name='mySymbol1',
                kind=lsp.SymbolKind.Function,
                range=lsp.Range(lsp.Position(line=10, character=5), lsp.Position(line=20, character=5)),
                selection_range=lsp.Range(lsp.Position(line=10, character=5), lsp.Position(line=10, character=15)),
            ),
            lsp.DocumentSymbol(
                name='mySymbol2',
                kind=lsp.SymbolKind.Function,
                range=lsp.Range(lsp.Position(line=21, character=5), lsp.Position(line=22, character=5)),
                selection_range=lsp.Range(lsp.Position(line=21, character=5), lsp.Position(line=22, character=15)),
            )
        ]

    # Hover
    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    def hover(ls, params: lsp.TextDocumentPositionParams):
        """Return a hover response."""
        # You'll typically inspect the params to determine the symbol that the
        # user is hovering over and fetch appropriate documentation or other
        # relevant details.
        message = lsp.MarkupContent(
            kind=lsp.MarkupKind.Markdown, # alt: MarkupKind.PlainText
            value="**This is hover documentation!**\n\nExample content for the hover feature."
        )
        range = lsp.Range(
            start=lsp.Position(line=params.position.line, character=params.position.character),
            end=lsp.Position(line=params.position.line, character=params.position.character + 1)
        )
        return lsp.Hover(contents=message, range=range)


    ##############################
    # Demo LSP Commands

    async def clear_diagnostic_after_delay(uri, delay):
        await asyncio.sleep(delay)
        server.publish_diagnostics(uri, [])

    @server.command("showDiagnostic")
    async def show_diagnostic(ls, args):
        ''' Make a diagnostic of the highlighted region, and then clear it after
        3 seconds. '''
        text_doc = ls.converter.structure(args[0], lsp.TextDocumentIdentifier)
        range = ls.converter.structure(args[1], lsp.Range)

        uri = text_doc.uri
        message = "Sample diagnostic message, will disappear after a few seconds."
        diagnostic = lsp.Diagnostic(
            range=range,
            message=message,
            severity=lsp.DiagnosticSeverity.Warning
        )
        server.publish_diagnostics(uri, [diagnostic])
        asyncio.create_task(clear_diagnostic_after_delay(uri, delay=3))
        return []

    @server.command("showPopup")
    def show_popup(ls, text_doc):
        ''' This shows up in the minibuffer in emacs. '''
        ls.show_message("This is a popup message", lsp.MessageType.Info)

    @server.command("showMsgRequest")
    async def show_msg_request(ls, text_doc):
        params = lsp.ShowMessageRequestParams(
            type=lsp.MessageType.Info,
            message="Is UniteAI the bomb dot com?",
            actions=[
                lsp.MessageActionItem(title="Yes"),
                lsp.MessageActionItem(title="Of course")
            ]
        )
        response = ls.lsp.send_request(lsp.WINDOW_SHOW_MESSAGE_REQUEST, params)
        response = await asyncio.wrap_future(response)
        if response:
            ls.show_message(f"You chose: {response.title}", lsp.MessageType.Info)

    async def send_report(ls):
        # Step 1: Create progress
        token = "sampleToken"
        create_params = lsp.WorkDoneProgressCreateParams(token=token)
        response_future = ls.lsp.send_request(lsp.WINDOW_WORK_DONE_PROGRESS_CREATE, create_params)
        await asyncio.wrap_future(response_future)

        # Step 2: Begin Progress
        progress_begin_params = lsp.ProgressParams(
            token=token,
            value=lsp.WorkDoneProgressBegin(
                title="Analyzing begins",
                cancellable=True,
                message="Starting Message"
            )
        )
        ls.send_notification("$/progress", progress_begin_params)

        # Simulated task with reports
        for percentage in range(0, 101, 10):
            await asyncio.sleep(0.5)  # Simulate some work
            progress_report_params = lsp.ProgressParams(
                token=token,
                value=lsp.WorkDoneProgressReport(
                    percentage=percentage,
                    message=f'Task {percentage}% complete'
                )
            )
            ls.send_notification("$/progress", progress_report_params)

        # Step 4: End Progress
        progress_end_params = lsp.ProgressParams(
            token=token,
            value=lsp.WorkDoneProgressEnd(
                message="Analysis complete"
            )
        )
        ls.send_notification("$/progress", progress_end_params)

    @server.command("reportProgress")
    async def report_progress(ls, args):
        asyncio.create_task(send_report(ls))
        return []

    @server.command("logSomething")
    async def log_something(ls, args):
        '''This is to help debug features that trigger commands, eg code lenses. '''
        log.info(f'logSomething log: {str(args)}')
        return []

    # Code Actions
    server.add_code_action(create_action("Example: Show Diagnostic Message", "showDiagnostic"))
    server.add_code_action(create_action("Example: Show Log Message", "showLog"))
    server.add_code_action(create_action("Example: Show Popup Message", "showPopup"))
    server.add_code_action(create_action("Example: Show Message Request", "showMsgRequest"))
    server.add_code_action(create_action("Example: Report Progress", "reportProgress"))

    return server


def post_initialization(config, server):
    # Actor
    server.add_actor(NAME, ExampleActor)

    # Initialize configuration in Actor
    server.tell_actor(NAME, {
        'command': 'set_config',
        'config': config,
    })
