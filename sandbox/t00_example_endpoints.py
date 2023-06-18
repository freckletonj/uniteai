'''

Here lie simple examples I created, and used for reference. This file can be
thrown away eventually.

'''

@server.feature(
    "textDocument/completion",
    CompletionOptions(trigger_characters=["."]),
)
def completions(params: CompletionParams) -> CompletionList:
    items: List[CompletionItem] = []

    highlighted_text = ""
    if params.context.trigger_kind == CompletionTriggerKind.TriggerCharacter:
        uri = params.text_document.uri
        doc = server.workspace.get_document(uri)
        highlighted_text = doc.source.split("\n")[params.position.line]

    completions = [suffix for suffix in ["_completed", "_done", "_finished"]]
    items.extend(
        CompletionItem(
            label=x,
            kind=2,
            insert_text=x,
        )
        for x in completions
    )

    return CompletionList(is_incomplete=False, items=items)

def hello_world(text_document: TextDocumentIdentifier, range: Range) -> WorkspaceEdit:
    uri = text_document.uri
    start = range.start  # start of the range
    end = Position(line=start.line, character=start.character + len('Hello Woorld!\n'))
    new_text = 'Hello Woorld!\n'
    return workspace_edit(uri, end, new_text)

def copy_and_uppercase(text_document: TextDocumentIdentifier, range: Range, mode: str) -> WorkspaceEdit:
    uri = text_document.uri
    start = range.start  # start of the range
    end = range.end  # end of the range

    doc = server.workspace.get_document(uri)
    lines = doc.source.split("\n")

    if mode == 'lines':
        selected_text = lines[start.line:end.line+1]
    else:  # mode == 'region'
        if start.line == end.line:
            selected_text = [lines[start.line][start.character:end.character]]
        else:
            start_line_text = lines[start.line][start.character:]  # Get text from start position to the end of the line
            end_line_text = lines[end.line][:end.character]  # Get text from the start of the line to end position
            middle_lines_text = lines[start.line + 1:end.line]  # Get all the lines between start line and end line
            selected_text = [start_line_text] + middle_lines_text + [end_line_text]

    selected_text = "\n".join(selected_text).upper()

    # Calculate the last line of the document
    last_line = len(lines)

    # Create a TextEdit for the end of the document
    new_start = Position(line=last_line, character=0)
    new_text = "\n" + selected_text + "\n"

    return workspace_edit(uri, new_start, new_text)


##################################################

# Streaming examples

@server.thread()  # multi-threading unblocks emacs
@server.command('command.startStreaming')
def start_streaming_command(ls: SimpleLanguageServer, args):
    text_document = converter.structure(args[0], TextDocumentIdentifier)
    range = converter.structure(args[1], Range)
    uri = text_document.uri

    def stream_hello():
        i = 0
        while not server._background_task.is_set():
            new_text = f"{i} "
            i += 1
            start = range.start  # start of the range
            end = Position(line=start.line, character=start.character + len(new_text))

            edit = workspace_edit(uri, end, new_text)
            params = ApplyWorkspaceEditParams(edit=edit)
            server.lsp.send_request("workspace/applyEdit", params)
            time.sleep(0.01)

    if server._background_task.is_set():
        server._background_task.clear()

    server._background_task = Event()
    stream_hello()

    return WorkspaceEdit()


@server.command('command.stopStreaming')
def stop_streaming_command(ls: SimpleLanguageServer, *args):
    server._background_task.set()

@server.feature("textDocument/codeAction")
def code_action(params: CodeActionParams) -> List[CodeAction]:
    text_document = params.text_document
    range = params.range
    return [
        CodeAction(
            title='Print Hello World',
            kind=CodeActionKind.Refactor,
            edit=hello_world(text_document, range)
        ),
        CodeAction(
            title='Copy and Uppercase Lines',
            kind=CodeActionKind.Refactor,
            edit=copy_and_uppercase(text_document, range, 'lines'),
        ),
        CodeAction(
            title='Copy and Uppercase Region',
            kind=CodeActionKind.Refactor,
            edit=copy_and_uppercase(text_document, range, 'region'),
        ),

        CodeAction(
            title='Start Streaming',
            kind=CodeActionKind.Refactor,
            command=Command(
                title='Start Streaming',
                command='command.startStreaming',
                # Note: these arguments get jsonified, not passed directly
                arguments=[text_document, range]
            )
        ),
        CodeAction(
            title='Stop Streaming',
            kind=CodeActionKind.Refactor,
            command=Command(
                title='Stop Streaming',
                command='command.stopStreaming'
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

    ]
