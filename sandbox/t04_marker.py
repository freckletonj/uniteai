import threading
import time
import pygls
from pygls.server import LanguageServer
from pygls.workspace import Document
from lsprotocol.types import (
    ApplyWorkspaceEditParams, Position, Range, DidChangeTextDocumentParams,
                          VersionedTextDocumentIdentifier, TextDocumentIdentifier, WorkspaceEdit, TextEdit)
import re
import queue

def workspace_edit(uri: str, version: int, start: Position, end: Position, new_text: str) -> WorkspaceEdit:
    ''' Build a `WorkspaceEdit` for pygls to send to LSP client. '''
    text_edit = TextEdit(range=Range(start=start, end=end), new_text=new_text)
    text_document_edit = {
        'textDocument': VersionedTextDocumentIdentifier(uri=uri, version=version),
        'edits': [text_edit],
    }
    return WorkspaceEdit(document_changes=[text_document_edit])

def find_tag(tag: str, doc_lines: [str]):
    ''' Find index of first element that contains `tag`. '''
    ix = 0
    for ix, line in enumerate(doc_lines):
        match = re.search(tag, line)
        if match:
            return ix, match.start(), match.end()
    return None

class Server(LanguageServer):
    START_TAG = ':START_TAG:'
    END_TAG   = ':END_TAG:'

    def __init__(self, name, version):
        super().__init__(name, version)
        self.start_position = Position(0, 0)
        self.end_position = Position(0, 0)
        self.inserted_count = 0
        self.uri = None
        self.insert_thread = threading.Thread(target=self._auto_insert, daemon=True)
        self.insert_thread.start()

    def _auto_insert(self):
        while True:
            time.sleep(1)

            if not self.uri:
                continue

            doc = self.workspace.get_document(self.uri)
            version = doc.version
            doc_lines = doc.source.split('\n')

            # Find tags
            m_start = find_tag(self.START_TAG, doc_lines)
            m_end = find_tag(self.END_TAG, doc_lines)
            if m_start and m_end:
                ix, s, e = m_start
                self.start_position = Position(ix, e)
                ix, s, e = m_end
                self.end_position   = Position(ix, s)
            else:
                # No tags found, don't insert
                continue

            # Build edit
            self.inserted_count += 1
            edit = workspace_edit(self.uri, version, self.start_position, self.end_position, str(self.inserted_count))
            params = ApplyWorkspaceEditParams(edit=edit)
            future = self.lsp.send_request("workspace/applyEdit", params)
            try:
                resp = future.result()
            except pygls.exceptions.JsonRpcException as e:
                print(f'FAIL: {e}')

server = Server(name='', version='1')

@server.feature('workspace/didChangeConfiguration')
def workspace_did_change_configuration(ls: Server, *args):
    # avoid a warning
    return []

@server.thread()
@server.feature('textDocument/didChange')
def did_change(ls, params: DidChangeTextDocumentParams) -> None:
    ls.uri = params.text_document.uri

server.start_tcp('localhost', 5033)
