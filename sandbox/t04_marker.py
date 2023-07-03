'''

An example of how to deal with where automated endpoints should be inserting their text.

1. Jobs (to edit a doc) are placed on a queue

2. These edits are attempted, to be inserted between the proper tags

3. If the document has been edited concurrently, and the version is out of date to apply this edit, we'll intelligently try again.

'''

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

def drain_queue(q):
    ''' Drain a queue, and return the latest item. '''
    x = None
    while True:
        try:
            x = q.get(False)
        except queue.Empty:
            return x

class Server(LanguageServer):
    START_TAG = ':START_TAG:'
    END_TAG   = ':END_TAG:'

    def __init__(self, name, version):
        super().__init__(name, version)
        self.start_position = Position(0, 0)
        self.end_position = Position(0, 0)
        self.inserted_count = 0
        self.uri = None

        self.jobs = queue.Queue()

        self.insert_thread = threading.Thread(target=self._auto_insert, daemon=True)
        self.insert_thread.start()

        self.process_thread = threading.Thread(target=self._process_jobs, daemon=True)
        self.process_thread.start()

    def _process_jobs(self):
        ''' A thread that continuously pulls jobs from the queue and attempts to execute them. '''
        failed_job = None
        while True:
            print('process')
            job = drain_queue(self.jobs)
            # retry failed jobs, if no new tasks exist
            job = job if job else failed_job
            if job is not None:
                success = self._attempt_job(job)
                if success:
                    failed_job = None
                if not success:
                    failed_job = job
            time.sleep(0.1)

    def _attempt_job(self, job):
        '''Try to execute a job. May fail if document versions don't match.'''
        try:
            doc = self.workspace.get_document(self.uri)
            version = doc.version
            doc_lines = doc.source.split('\n')

            m_start = find_tag(self.START_TAG, doc_lines)
            m_end = find_tag(self.END_TAG, doc_lines)
            if m_start and m_end:
                ix, s, e = m_start
                start_position = Position(ix, e)
                ix, s, e = m_end
                end_position   = Position(ix, s)

                edit = workspace_edit(self.uri, version, start_position, end_position, str(job))
                params = ApplyWorkspaceEditParams(edit=edit)
                future = self.lsp.send_request("workspace/applyEdit", params)
                resp = future.result()
                return True

        except pygls.exceptions.JsonRpcException as e:
            # Most likely a version mismatch, which is fine. It just means
            # someone edited the document concurrently.
            return False

    def _auto_insert(self):
        ''' Generate jobs and place them on the queue. '''
        while True:
            time.sleep(1)
            if not self.uri:
                continue
            self.inserted_count += 1
            self.jobs.put(self.inserted_count)


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
