'''

Jobs for applying textual edits to the LSP client

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


##################################################
# Types


@dataclass
class Job:
    uri: str
    start_tag: str
    end_tag: str
    text: str


JOB_DELAY = 0.2


##################################################
# Util

def workspace_edit(uri: str,
                   version: int,
                   start: Position,
                   end: Position,
                   new_text: str) -> WorkspaceEdit:
    ''' Build a `WorkspaceEdit` for pygls to send to LSP client. '''
    text_edit = TextEdit(range=Range(start=start, end=end), new_text=new_text)
    text_document_edit = {
        'textDocument': VersionedTextDocumentIdentifier(uri=uri,
                                                        version=version),
        'edits': [text_edit],
    }
    return WorkspaceEdit(document_changes=[text_document_edit])


def workspace_edits(uri: str,
                    version: int,
                    start_end_text: List[Tuple[Position, Position, str]]
                    ) -> WorkspaceEdit:
    ''' Build a `WorkspaceEdit` for pygls to send to LSP client. '''
    text_edits = [
        TextEdit(range=Range(start=start, end=end), new_text=new_text)
        for start, end, new_text in start_end_text
    ]
    text_document_edit = {
        'textDocument': VersionedTextDocumentIdentifier(uri=uri,
                                                        version=version),
        'edits': text_edits,
    }
    return WorkspaceEdit(document_changes=[text_document_edit])


def extract_range(doc: str, range: Range) -> str:
    '''Extract the highlighted region of the text coming from the client.'''
    lines = doc.split("\n")
    start = range.start
    end = range.end
    if start.line == end.line:
        return lines[start.line][start.character:end.character]
    else:
        start_line_text = lines[start.line][start.character:]
        middle_lines_text = lines[start.line + 1:end.line]
        end_line_text = lines[end.line][:end.character]
        return "\n".join([start_line_text] +
                         middle_lines_text +
                         [end_line_text])


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
        except Empty:
            return x


def find_pattern_in_document(
        document: str,
        pattern: str) -> List[Tuple[int, int, int]]:
    '''Return (line, start_col, end_col) for each match. Regex cannot span
    newlines.'''
    result = []
    compiled_pattern = re.compile(pattern)

    for line_number, line in enumerate(document.split('\n')):
        for match in compiled_pattern.finditer(line):
            start, end = match.span()
            result.append((line_number, start, end))

    return result


# @@@@@@@@@@
# Tests
document = '''
Hello, world!
Regex is fun.
I like programming in Python.
'''

assert find_pattern_in_document(document, "o") == [(1, 4, 5), (1, 8, 9),
                                                   (3, 9, 10), (3, 26, 27)]
assert find_pattern_in_document(document, "P...on") == [(3, 22, 28)]
assert find_pattern_in_document(document, "Java") == []
assert find_pattern_in_document('', "o") == []
# @@@@@@@@@@


##################################################
#

class Edits:
    '''Edits are saved as jobs to the queue, and applied when document versioning
        aligns properly. Each key is a separate queue. Each queue maintains an
        expectation that only the most recent edit matters, and previous edits
        in the queue have been deprecated, and therefore can be dropped.

    '''

    def __init__(self, ls: LanguageServer):
        # A dict of queues, one for each separate process of edits (eg
        # Transcription, Local LLM, API-based LLM).
        self.edit_jobs = {}

    def create_job_queue(self, name):
        if name not in self.edit_jobs:
            self.edit_jobs[name] = Queue()

    def start(self, ls: LanguageServer):
        self.job_thread = Thread(target=self._process_edit_jobs, daemon=True)
        self.job_thread.start()

    def _process_edit_jobs(self):
        '''A thread that continuously pulls jobs from the queue and attempts to
        execute them.'''
        n_retries = 10
        failed_job = None
        failed_count = 0
        while True:
            for k, q in self.edit_jobs.items():
                job = drain_queue(q)
                # retry failed jobs, if no new tasks exist
                job = job if job else failed_job
                if job is not None and failed_count <= n_retries:
                    print(f'{k}: {job}')
                    success = _attempt_edit_job(self.ls, job)
                    if success:
                        failed_job = None
                        failed_count = 0
                    if not success:
                        failed_job = job
                        failed_count += 1
            time.sleep(JOB_DELAY)


def _attempt_edit_job(ls: LanguageServer, job: Job):
    '''Try to execute a job (apply an edit to the document). May fail if document
    versions don't match.

    '''
    try:
        doc = ls.workspace.get_document(job.uri)
        version = doc.version
        doc_lines = doc.source.split('\n')

        m_start = find_tag(job.start_tag, doc_lines)
        m_end = find_tag(job.end_tag, doc_lines)
        if m_start and m_end:
            ix, s, e = m_start
            start_position = Position(ix, e)
            ix, s, e = m_end
            end_position = Position(ix, s)

            edit = workspace_edit(job.uri,
                                  version,
                                  start_position,
                                  end_position,
                                  job.text)
            params = ApplyWorkspaceEditParams(edit=edit)
            future = ls.lsp.send_request("workspace/applyEdit", params)
            resp = future.result()
            return True

    except pygls.exceptions.JsonRpcException:
        # Most likely a document version mismatch, which is fine. It just
        # means someone edited the document concurrently, and this is set up
        # to try applying the job again.
        return False
