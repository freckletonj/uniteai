'''

Commonly needed functions. TODO: should probably be relocated

'''

from threading import Lock
import re
from typing import List, Tuple

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
    OptionalVersionedTextDocumentIdentifier,
    TextEdit,
    WorkspaceEdit,
    DidChangeTextDocumentParams,
)
import logging


##################################################


def mk_logger(name, level):
    ''' A logger builder helper. This helps out since Thespian is overly
    opinionated about logging.'''
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s => %(message)s [%(pathname)s:%(lineno)d]')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger



##################################################

class ThreadSafeCounter:
    '''
    A threadsafe incrementable integer.
    '''

    def __init__(self):
        self.value = 0
        self._lock = Lock()

    def increment(self):
        with self._lock:
            self.value += 1
            return self.value

    def get(self):
        return self.value


##################################################
# Dict helpers

def get_nested(config, keys):
    ''' Ex: get_nested(my_dict, ['some', 'nested', 'key'] '''
    temp = config
    for k in keys:
        if k in temp:
            temp = temp[k]
        else:
            return None
    return temp


##################################################
# String helpers

def insert_text_at(document, insert_text, line_no, col_no):
    # Split the document into lines
    lines = document.split('\n')

    # Validate line number
    if not (-1 <= line_no <= len(lines)):
        raise ValueError('Line number out of range')
    is_end_line = line_no == len(lines) or line_no == -1
    line = '' if is_end_line else lines[line_no]

    # Validate Column number
    if col_no >= len(line)+1 or col_no < -1:
        raise ValueError('Column number out of range')
    is_end_col = col_no == len(line) or col_no == -1

    if is_end_col:
        updated_line = line + insert_text
    else:
        updated_line = line[:col_no] + insert_text + line[col_no:]

    if is_end_line:
        lines += [updated_line]
    else:
        lines[line_no] = updated_line
    return '\n'.join(lines)


def find_tag(tag: str, doc_lines: [str]):
    ''' Find index of first element that contains `tag`. '''
    ix = 0
    for ix, line in enumerate(doc_lines):
        match = re.search(tag, line)
        if match:
            return ix, match.start(), match.end()
    return None


def find_block(start_tag, end_tag, doc):
    '''Fine the indices of a start/end-tagged block.'''
    doc_lines = doc.split('\n')
    s = find_tag(start_tag, doc_lines)
    e = find_tag(end_tag, doc_lines)
    return s, e


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


##################################################
# LSP-Specific

def workspace_edit(uri: str,
                   version: int,
                   start: Position,
                   end: Position,
                   new_text: str) -> WorkspaceEdit:
    ''' Build a `WorkspaceEdit` for pygls to send to LSP client. '''
    text_edit = TextEdit(range=Range(start=start, end=end), new_text=new_text)
    text_document_edit = {
        'textDocument': OptionalVersionedTextDocumentIdentifier(uri=uri,
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
        'textDocument': OptionalVersionedTextDocumentIdentifier(uri=uri,
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
