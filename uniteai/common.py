'''

Commonly needed functions. TODO: should probably be relocated

'''

import re
from typing import List, Tuple
from lsprotocol.types import (
    Position,
    Range,
    OptionalVersionedTextDocumentIdentifier,
    TextEdit,
    WorkspaceEdit,
)

from pypdf import PdfReader
from bs4 import BeautifulSoup
from pathlib import Path
from io import BytesIO
import nbformat
import logging


##################################################


def mk_logger(name, level):
    ''' A logger builder helper. This helps out since Thespian is overly
    opinionated about logging.'''
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s:%(name)s => %(message)s [%(pathname)s:%(lineno)d]'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


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
# IO Helpers


def convert_ipynb_to_py(ipynb_buf):
    ''' Convert ipynb to human-readable python source file.'''
    nb = nbformat.read(ipynb_buf, as_version=4)

    # Convert cells to Python code with comments
    python_code = ""
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Add the code cell's content to the Python code
            python_code += cell.source.strip() + "\n\n"
        elif cell.cell_type == 'markdown':
            # Add the markdown content as comments
            lines = cell.source.strip().split('\n')
            for line in lines:
                python_code += f"# {line}\n"
            python_code += "\n"
    return python_code


def read_unicode(file_path, buf=None):
    '''Read a file_path in as a unicode string.

    Args:
      file_path: a path to a file to read
      buf: if the file has already been read into an IO buffer, we can use it
    '''
    file_path = Path(file_path)

    if buf is None:
        with open(file_path, 'rb') as fp:
            buf = BytesIO(fp.read())

    file_content = buf.getvalue()
    if file_path.suffix == '.pdf':
        pdf = PdfReader(BytesIO(file_content))
        return '\n'.join(page.extract_text() for page in pdf.pages)
    elif file_path.suffix == '.html':
        soup = BeautifulSoup(file_content, "html.parser")
        return soup.get_text()
    elif file_path.suffix == '.txt' or file_path.suffix == '.json':
        return file_content.decode('utf-8')
    elif file_path.suffix == '.txt' or file_path.suffix == '.ipynb':
        py = convert_ipynb_to_py(buf)
        return py
    else:
        return file_content.decode('utf-8', errors='ignore')


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
        'textDocument': OptionalVersionedTextDocumentIdentifier(
            uri=uri,
            version=version
        ),
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
        'textDocument': OptionalVersionedTextDocumentIdentifier(
            uri=uri,
            version=version
        ),
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
