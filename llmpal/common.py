'''

Commonly needed functions. TODO: should probably be relocated

'''

from threading import Lock


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
