'''

An Assistant is given domain over a block within a document, defined by start
and end tags, and will stream text to the block. It will handle the lifecycle
of creating the tags, and cleaning them up, as well as handling concurrency
issues as the streaming function it's given churns on a separate thread.

'''

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
from llmpal.edit import Edits
from threading import Event
from concurrent.futures import Future
from typing import Callable, Optional
from llmpal.common import ThreadSafeCounter


class Assistant:
    ''' An `Assistant` controls a block of text, can take stream the output of
    a multithreaded function into that block, and cleans up its
    block-demarcating tags.
    '''

    def __init__(self,
                 name: str,
                 executor,
                 edits: Edits,
                 on_start_but_running: Callable[[], None],
                 on_stop_but_stopped: Callable[[], None],
                 ):

        self.name = name
        self.edits = edits
        self.executor = executor
        self.on_start_but_running = on_start_but_running
        self.on_stop_but_stopped = on_stop_but_stopped

        # Concurrency things
        self.local_counter = ThreadSafeCounter()
        self.cleanup_function = None
        self.current_future: Optional[Future] = None
        self.is_running = Event()
        self.should_stop = Event()

    def start(self,
              init_function: Callable[Edits, None],
              streaming_function: Callable[[Event, Edits], None],
              cleanup_function: Callable[[Edits], None]
              ):
        # Check if it's safe to run
        if self.is_running.is_set():
            return self.on_start_but_running()

        # Start streaming in a new thread
        init_function(self.edits)
        self.cleanup_function = cleanup_function
        self.is_running.set()
        self.current_future = self.executor.submit(
            streaming_function,
            self.should_stop,
            self.edits
        )

    def stop(self):
        # Check if it's safe to stop
        if not self.is_running.is_set():
            return self.on_stop_but_stopped()

        # Set the stop event
        self.should_stop.set()

        if self.current_future:
            self.current_future.result()  # block, wait to finish
            self.current_future = None

        self.is_running.clear()
        self.should_stop.clear()
        self.cleanup_function(self.edits)



def lsp_assistant():
    pass
