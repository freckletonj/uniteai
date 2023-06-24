'''An Assistant is given domain over a block within a document, defined by start
and end tags, and will stream text to the block. It will handle the lifecycle
of creating the tags, and cleaning them up, as well as handling concurrency
issues as the streaming function it's given churns on a separate thread.

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
from llmpal.edits import Job, Edits

class Assistant:
    def __init__(self,
                 streaming_function,
                 edits: Edits,
                 start_tag: str,
                 end_tag: str):
        self.streaming_function = streaming_function
        self.edits = edits
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.stop_event = Event()

    def start(self, text: str):
        # Make sure the stop event is cleared
        self.stop_event.clear()

        # Start streaming in a new thread
        ls.executor.submit(self._stream, ls, text, job_queue, start_tag, end_tag)

    def _stream(self, text: str):
        # Stream the results using the streaming function
        for new_text in self.streaming_function(text):
            # Check for the stop event
            if self.stop_event.is_set():
                break

            # Create a new job and add it to the job queue
            job = Job(uri, start_tag, end_tag, new_text)
            self.edits.add_job(job_queue, job)

    def stop(self, ls: Server, doc_source: str, uri: str, version: int):
        # Set the stop event
        self.stop_event.set()

        # Clean up the block tags
        start_tag = r':START:\d+:'
        end_tag = r':END:\d+:'
        remove_regex(ls, [start_tag, end_tag], doc_source, uri, version)
