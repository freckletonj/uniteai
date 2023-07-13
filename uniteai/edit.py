'''

Jobs for applying textual edits to the LSP client

'''

from typing import List, Union
import pygls
from pygls.server import LanguageServer
from pygls.workspace import Workspace
from lsprotocol.types import (
    ApplyWorkspaceEditParams,
    Range,
    Position,
)

from threading import Thread
from queue import Queue, Empty
import time
from dataclasses import dataclass
from uniteai.common import find_tag, workspace_edit, workspace_edits, mk_logger
import logging

log = mk_logger('edit', logging.DEBUG)


##################################################
# Types


@dataclass
class BlockJob:
    uri: str
    start_tag: str
    end_tag: str
    text: str
    # strict jobs MUST be applied, non-strict may be skipped (eg interim state
    # when streaming)
    strict: bool


@dataclass
class DeleteJob:
    uri: str
    regexs: List[str]
    strict: bool


@dataclass
class InsertJob:
    uri: str
    text: str
    line: int
    column: int
    strict: bool


LSPJob = Union[InsertJob, BlockJob, DeleteJob]


##################################################
# Util

def drain_non_strict_queue(q):
    '''Drain a queue up until the first "strict" job or the latest job.'''
    x = None
    while True:
        try:
            x = q.get(False)
            if x.strict:
                return x
            else:
                continue
        except Empty:
            return x


##################################################
#

class Edits:
    '''Edits are saved as jobs to the queue, and applied when document
    versioning aligns properly. Each key is a separate queue. Each queue
    maintains an expectation that only the most recent edit matters, and
    previous edits in the queue have been deprecated, and therefore can be
    dropped.

    '''

    def __init__(self, applicator_fn, job_delay=0.2):
        # A function for applying jobs
        self.applicator_fn = applicator_fn
        self.job_delay = job_delay
        # A dict of queues, one for each separate process of edit blocks.
        self.edit_jobs = {}
        self.job_thread = None

    def create_job_queue(self, name: str):
        ''' Create a job queue. '''
        if name not in self.edit_jobs:
            q = Queue()
            self.edit_jobs[name] = q
            return q
        return self.edit_jobs[name]

    def add_job(self, name: str, job):
        ''' Add a "job", which can be any dataclass and have any data, but must
        also have a field called `strict` which determines if this edit must be
        applied, or can possibly be dropped (eg in the case of streaming data,
        interim state can be dropped.  '''
        log.debug(
            f'ADDING JOB, current queue size: {self.edit_jobs[name].qsize()}'
        )
        self.edit_jobs[name].put(job)

    def start(self):
        self.job_thread = Thread(target=self._process_edit_jobs, daemon=True)
        self.job_thread.start()

    def _process_edit_jobs(self):
        '''A thread that continuously pulls jobs from the queue and attempts to
        execute them.

        Rules:

        * If a non-strict job comes in, try to apply it, until a newer job
          comes in, in which case this job is ok to drop. This is useful during
          streaming.

        * If a strict job comes in, it must be applied before drawing more jobs
          off the queue.

        '''
        n_retries = 5
        n_retries_strict = 5
        failed_job = None
        failed_count = 0
        while True:
            for k, q in self.edit_jobs.items():
                job = None  # reset

                # Strict jobs must apply before continuing to pull off the
                # queue.
                if failed_job and failed_job.strict:
                    if failed_count > n_retries_strict:
                        msg = (
                            f'a strict job has failed to apply after '
                            f'{n_retries_strict} attempts. The job was: '
                            f'{failed_job}')
                        logging.error(msg)
                        failed_job = None
                        failed_count = 0
                    else:
                        job = failed_job

                # If no strict failed_jobs exist, try getting latest off queue,
                # or previous non-strict failed jobs.
                else:
                    # get next strict, or the latest (dropping interim)
                    job = drain_non_strict_queue(q)
                    if job:
                        failed_job = None
                        failed_count = 0

                    # retry failed jobs, if no new tasks exist
                    if not job and failed_count <= n_retries:
                        job = failed_job

                # Execute extant jobs.
                if job is not None:
                    success = self.applicator_fn(job)
                    if success:
                        failed_job = None
                        failed_count = 0
                    if not success:
                        failed_job = job
                        failed_count += 1
            time.sleep(self.job_delay)

def _attempt_edit_job(ls: LanguageServer, job: LSPJob):
    if isinstance(job, InsertJob):
        return _attempt_insert_job(ls, job)
    if isinstance(job, BlockJob):
        return _attempt_block_job(ls, job)
    elif isinstance(job, DeleteJob):
        return _attempt_delete_job(ls, job)


def _attempt_insert_job(ls: LanguageServer, job: InsertJob):
    ''' An `applicator_fn` specialized to pygls Servers.

    Try to execute a job that inserts some text. May fail if document versions
    don't match.

    '''
    log.debug('APPLYING INSERT')
    try:
        doc = ls.workspace.get_document(job.uri)
        if not doc:
            log.error(f'Document not managed by Workspace. Make sure this file type is managed by the client, so it sends `didOpen`. uri={job.uri}')
            return False

        version = doc.version
        log.debug(f'INSERT: uri={job.uri}, doc={doc}, version={version}')
        position = Position(job.line, job.column)
        edit = workspace_edit(job.uri,
                              version,
                              position,
                              position,
                              job.text)
        params = ApplyWorkspaceEditParams(edit=edit)
        future = ls.lsp.send_request("workspace/applyEdit", params)
        future.result()  # blocks
        return True

    except pygls.exceptions.JsonRpcException as e:
        # Most likely a document version mismatch, which is fine. It just
        # means someone edited the document concurrently, and this is set up
        # to try applying the job again.
        log.info(f'ATTEMPT_INSERT_JOB: {e}')
        return False


def _attempt_delete_job(ls: LanguageServer, job: DeleteJob):
    ''' An `applicator_fn` specialized to pygls Servers.

    Try to execute a job that deletes some regex. May fail if document versions
    don't match.

    '''
    try:
        doc = ls.workspace.get_document(job.uri)
        if not doc:
            log.error(f'Document not managed by Workspace. Make sure this file type is managed by the client, so it sends `didOpen`. uri={job.uri}')
            return False
        version = doc.version
        doc_lines = doc.source.split('\n')

        # Collect positions of regexs
        failed_regexs = []
        start_end_texts = []
        for regex in job.regexs:
            m_found = find_tag(regex, doc_lines)
            if m_found:
                ix, s, e = m_found
                start_position = Position(ix, s)
                end_position = Position(ix, e)
                start_end_texts.append((start_position, end_position, ''))
            else:
                failed_regexs.append(regex)

        # log warning if tags missed
        if failed_regexs:
            msg = (f'tags not found in document to apply deletion: '
                   f'{failed_regexs}')
            logging.warn(msg)

        # fire off a delete request for the regexs that were found.
        if start_end_texts:
            edit = workspace_edits(job.uri,
                                   version,
                                   start_end_texts)
            params = ApplyWorkspaceEditParams(edit=edit)
            future = ls.lsp.send_request("workspace/applyEdit", params)
            future.result()  # blocks
            return True

    except pygls.exceptions.JsonRpcException as e:
        # Most likely a document version mismatch, which is fine. It just
        # means someone edited the document concurrently, and this is set up
        # to try applying the job again.
        log.info(f'ATTEMPT_DELETE_JOB: {e}')
        return False


def _attempt_block_job(ls: LanguageServer, job: BlockJob):
    ''' An `applicator_fn` specialized to pygls Servers.

    Try to execute a job (apply an edit to the document) within the confines of
    a "block" demarcated by start and end tags. May fail if document versions
    don't match.

    '''
    log.debug('APPLYING BLOCK')
    try:
        doc = ls.workspace.get_document(job.uri)
        if not doc:
            log.error(f'Document not managed by Workspace. Make sure this file type is managed by the client, so it sends `didOpen`. uri={job.uri}. All docs: {ls.workspace.documents.keys()}')
            return False
        version = doc.version
        doc_lines = doc.source.split('\n')
        log.debug(f'BLOCK CONTEXT: uri={job.uri}, version={version}, doc={type(doc)}|{doc}, doc_source[:100]={doc.source[:100]}')

        m_start = find_tag(job.start_tag, doc_lines)
        m_end = find_tag(job.end_tag, doc_lines)
        if m_start and m_end:
            log.debug('BLOCK: found tags, applying edit')
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
            future.result()  # blocks
            return True
        elif job.strict:  # couldn't find tags
            log.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
                      f'DOCUMENT: {doc.source}\n'
                      '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            msg = (
                f'tags not found in document {job.uri} to apply edit: '
                f'{job.text}. Tags: {job.start_tag} and {job.end_tag}'
            )
            logging.warning(msg)

    except pygls.exceptions.JsonRpcException as e:
        # Most likely a document version mismatch, which is fine. It just
        # means someone edited the document concurrently, and this is set up
        # to try applying the job again.
        log.info(f'ATTEMPT_BLOCK_JOB: {e}')
        return False


##################################################
# Smart constructors

def init_block(edit_name,
               tags,
               uri,
               range_or_pos: Union[Range, Position],
               edits):
    ''' Insert new tags, demarcating a new block at the end of the highlighted
    range. '''
    if isinstance(range_or_pos, Range):
        range = range_or_pos
    elif isinstance(range_or_pos, Position):
        range = Range(start=range_or_pos, end=range_or_pos)

    tags = '\n'.join(tags)
    job = InsertJob(
        uri=uri,
        text=tags,
        line=range.end.line+1,
        column=0,
        strict=True,
    )
    edits.add_job(edit_name, job)


def cleanup_block(edit_name, tags, uri, edits):
    ''' Delete tags that demarcated a block. '''
    edits.add_job(edit_name, DeleteJob(
        uri=uri,
        regexs=tags,
        strict=True,
    ))
