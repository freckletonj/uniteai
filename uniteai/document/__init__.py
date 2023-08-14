'''

Document Search.

----------

How to format a request to this feature:

    query:
    docs:
      - title:
        url:
      - title:
        url:

'''


from lsprotocol.types import (
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Command,
    Range,
    TextDocumentIdentifier,
    WorkspaceEdit,
)
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from thespian.actors import Actor
import argparse
import logging
import yaml

from uniteai.edit import init_block, cleanup_block, BlockJob
from uniteai.common import extract_range, find_block, mk_logger, get_nested
from uniteai.server import Server
import uniteai.document.embed as embed
import uniteai.document.download as download
import traceback


##################################################
# Document

START_TAG = ':START_DOCUMENT:'
END_TAG = ':END_DOCUMENT:'
NAME = 'document'
log = mk_logger(NAME, logging.DEBUG)


class DocumentActor(Actor):
    def __init__(self):
        log.debug('ACTOR INIT')
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.current_future = None
        self.should_stop = Event()
        self.tags = [START_TAG, END_TAG]

    def receiveMessage(self, msg, sender):
        command = msg.get('command')
        doc = msg.get('doc')
        edits = msg.get('edits')
        log.debug(f'''
%%%%%%%%%%
ACTOR RECV: {msg["command"]}
ACTOR STATE:
is_running: {self.is_running}
should_stop: {self.should_stop.is_set()}
current_future: {self.current_future}

EDITS STATE:
job_thread alive: {edits.job_thread.is_alive() if edits and edits.job_thread else "NOT STARTED"}
%%%%%%%%%%''')

        if command == 'set_config':
            config = msg['config']
            self.download_cache = config.download_cache
            self.embedding_cache = config.embedding_cache
            self.db_path = config.db_path
            self.model_name = config.model_name
            self.denoise_window_size = config.denoise_window_size
            self.denoise_poly_order = config.denoise_poly_order
            self.percentile = config.percentile

        if command == 'initialize':
            # import here to speed up startup
            from sentence_transformers import SentenceTransformer

            # Load ML Model
            self.model = SentenceTransformer(self.model_name)

            # Load helper classes
            download.initialize_database(self.db_path)
            self.dl = download.Downloader(self.download_cache)
            self.embedding = embed.Embedding(self.model, self.embedding_cache)
            self.search = embed.Search(self.model,
                                       self.embedding_cache,
                                       self.embedding,
                                       denoise_window_size=1000,
                                       denoise_poly_order=1,
                                       percentile=95,)

        if command == 'start':
            uri = msg.get('uri')
            range = msg.get('range')
            prompt = msg.get('prompt')
            edits = msg.get('edits')

            # check if block already exists
            start_ixs, end_ixs = find_block(START_TAG,
                                            END_TAG,
                                            doc)

            if not (start_ixs and end_ixs):
                init_block(NAME, self.tags, uri, range, edits)

            self.start(uri, range, prompt, edits)

        elif command == 'stop':
            self.stop()

    def start(self, uri, range, prompt, edits):
        if self.is_running:
            log.info('WARN: ON_START_BUT_RUNNING')
            return
        log.debug('ACTOR START')

        self.is_running = True
        self.should_stop.clear()

        def f(uri_, prompt_, should_stop_, edits_):
            ''' Compose the streaming fn with some cleanup. '''
            log.debug('START: DOCUMENT_STREAM_FN')
            try:
                args = yaml.safe_load(prompt_)

                # Format of documents to load
                #
                #     query:
                #     docs:
                #       - title:
                #         url:
                #       - title:
                #         url:

                query = args['query']
                docs = args['docs']

                # TODO this is ugly
                titles_urls = []
                for d in docs:
                    titles_urls.append((d['title'] if 'title' in d else None,
                                        d['url']))
                download.save_docs(titles_urls, self.dl, self.db_path)

                search_results = embed.search_helper(
                    self.search,
                    self.db_path,
                    titles_urls,
                    query,
                    window_size=2000,
                    stride=50,
                    top_n=3,
                    visualize=False)

                output = output_template(prompt_, search_results)
                # emacs lsp-mode doesn't like unicode NULLs
                output = output.replace('\u0000', '')
                job = BlockJob(
                    uri=uri,
                    start_tag=START_TAG,
                    end_tag=END_TAG,
                    text=f'\n{output}\n',
                    strict=True,
                )
                edits.add_job(NAME, job)

            except Exception as e:
                log.error(f'Error: Document, {e}')
                error_info = traceback.format_exc()
                log.error(error_info)

            # Cleanup
            log.debug('CLEANING UP')
            cleanup_block(NAME, self.tags, uri_, edits_)
            self.is_running = False
            self.current_future = None
            self.should_stop.clear()

        self.current_future = self.executor.submit(
            f, uri, prompt, self.should_stop, edits
        )
        log.debug('START CAN RETURN')

    def stop(self):
        log.debug('ACTOR STOP')
        if not self.is_running:
            log.info('WARN: ON_STOP_BUT_STOPPED')

        self.should_stop.set()

        if self.current_future:
            self.current_future.result()  # block, wait to finish
            self.current_future = None
        log.debug('FINALLY STOPPED')


def output_template(prompt, search_results):
    out = ''
    for url, title, results in search_results:
        for i, result in enumerate(results):
            out += f'''
--------------------------------------------------
TITLE: {title}
URL: {url}
RESULT {i+1} / {len(results)}:

{result}

'''
        out += '\n\n'
    return out


##################################################
# Document

def code_action_document(params: CodeActionParams):
    '''Trigger a GPT Autocompletion response. A code action calls a command,
    which is set up below to `tell` the actor to start streaming a response.'''
    text_document = params.text_document
    range = params.range
    return CodeAction(
        title='Document',
        kind=CodeActionKind.Refactor,
        command=Command(
            title='Document',
            command='command.document',
            # Note: these arguments get jsonified, not passed as python objs
            arguments=[text_document, range]
        )
    )


##################################################
# Setup

def configure(config_yaml):
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_cache', default=get_nested(config_yaml, ['document', 'download_cache']))
    parser.add_argument('--embedding_cache', default=get_nested(config_yaml, ['document', 'embedding_cache']))
    parser.add_argument('--db_path', default=get_nested(config_yaml, ['document', 'db_path']))
    parser.add_argument('--model_name', default=get_nested(config_yaml, ['document', 'model_name']))
    parser.add_argument('--denoise_window_size', default=get_nested(config_yaml, ['document', 'denoise_window_size']))
    parser.add_argument('--denoise_poly_order', default=get_nested(config_yaml, ['document', 'denoise_poly_order']))
    parser.add_argument('--percentile', default=get_nested(config_yaml, ['document', 'percentile']))

    # bc this is only concerned with this module's params, do not error if
    # extra params are sent via cli.
    args, _ = parser.parse_known_args()
    return args


def initialize(config, server):
    # Actor
    server.add_actor(NAME, DocumentActor)

    # CodeActions
    server.add_code_action(
        lambda params:
        code_action_document(params))

    server.tell_actor(NAME, {
        'command': 'set_config',
        'config': config
    })
    server.tell_actor(NAME, {
        'command': 'initialize'
    })

    # Modify Server
    @server.thread()
    @server.command('command.document')
    def document(ls: Server, args):
        if len(args) != 2:
            log.error(f'command.document: Wrong arguments, received: {args}')
        text_document = ls.converter.structure(args[0], TextDocumentIdentifier)
        range = ls.converter.structure(args[1], Range)
        uri = text_document.uri
        doc = ls.workspace.get_document(uri)
        doc_source = doc.source

        # Extract the highlighted region
        prompt = extract_range(doc_source, range)

        # Send a message to start the stream
        actor_args = {
            'command': 'start',
            'uri': uri,
            'range': range,
            'prompt': prompt,
            'edits': ls.edits,
            'doc': doc_source,
        }
        ls.tell_actor(NAME, actor_args)

        # Return null-edit immediately (the rest will stream)
        return WorkspaceEdit()
