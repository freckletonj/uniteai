'''

An opinionated text-embedding suite for use with `document_chat`.

InstructorEmbedding
sentence-transformers

'''

from InstructorEmbedding import INSTRUCTOR
from dataclasses import dataclass
from pypdf import PdfReader
from scipy.signal import savgol_filter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import List, Any
import argparse
import logging
import numpy as np
import os
import pickle
from asyncio import Queue
import asyncio
import threading
import concurrent.futures
from queue import Empty
import multiprocessing
import torch
from uniteai.common import mk_logger, read_unicode
import hashlib

log = mk_logger('document_chat', logging.DEBUG)


##################################################

def generate_embeddings(model, document: str, window_size: int, stride: int) -> List[np.ndarray]:
    embeddings = []
    offsets = list(enumerate(range(0, len(document) - window_size + 1, stride)))
    for emb_i, doc_i in tqdm(offsets):
        chunk = document[doc_i:doc_i+window_size]
        chunk_e = model.encode(chunk)
        embeddings.append(chunk_e)
    return embeddings


class Embedding:
    def __init__(self, model, cache_dir: str):
        self.model = model
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _cache_path(self, path: str, window_size: int, stride: int) -> str:
        name = hashlib.sha1(path.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{name}_{window_size}_{stride}.npy")

    def _load_embeddings_from_cache(self, path: str, window_size: int, stride: int) -> List[np.ndarray]:
        cache_path = self._cache_path(path, window_size, stride)
        if os.path.exists(cache_path):
            return np.load(cache_path, allow_pickle=True)
        return None

    def _save_embeddings_to_cache(self, path: str, window_size: int, stride: int, embeddings: List[np.ndarray]):
        cache_path = self._cache_path(path, window_size, stride)
        np.save(cache_path, embeddings)

    def embeddings(self, path: str, window_size: int, stride: int) -> List[np.ndarray]:
        embeddings = self._load_embeddings_from_cache(path, window_size, stride)
        if embeddings is None:
            document = read_unicode(path)
            embeddings = generate_embeddings(self.model, document, window_size, stride)
            self._save_embeddings_to_cache(path, window_size, stride, embeddings)
        return embeddings




##################################################

##################################################

##################################################

# A document with its embeddings, and metadata
@dataclass
class Meta:
    name: str
    path: str
    window_size: int
    stride: int
    percentile: int
    text: str
    embeddings: List[np.ndarray]
    query_instruction: str
    embed_instruction: str
    denoise_window_size: int
    denoise_poly_order: int


##################################################
# PDF

def load_doc(meta: Meta):
    ''' Mutate `meta` to include `text` '''
    if meta.text is None:
        path = meta.path
        path = os.path.expanduser(path)
        _, ext = os.path.splitext(path)
        if ext == '.pdf':
            reader = PdfReader(path)
            text = ''
            for page in reader.pages:
                text += '\n' + page.extract_text()
            meta.text = text
        else:
            with open(path, 'r') as f:
                meta.text = f.read()


def get_file_name(path):
    full_name = os.path.basename(path)
    name, ext = os.path.splitext(full_name)
    return name


def load_pkl(pdf_key):
    path = f'{pdf_key}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            xs = pickle.load(f)
        return xs
    return None


def save_pkl(pdf_key, xs):
    path = f'{pdf_key}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(xs, f)
    log.info(f'Saved: {path}')


def load_embeddings(model, meta):
    ''' Mutate `meta` to include embeddings. '''
    pdf_key = meta.name
    document = meta.text
    window_size = meta.window_size
    stride = meta.stride
    embed_instruction = meta.embed_instruction

    # Try to load embeddings from disk
    embeddings = load_pkl(pdf_key)
    if embeddings is not None:
        log.info(f'Loaded {pdf_key} embeddings')
        meta.embeddings = embeddings
    else:  # If not found, then calculate
        log.info(f'Preparing embeddings for {pdf_key}')
        embeddings = []
        # Loop through the document with given stride
        offsets = list(
            enumerate(range(0, len(document) - window_size + 1, stride)))
        for emb_i, doc_i in tqdm(offsets):
            # Extract the chunk from document
            chunk = document[doc_i:doc_i+window_size]
            # Embed the chunk
            chunk_e = model.encode([[embed_instruction, chunk]])
            embeddings.append(chunk_e)

        meta.embeddings = embeddings
        save_pkl(pdf_key, embeddings)


def similar_tokens(model, query: str, meta: Meta) -> List[float]:
    '''Compare a `query` to a strided window over a `document`.'''
    embeddings = meta.embeddings
    pdf_key = meta.name
    document = meta.text
    window_size = meta.window_size
    stride = meta.stride
    query_instruction = meta.query_instruction
    # Initialize a numpy array for storing similarities and overlaps
    document_length = len(embeddings) * stride + window_size - 1  # Derive the document length from embeddings
    similarities = np.zeros(document_length, dtype=float)
    overlaps = np.zeros(document_length, dtype=float)

    query_e = model.encode([[query_instruction, query]])

    # Loop through the document with given stride
    offsets = list(range(0, document_length - window_size + 1, stride))
    for chunk_e, doc_i in tqdm(zip(embeddings, offsets)):
        sim = cosine_similarity(query_e, chunk_e)[0][0]

        # Update the similarities and overlaps array
        for j in range(doc_i, doc_i + window_size):
            similarities[j] += sim
            overlaps[j] += 1

    # Average the similarities with the number of overlaps
    similarities /= np.where(overlaps != 0, overlaps, 1)
    return similarities


def find_spans(arr, threshold=0.5):
    ''' '''
    # Create an array that is 1 where arr is above threshold, and padded with 0s at the edges
    is_over_threshold = np.concatenate(([0], np.greater(arr, threshold), [0]))

    # Find the indices of rising and falling edges
    diffs = np.diff(is_over_threshold)
    starts = np.where(diffs > 0)[0]
    ends = np.where(diffs < 0)[0]
    return list(zip(starts, ends - 1))


def tune_percentile(xs, percentile):
    ''' 0-out all elements below percentile. Essentially, this will leave some
    `1-percentile` percentage of the document highlighted. '''
    xs = np.copy(xs)  # don't mutate original
    p = np.percentile(xs, percentile)
    xs[xs < p] *= 0
    return xs


def segments(similarities, document, threshold=0.0):
    out = ''
    last_thresh = False  # for finding edge

    text = ''
    sims = []
    out = []  # [(text, sims), ...]
    for sim, char in zip(similarities, document):
        super_thresh = sim > threshold
        # no longer a super_thresh run
        if last_thresh and not super_thresh:
            out.append((text, np.array(sims)))
            text = ''
            sims = []

        # is a super_thresh run
        if super_thresh:
            text += char
            sims.append(sim)
        last_thresh = super_thresh
    if len(text) > 0:
        out.append((text, np.array(sims)))

    return out


def rank(segments, rank_fn):
    '''Sort segments according to an aggregate function of their scores.'''
    scores = []
    for text, sims in segments:
        scores.append(rank_fn(sims))
    out = []
    for score, (text, sims) in reversed(sorted(zip(scores, segments))):
        out.append(text)
    return out


def denoise_similarities(similarities, window_size=2000, poly_order=2):
    ''' Apply Savitzky-Golay filter to smooth out the similarity scores. '''
    return savgol_filter(similarities, window_size, poly_order)


def top_segments(model, cache, query, doc_name, top_n, visualize=False):
    meta = cache[doc_name]
    document = meta.text
    denoise_window_size = meta.denoise_window_size
    denoise_poly_order = meta.denoise_poly_order
    percentile = meta.percentile
    similarities = similar_tokens(model, query, meta)

    # remove outlier at end
    last_edge = int(len(similarities) * 0.01)
    similarities[-last_edge:] = similarities[-last_edge]

    # Denoise salience scores
    # similarities = tune_percentile(similarities, percentile)
    d_similarities = denoise_similarities(similarities,
                                          denoise_window_size,
                                          denoise_poly_order)
    d_similarities -= d_similarities.min()  # normalize
    d_similarities /= d_similarities.max()
    d_similarities = tune_percentile(d_similarities, percentile)

    segs = segments(d_similarities, document)
    ranked_segments = rank(segs, np.mean)[:top_n]

    if visualize:
        import matplotlib.pyplot as plt
        plt.plot(similarities)
        plt.plot(d_similarities)
        plt.show()

    return ranked_segments, d_similarities


##################################################
# Ranked Segments

def make_cache(model):
    log.info('Populating cache')
    cache = {
        'bitcoin': Meta(
            name='bitcoin',
            path='./discord_bot/bots/document_chat/bitcoin.pdf',
            window_size=300,
            stride=57,
            text=None,
            embeddings=None,
            query_instruction='Represent the Science question for retrieving supporting documents: ',
            embed_instruction='Represent the Science document for retrieval: ',
            denoise_window_size=2000,
            denoise_poly_order=2,
            percentile=80,
        ),
        # 'idaho': Meta(
        #     name='idaho',
        #     path='./discord_bot/bots/document_chat/land use and development code.pdf',
        #     window_size=300,
        #     stride=100,
        #     text=None,
        #     embeddings=None,
        #     query_instruction='Represent the wikipedia question for retrieving supporting documents: ',
        #     embed_instruction='Represent the wikipedia document for retrieval: ',
        #     denoise_window_size=5000,
        #     denoise_poly_order=2,
        #     percentile=80,
        # ),
        'alice': Meta(
            name='alice',
            path='./discord_bot/bots/document_chat/alice.txt',
            window_size=300,
            stride=57,
            text=None,
            embeddings=None,
            query_instruction='Represent the novel question for retrieving supporting documents: ',
            embed_instruction='Represent the novel document for retrieval: ',
            denoise_window_size=2000,
            denoise_poly_order=2,
            percentile=80,
        ),
    }
    for k, m in cache.items():
        load_doc(m)
        load_embeddings(model, m)

    return cache

@dataclass
class Job:
    ctx: Any
    proc_reply: Any
    query: str
    doc_name: str


def go(model, cache, query, doc_name, top_n):
    log.debug(f'processing for {doc_name}: {query}')
    ranked_segments, sims = top_segments(model, cache, query, doc_name, top_n)
    out_msg = f"**Query on {doc_name}:** {query}\n\n"
    for i, seg in enumerate(ranked_segments):
        # discord has length limit of 2000
        out_msg += f'**Relevant Passage #{i+1}:**\n'
        out_msg += seg[:900] + '... <CONTINUES>'
        out_msg += '\n\n'
    return out_msg

async def process_queue(queue):
    ''' Single-thread calls to the `model` '''
    log.info('Starting process queue')
    model = INSTRUCTOR('hkunlp/instructor-base')
    cache = make_cache(model)

    while True:

        ##########
        # Process requests
        try:
            try:
                job = await asyncio.wait_for(queue.get(), 0.1)
            except asyncio.TimeoutError:
                continue

            await job.proc_reply.edit(content=f"Waiting in line. Request: {job.query}")
            log.info(f'Got job: {job.query}')

            with concurrent.futures.ThreadPoolExecutor() as pool:
                out_msg = await asyncio.get_running_loop().run_in_executor(
                    pool,
                    go,
                    model,
                    cache,
                    job.query,
                    job.doc_name,
                    2)
                await job.proc_reply.edit(content=out_msg)

            queue.task_done()
        except Exception as e:
            log.error(e)


# TODO: Ideally, we'd init `process_queue` in `on_ready`, but, this is quicker
#       and I'm out of time.
is_started = threading.Event()


def configure(config_yaml):
    parser = argparse.ArgumentParser()
    # bc this is only concerned with this module's params, do not error if
    # extra params are sent via cli.
    args, _ = parser.parse_known_args()
    return args


def initialize(args, server):
    log.info('Initializing Document Chat Bot')
    q = Queue()
    def start_process():
        # Initialize `process_queue` if needed.
        if not is_started.is_set():
            log.info('Starting job queue for document_chat')
            asyncio.create_task(process_queue(q))
            is_started.set()

    @server.hybrid_command(name="ask_bitcoin",
                           description="Look for relevant passages from the original Bitcoin whitepaper.")
    async def ask_bitcoin(ctx, query: str):
        proc_reply = await ctx.reply("Processing...")
        await q.put(Job(ctx, proc_reply, query, 'bitcoin'))
        start_process()

    # @server.hybrid_command(name="ask_idaho",
    #                        description="Search the Idaho Land Use Code, a 400 page doc.")
    # async def ask_idaho(ctx, query: str):
    #     proc_reply = await ctx.reply("Processing...")
    #     await q.put(Job(ctx, proc_reply, query, 'idaho'))
    #     start_process()

    @server.hybrid_command(name="ask_alice",
                           description="Ask questions about Alice in Wonderland, find relevant passages.",)
    async def ask_alice(ctx, query: str):
        proc_reply = await ctx.reply("Processing...")
        await q.put(Job(ctx, proc_reply, query, 'alice'))
        start_process()

    return server
