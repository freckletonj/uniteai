'''

An opinionated text-embedding suite for use with `document_chat`.

TODO: This should probably use `pgvector` eventually


DEPENDENCIES:
- InstructorEmbedding
- sentence-transformers

Note, I'm not sure which embedding lib I like most; need to test.

'''

from scipy.signal import savgol_filter
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import logging
import numpy as np
import os
from uniteai.common import mk_logger, read_unicode
import hashlib
import sys

# TQDM can be useful for debugging, but prints to stderr which makes VSCode
# flip out.
DEBUG = False
if DEBUG:
    from tqdm import tqdm
else:
    def tqdm(x, *args, **kwargs):
        return x

log = mk_logger('document_chat', logging.DEBUG)


##################################################

def generate_embeddings(model,
                        document: str,
                        window_size: int,
                        stride: int) -> List[np.ndarray]:
    embeddings = []
    offsets = list(enumerate(range(0, len(document) - window_size + 1, stride)))
    for emb_i, doc_i in tqdm(offsets):
        chunk = document[doc_i:doc_i+window_size]
        chunk_e = model.encode(chunk)
        embeddings.append(chunk_e)
    return embeddings


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


class Embedding:
    '''
    Embeddings helper that handles things like caching embeddings
    '''

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


class Search:
    def __init__(self, model, cache_dir: str,
                 embedding: Embedding,
                 percentile: int = 75,
                 denoise_window_size: int = 1000,
                 denoise_poly_order: int = 2):
        self.model = model
        self.embedding = embedding
        self.percentile = percentile
        self.denoise_window_size = denoise_window_size
        self.denoise_poly_order = denoise_poly_order
        self.cache_dir = cache_dir

    def _similar_tokens(self,
                        query: str,
                        embeddings: List[np.ndarray],
                        window_size: int,
                        stride: int) -> List[float]:
        '''
        Compare a `query` to a strided window over a `document`.
        '''
        document_length = len(embeddings) * stride + window_size - 1
        similarities = np.zeros(document_length, dtype=float)
        overlaps = np.zeros(document_length, dtype=float)

        # OK, this is ugly. SentenceTransformers uses `tqdm` progress bars which
        # output to `stderr`, which VSCode interprets as an error, and then
        # blows up. To combat this, we must temporarily redirect to devnull.

        # Backup the original stdout/stderr, then restore after.
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        query_e = self.model.encode(query)  # SentenceTransformer

        sys.stdout = original_stdout
        sys.stderr = original_stderr

        offsets = list(range(0, document_length - window_size + 1, stride))
        for chunk_e, doc_i in tqdm(zip(embeddings, offsets)):
            sim = cosine_similarity(query_e.reshape(1, -1),
                                    chunk_e.reshape(1, -1))[0][0]
            for j in range(doc_i, doc_i + window_size):
                similarities[j] += sim
                overlaps[j] += 1

        similarities /= np.where(overlaps != 0, overlaps, 1)
        return similarities

    def _denoise_similarities(self, similarities):
        '''
        Apply Savitzky-Golay filter to smooth out the similarity scores.
        '''
        # return similarities

        if len(similarities) < self.denoise_window_size:
            return similarities
        else:
            return savgol_filter(similarities,
                                 self.denoise_window_size,
                                 self.denoise_poly_order)

    def _tune_percentile(self, xs):
        '''
        0-out all elements below percentile. Essentially, this will leave some
        `1-percentile` percentage of the document highlighted.
        '''
        xs = np.copy(xs)
        p = np.percentile(xs, self.percentile)
        xs[xs < p] *= 0
        return xs

    def search(self,
               path: str,
               query: str,
               window_size: int,
               stride: int,
               top_n=5,
               visualize=False):
        '''
        Entry point for the search operation.
        '''
        document = read_unicode(path)
        embeddings = self.embedding.embeddings(path, window_size, stride)
        similarities = self._similar_tokens(query, embeddings, window_size, stride)
        # # Instruct Embeddings needs this
        # last_edge = int(len(similarities) * 0.01)
        # similarities[-last_edge:] = similarities[-last_edge]
        d_similarities = self._denoise_similarities(similarities)
        d_similarities -= d_similarities.min()
        d_similarities /= d_similarities.max()
        d_similarities = self._tune_percentile(d_similarities)

        segs = segments(d_similarities, document)
        # ranked_segments = rank(segs, np.mean)[:top_n]
        ranked_segments = rank(segs, np.max)[:top_n]
        if visualize:
            import matplotlib.pyplot as plt
            plt.plot(similarities)
            plt.plot(d_similarities)
            plt.show()
        return ranked_segments
