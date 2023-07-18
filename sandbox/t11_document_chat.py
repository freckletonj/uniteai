'''

Reading in and indexing documents.

pip install pypdf
pip install InstructorEmbedding
pip install sentence-transformers

'''

import os
from InstructorEmbedding import INSTRUCTOR
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from tqdm import tqdm
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter

WINDOW_SIZE = 300
STRIDE = 100
PERCENTILE = 80

pdf_paths = {
    'bitcoin': '~/Documents/misc/bitcoin.pdf',
    'idaho': '~/Documents/misc/land use and development code.pdf'
}


##################################################
# PDF

def read_pdf(path):
    path = os.path.expanduser(path)
    reader = PdfReader(path)
    text = ''
    for page in reader.pages:
        text += '\n' + page.extract_text()
    return text

try:
    pdf_cache
except:
    pdf_cache = {}

    print('Loading PDF Text')
    for k, path in pdf_paths.items():
        pdf_cache[k] = read_pdf(path)

try:
    embedding_cache
except:
    embedding_cache = {}


##################################################
# Load Model

try:
    already_loaded
except:
    model = INSTRUCTOR('hkunlp/instructor-base')
    already_loaded = True

query_instruction = 'Represent the Science question for retrieving supporting documents: '
embed_instruction = 'Represent the Science document for retrieval: '


def embed(xs: List[str]):
    ''' Build sentence embeddings for each sentence in `xs` '''
    return model.encode(xs)


def similar_tokens(query: str,
                   pdf_key: str,
                   ) -> List[float]:
    '''Compare a `query to a strided window over a `document`.'''
    global embedding_cache
    # Initialize a numpy array for storing similarities and overlaps
    document = pdf_cache[pdf_key]

    similarities = np.zeros(len(document), dtype=float)
    overlaps = np.zeros(len(document), dtype=float)

    query_e = embed([[query_instruction, query]])

    if pdf_key in embedding_cache:
        embedding_is_saved = True
        embeddings = embedding_cache[pdf_key]
    else:
        embedding_is_saved = False
        embeddings = []

    # Loop through the document with given stride
    #   listify offsets to help out tqdm
    offsets = list(enumerate(range(0, len(document) - WINDOW_SIZE + 1, STRIDE)))
    for emb_i, doc_i in tqdm(offsets):
        # Extract the chunk from document
        chunk = document[doc_i:doc_i+WINDOW_SIZE]

        # Similarity
        if embedding_is_saved:
            chunk_e = embeddings[emb_i]
        else:
            chunk_e = embed([[embed_instruction, chunk]])
            embeddings.append(chunk_e)
        sim = cosine_similarity(query_e, chunk_e)[0][0]

        # Update the similarities and overlaps array
        for j in range(doc_i, doc_i + WINDOW_SIZE):
            similarities[j] += sim
            overlaps[j] += 1

    embedding_cache[pdf_key] = embeddings
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


def find_similar(query, pdf_key):
    '''
    query: a query that you want to find similar passages to
    pdf_key: {bitcoin, idaho}
    '''
    global embedding_cache

    # Embeddings
    print('Calculating embeddings')
    return similar_tokens(query, pdf_key)


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
    for score, (text, sims) in sorted(zip(scores, segments)):
        out.append(text)
    return out


def denoise_similarities(similarities, window_size=2000, poly_order=2):
    ''' Apply Savitzky-Golay filter to smooth out the similarity scores. '''
    denoised_scores = savgol_filter(similarities, window_size, poly_order)
    return denoised_scores


# query = 'whats in it for participants to the blockchain?'
# query = 'how does this protect my anonymity?'
# query = 'im concerned my hdd isnt big enough'
query = 'who contributed to this paper?'
pdf_key = 'bitcoin'
document = pdf_cache[pdf_key]
similarities = find_similar(query, pdf_key)

# remove outlier
last_edge = int(len(similarities) * 0.02)
similarities[-last_edge:] = similarities[-last_edge]

# Denoise salience scores
d_similarities = denoise_similarities(similarities)
d_similarities -= d_similarities.min()  # normalize
d_similarities /= d_similarities.max()
d_similarities = tune_percentile(d_similarities, percentile=75)

segs = segments(d_similarities, document)
ranked_segments = rank(segs, np.mean)

import matplotlib.pyplot as plt
plt.plot(similarities)
plt.plot(d_similarities)
# plt.plot(tune_percentile(similarities))
plt.show()

for x in ranked_segments:
    print('--------------------------------------------------')
    print(x)
