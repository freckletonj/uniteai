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
import pickle
from dataclasses import dataclass

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
# Load Model

try:
    already_loaded
except:
    model = INSTRUCTOR('hkunlp/instructor-base')
    already_loaded = True



def embed(xs: List[str]):
    ''' Build sentence embeddings for each sentence in `xs` '''
    return model.encode(xs)


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
    print(f'Saved: {path}')


def load_embeddings(meta):
    ''' Mutate `meta` to include embeddings. '''
    pdf_key = meta.name
    document = meta.text
    window_size = meta.window_size
    stride = meta.stride
    embed_instruction = meta.embed_instruction

    # Try to load embeddings from disk
    embeddings = load_pkl(pdf_key)
    if embeddings is not None:
        print(f'Loaded {pdf_key} embeddings')
        meta.embeddings = embeddings
    else:  # If not found, then calculate
        print(f'Preparing embeddings for {pdf_key}')
        embeddings = []
        # Loop through the document with given stride
        offsets = list(
            enumerate(range(0, len(document) - window_size + 1, stride)))
        for emb_i, doc_i in tqdm(offsets):
            # Extract the chunk from document
            chunk = document[doc_i:doc_i+window_size]
            # Embed the chunk
            chunk_e = embed([[embed_instruction, chunk]])
            embeddings.append(chunk_e)

        meta.embeddings = embeddings
        save_pkl(pdf_key, embeddings)


def similar_tokens(query: str, meta: Meta) -> List[float]:
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

    query_e = embed([[query_instruction, query]])

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


def top_segments(query, doc_name, top_n, visualize=False):
    meta = cache[doc_name]
    document = meta.text
    denoise_window_size = meta.denoise_window_size
    denoise_poly_order = meta.denoise_poly_order
    percentile = meta.percentile
    similarities = similar_tokens(query, meta)

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
# Visualization

import webbrowser
from html import escape
import os
from typing import List, Tuple
import numpy as np


def hex_to_rgb(hex_color: str):
    """ Converts a hexadecimal color string to an RGB tuple. """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def interpolate_color(similarity: float, color):
    """ Scales an RGB color tuple by a similarity factor. """
    rgb_color = hex_to_rgb(color)
    return tuple(similarity * channel for channel in rgb_color)


def add_color(a, b):
    """ Adds two RGB color tuples element-wise. """
    return tuple(ai + bi for ai, bi in zip(a, b))


def rgb_to_hex(rgb):
    """ Converts an RGB color tuple to a hexadecimal color string. """
    r,g,b = rgb
    rgb = (int(r), int(g), int(b))
    return '#%02x%02x%02x' % rgb


def colorize_text(text: str, queries_and_colors_and_similarity: List[Tuple[str, str, np.ndarray]]) -> str:
    """ Colorizes text based on similarity scores for different queries. """
    # Initialize an empty list for the HTML parts
    html_parts = []

    # Get the color names and similarity scores
    color_names = [item[1] for item in queries_and_colors_and_similarity]
    similarities_tr = zip(*[item[2] for item in queries_and_colors_and_similarity])  # transposed

    # Loop through the text with the corresponding similarity
    for similarities, char in zip(similarities_tr, text):
        # Initialize color as black
        color = (0, 0, 0)
        for name, sim in zip(color_names, similarities):
            color = add_color(color, interpolate_color(sim, name))
        hex_color = rgb_to_hex(color)

        if char == '\n':
            html_parts.append('<br>')
            continue
        char = escape(char)
        html_parts.append(f'<span style="color: {hex_color};">{char}</span>')

    return ''.join(html_parts)


def create_colorful_html(queries_and_colors_and_similarity: List[Tuple[str, str, np.ndarray]],
                         document: str) -> None:
    """ Creates an HTML page with colorized query and document text. """
    # Colorize the queries and document
    queries_html = [colorize_text(query, [(query, color, np.ones(len(query)))])
                    for query, color, self_similarities
                    in queries_and_colors_and_similarity]
    document_html = colorize_text(document, queries_and_colors_and_similarity)

    # Combine the HTML strings for the queries and document
    html = '''
<head>
  <style>
    body {{
      background-color: #DDDDDD;
    }}
  </style>
</head>
'''
    for i, query_html in enumerate(queries_html):
        html += f'<h1>QUERY {i+1}:</h1>\n<span>{query_html}</span>\n<br><br>\n'
    html += '<hr>\n<br><br>\n<h1>DOCUMENT:</h1>\n' + document_html

    # Write the HTML string to a temporary file
    with open('temp.html', 'w') as f:
        f.write(html)

    # Open the HTML file in the default web browser
    webbrowser.open('file://' + os.path.realpath('temp.html'))


##################################################
# Ranked Segments

try:
    cache_is_loaded
except:
    print('Populating cache')
    cache = {
        'bitcoin': Meta(
            name='bitcoin',
            path='~/Documents/misc/bitcoin.pdf',
            window_size=300,
            stride=100,
            text=None,
            embeddings=None,
            query_instruction='Represent the Science question for retrieving supporting documents: ',
            embed_instruction='Represent the Science document for retrieval: ',
            denoise_window_size=2000,
            denoise_poly_order=2,
            percentile=80,
        ),
        'idaho': Meta(
            name='idaho',
            path='~/Documents/misc/land use and development code.pdf',
            window_size=300,
            stride=100,
            text=None,
            embeddings=None,
            query_instruction='Represent the wikipedia question for retrieving supporting documents: ',
            embed_instruction='Represent the wikipedia document for retrieval: ',
            denoise_window_size=5000,
            denoise_poly_order=2,
            percentile=80,
        ),
        '2001_positive': Meta(
            name='2001_positive',
            path='./2001_positive.md',
            window_size=500,
            stride=25,
            text=None,
            embeddings=None,
            query_instruction='Represent the book review question for retrieving supporting documents: ',
            embed_instruction='Represent the book review document for retrieval: ',
            denoise_window_size=250,
            denoise_poly_order=3,
            percentile=80,
        ),
        '2001_negative': Meta(
            name='2001_negative',
            path='./2001_negative.md',
            window_size=500,
            stride=25,
            text=None,
            embeddings=None,
            query_instruction='Represent the book review question for retrieving supporting documents: ',
            embed_instruction='Represent the book review document for retrieval: ',
            denoise_window_size=250,
            denoise_poly_order=3,
            percentile=80,
        ),
    }
    for k, m in cache.items():
        load_doc(m)
        load_embeddings(m)
    cache_is_loaded = True

# # doc_name = 'bitcoin'
# # query = 'whats in it for participants to the blockchain?'
# # query = 'how does this protect my anonymity?'
# # query = 'im concerned my hdd isnt big enough'
# # query = 'who contributed to this paper?'

# doc_name = 'idaho'
# # query = 'how close can my silver mine be to a farm?'
# # query = 'how do houses on the lake need to be addressed? marine addressing.'
# # query = 'How can I rezone my property? Rezoning.'
# # query = 'What signs can I put on my property?'
# query = 'How does this document define "sign"?'

# ranked_segments, sims = top_segments(query, doc_name, top_n=3)

# for x in reversed(ranked_segments):
#     print('\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n')
#     print(x)


##################################################
# Colorize document

doc_name = '2001_positive'
queries = [
    ('The reviewer loves the book.', '00cc00'),
    ('The reviewer hates the book.', 'cc0000'),
]
query_tups = []
for (q, c) in queries:
    _, sims = top_segments(q, doc_name, top_n=0)
    query_tups.append((q, c, sims))

create_colorful_html(query_tups, cache[doc_name].text)
