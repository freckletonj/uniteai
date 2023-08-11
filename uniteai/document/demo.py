'''

Stitch together `download` and `embed`. This file should be deleted eventually.

'''

import uniteai.document.download as d
import uniteai.document.embed as e

from InstructorEmbedding import INSTRUCTOR
from dataclasses import dataclass
from pypdf import PdfReader
from scipy.signal import savgol_filter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import List, Any, Union
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
import sqlite3
from sentence_transformers import SentenceTransformer

# if __name__ != '__main__':
#     BREAK

DOWNLOAD_CACHE = "./download_cache/"
EMBEDDING_CACHE = "./embedding_cache/"
DB_PATH = 'db.sqlite3'


##################################################
# Downloads

references = [
    (None, 'https://arxiv.org/abs/2212.06094'),
    (None, "https://arxiv.org/abs/2306.03081"),
    (None, "https://github.com/microsoft/guidance"),
    (None, "https://arxiv.org/abs/2201.11227"),
    (None, "https://arxiv.org/abs/1704.07535"),
    (None, 'http://cs.brown.edu/courses/cs146/assets/papers/language_models_are_unsupervised_multitask_learners.pdf'),
    (None, "https://arxiv.org/abs/2109.05093"),
    (None, "https://arxiv.org/abs/1508.07909"),
    (None, 'https://arxiv.org/abs/1706.03762'),
    (None, "https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/"),
    (None, "https://github.com/r2d4/rellm"),
    (None, 'https://github.com/r2d4/parserllm'),
    (None, 'https://github.com/normal-computing/outlines'),
    (None, 'https://github.com/shreyar/guardrails'),
    (None, 'https://github.com/eth-sri/lmql'),
    (None, 'https://github.com/1rgs/jsonformer'),
    (None, 'https://github.com/Shopify/torch-grammar/'),
    (None, 'https://github.com/ggerganov/llama.cpp'),
    (None, 'https://www.youtube.com/watch?v=Se91Pn3xxSs'),
    (None, 'http://paulgraham.com/ds.html'),
    ('alice in wonderland', 'https://www.gutenberg.org/cache/epub/11/pg11.txt'),
]


if False:
    # args
    ACTION = 'save'  # {save, load}
    d.initialize_database(DB_PATH)

    dl = d.Downloader(DOWNLOAD_CACHE)

    if ACTION == 'save':
        print('SAVING')
        saved = d.save_docs(references, dl, DB_PATH)

    if ACTION == 'load':
        print('LOADING')
        loaded = {}
        for url in references:
            loaded[url] = dl.fetch_utf8(url)


def load_from_database(connection, title: Union[str, None], url):
    cursor = connection.cursor()
    if title is not None:
        cursor.execute("""
SELECT title, file_path, content, url FROM resources
WHERE title = ? OR url = ?
""", (title, url))
    else:
        cursor.execute("""
SELECT title, file_path, content, url FROM resources
WHERE url = ?
""", (url,))
    result = cursor.fetchone()
    return result if result else (None, None, None, None)


def search_helper(search,
                  db_path,
                  titles_urls,
                  query,
                  window_size=200, stride=50, top_n=5, visualize=False):
    with sqlite3.connect(db_path) as conn:
        results = []
        for title, url in titles_urls:
            title, file_path, content, url = load_from_database(conn, title, url)
            if not title:
                continue
            result = search.search(file_path, query, window_size, stride, top_n, visualize)
            results.append((url, title, result))
        return results


##################################################
# Test

if False:
    try:
        already_loaded
    except:
        # model = INSTRUCTOR('hkunlp/instructor-base')

        # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        already_loaded = True

    embedding = e.Embedding(model, EMBEDDING_CACHE)
    search = e.Search(model,
                      EMBEDDING_CACHE,
                      embedding,
                      denoise_window_size=1000,
                      denoise_poly_order=1,
                      percentile=95,)

    titles_urls = [
        # ('alice in wonderland', 'https://www.gutenberg.org/cache/epub/11/pg11.txt'),
        # (None, 'https://github.com/normal-computing/outlines'),
        (None, 'https://www.youtube.com/watch?v=Se91Pn3xxSs'),
    ]

    # query = 'where is the scene where her size changes?'
    # query = 'the most psychedelic scene.'
    # query = 'sea creatures'
    # query = 'the girl meets a cat for the first time'
    # query = 'alice is asleep'
    # query = 'alice meets royalty'
    # query = 'corporal punishment'
    # query = 'alice talks to a king'
    # query = 'alice talks to a queen'
    # query = 'alice talks to a rabbit'
    # query = 'the chapter title'

    # query = 'a benefit of artificial intelligence'
    query = 'the greatest dangers from artificial intelligence'
    # query = 'an impact on artists'
    # query = 'a new industry will come from A.I.'


    xs = search_helper(search,
                       DB_PATH,
                       titles_urls,
                       query,
                       window_size=5000,
                       stride=50,
                       top_n=3,
                       visualize=True)

    print('\n'*20)
    for url, title, res in xs:
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(f'TITLE: {title}')
        for r in res:
            print('------------------------------')
            print(r)
