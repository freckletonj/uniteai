'''

File Downloader

'''


from pypdf import PdfReader
from bs4 import BeautifulSoup
import os
import requests
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from git import Repo
from typing import List, Tuple, Union, Any
from pathlib import Path
from io import BytesIO
import nbformat
from youtube_transcript_api import YouTubeTranscriptApi
from uniteai.common import read_unicode

FILETYPES = [
    '.pdf',
    '.py',  '.pyc', '.ipynb',
    '.txt', '.md', '.org',
    '.java',
    '.c', '.h', '.cpp',
    '.js', '.html', '.css', '.ts', '.jsx',
    '.hs',
]


##################################################
# Youtube

def extract_video_id(url):
    regex_patterns = [
        r"youtube\.com/watch\?v=(\w+)",  # Standard URL
        r"youtu\.be/(\w+)",              # Shortened URL
        r"youtube\.com/embed/(\w+)",     # Embed URL
        r"youtube\.com/v/(\w+)",         # V URL
        # URL with timestamp
        r"youtube\.com/watch\?v=(\w+)&.*#t=(\d+)m(\d+)s",
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_video_title(url: str):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.find('meta', property="og:title")["content"]


def get_transcript(url):
    ''' youtube '''
    video_id = extract_video_id(url)
    if video_id is None:
        print("Could not extract video ID from URL")
        return None

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ""
        for line in transcript:
            full_transcript += " " + line['text']
        return full_transcript
    except Exception as e:
        print("An error occurred:", e)
        return None


##################################################
# Downloader

def sanitize(filename):
    ''' Sanitize strings for becoming filenames. '''
    # Replace any characters not allowed in filenames with underscores
    sanitized_filename = re.sub(r'[^\w\-_.]', '_', filename)

    # Remove multiple consecutive underscores
    sanitized_filename = re.sub(r'__+', '_', sanitized_filename)

    # Remove leading and trailing underscores
    sanitized_filename = sanitized_filename.strip('_')

    # Limit the filename length to 255 characters (a common limitation on some systems)
    max_filename_length = 255
    sanitized_filename = sanitized_filename[:max_filename_length]

    return sanitized_filename


class Downloader:
    '''
    1. Download and cache a url to disk
    2. Fetch it, and convert nicely to unicode
    '''

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)

    def _get_extension(self, content_type: str) -> str:
        mapping = {'text/html': '.html',
                   'application/json': '.json',
                   'text/plain': '.txt',
                   'application/pdf': '.pdf'
                   }
        return mapping.get(content_type, '.bin')

    def _preprocess_url(self, url: str):
        if 'arxiv.org' in url:
            url = url.replace('abs', 'pdf')
        return url

    def fetch(self, url: str) -> List[BytesIO]:
        ''' Fetches the bytes of a file, reads from cache if it already exists,
        or downloads otherwise. '''
        # Preprocess
        url = self._preprocess_url(url)

        # url_hash = self._get_url_hash(url)
        url_hash = sanitize(url)  # instead of hashing
        dir_path = self.cache_dir / url_hash

        # Save if not cached
        if not dir_path.exists():
            os.makedirs(dir_path, exist_ok=True)

            # Github
            if 'github.com' in url:
                Repo.clone_from(url, dir_path)

            # Youtube
            elif 'youtube.com' in url or 'youtu.be' in url:
                video_id = extract_video_id(url)
                if video_id is None:
                    print("Could not extract video ID from URL")
                else:
                    try:
                        title = get_video_title(url)
                        if title is not None:
                            title = sanitize(title)
                        else:
                            title = video_id

                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        full_transcript = ""
                        for line in transcript:
                            full_transcript += " " + line['text']
                        file_path = dir_path / f'{title}.txt'
                        with open(file_path, 'w') as fp:
                            fp.write(full_transcript)
                    except Exception as e:
                        print("An error occurred:", e)

            # Other
            else:
                response = requests.get(url)  # , stream=True)
                if response.status_code == 200:
                    file_ext = self._get_extension(response.headers.get('Content-Type', ''))

                    if file_ext == '.html':
                        content = response.text
                        soup = BeautifulSoup(content, 'html.parser')

                        # Try to find HTML title in meta tags
                        meta_title = soup.find('meta', property="og:title")
                        if meta_title is None:
                            meta_title = soup.find('meta', property="title")
                        if meta_title is not None:
                            meta_title = meta_title["content"]

                        # Try to find HTML title in `title`
                        if meta_title is None:
                            meta_title = soup.find('title')
                        if meta_title is not None:
                            meta_title = meta_title.getText()

                        # If found a meta_title, keep it
                        if meta_title is not None:
                            title = meta_title

                    # backup title
                    if title is None:
                        title = url_hash

                    file_path = dir_path / f'{title}{file_ext}'
                    with open(file_path, 'w') as fp:
                        print(f'writing: {file_path}')
                        fp.write(soup.get_text(separator=' '))

        # Read back off disk
        files = []
        for file_path in dir_path.iterdir():
            if not any([file_path.suffix == typ for typ in FILETYPES]):
                continue
            with open(file_path, 'rb') as fp:
                buf = BytesIO(fp.read())
                relative_path = file_path.relative_to(dir_path)
                files.append((relative_path, buf))

        return files

    def fetch_utf8(self, url: str) -> List[Tuple[str, str]]:
        ''' Fetches and converts files to unicode. Reads from cache if it
        already exists, or downloads otherwise. '''
        files = self.fetch(url)
        contents = []
        for file_path, buf in files:
            contents.append(read_unicode(file_path, buf))

        return contents


##################################################
# Database

migrations = ['''
CREATE TABLE IF NOT EXISTS resources (
    id INTEGER PRIMARY KEY,
    title TEXT,
    url TEXT,
    file_path TEXT,
    content TEXT
);''',

'''
CREATE UNIQUE INDEX IF NOT EXISTS resources_unique_url_file_path ON resources(url, file_path);
''',
]


def initialize_database(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    for mig in migrations:
        cursor.execute(mig)
        connection.commit()


def insert_to_database(connection, url: str, title: str, file_path: str, content: str):
    cursor = connection.cursor()
    cursor.execute("""
INSERT OR IGNORE INTO resources (url, title, file_path, content)
VALUES (?, ?, ?, ?)
    """, (url, title, file_path, content))
    connection.commit()


##################################################
# Go

def save_url(url: str, dl: Downloader, db_path):
    files = dl.fetch_utf8(url)
    with sqlite3.connect(db_path) as conn:
        for path, content in files:
            print(f'{path.stem}, {url[:20]}, {str(path)[-20:]}, {content[:100]}')
            insert_to_database(conn, url, path.stem, str(path), content)
        return files


def save_docs(references, dl, db_path):
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for url in references:
            if url is None:
                print("No URL provided")
                continue

            futures.append(executor.submit(save_url, url, dl, db_path))

        # Wait for all downloads to finish
        out = []
        for future in tqdm(futures, desc="Fetching"):
            files = future.result()
            out += files
        return out


if True or __name__ == '__main__':
    references = [
        'https://arxiv.org/abs/2212.06094',
        "https://arxiv.org/abs/2306.03081",
        "https://github.com/microsoft/guidance",
        "https://arxiv.org/abs/2201.11227",
        "https://arxiv.org/abs/1704.07535",
        'http://cs.brown.edu/courses/cs146/assets/papers/language_models_are_unsupervised_multitask_learners.pdf',
        "https://arxiv.org/abs/2109.05093",
        "https://arxiv.org/abs/1508.07909",
        'https://arxiv.org/abs/1706.03762',
        "https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/",
        "https://github.com/r2d4/rellm",
        'https://github.com/r2d4/parserllm',
        'https://github.com/normal-computing/outlines',
        'https://github.com/shreyar/guardrails',
        'https://github.com/eth-sri/lmql',
        'https://github.com/1rgs/jsonformer',
        'https://github.com/Shopify/torch-grammar/',
        'https://github.com/ggerganov/llama.cpp',
        'https://www.youtube.com/watch?v=Se91Pn3xxSs',
        'http://paulgraham.com/ds.html',
    ]

    # args
    ACTION = 'save'  # {save, load}
    OUTPUT_PATH = "./t12_outputs"
    DB_PATH = 't12_resources.sqlite3'
    initialize_database(DB_PATH)

    dl = Downloader(OUTPUT_PATH)

    if ACTION == 'save':
        print('SAVING')
        saved = save_docs(references, dl, DB_PATH)

    if ACTION == 'load':
        print('LOADING')
        loaded = {}
        for url in references:
            loaded[url] = dl.fetch_utf8(url)
