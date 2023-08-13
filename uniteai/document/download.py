'''

File Download

'''

from bs4 import BeautifulSoup
import os
import requests
import re
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from git import Repo
from typing import List, Tuple, Union
from pathlib import Path
from io import BytesIO
from youtube_transcript_api import YouTubeTranscriptApi
from uniteai.common import read_unicode
from uniteai.common import mk_logger
import logging


# TQDM can be useful for debugging, but prints to stderr which makes VSCode
# flip out.
DEBUG = False
if DEBUG:
    from tqdm import tqdm
else:
    def tqdm(x, *args, **kwargs):
        return x

FILETYPES = [
    '.pdf',
    '.py',  '.pyc', '.ipynb',
    '.txt', '.md', '.org',
    '.java',
    '.c', '.h', '.cpp',
    '.js', '.html', '.css', '.ts', '.jsx',
    '.hs',
]


log = mk_logger('document-download', logging.DEBUG)


##################################################
# Youtube

def extract_video_id(url):
    regex_patterns = [
        r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",      # Standard URL
        r"youtu\.be/([a-zA-Z0-9_-]{11})",                  # Shortened URL
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",         # Embed URL
        r"youtube\.com/v/([a-zA-Z0-9_-]{11})",             # V URL
        # URL with timestamp
        r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})&.*#t=(\d+)m(\d+)s",
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
}


def get_video_title(url: str):
    ''' Try to get the title of a Youtube video. '''
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.find('meta', property="og:title")["content"]


def get_transcript(url):
    ''' Try to get the transcript of a Youtube video. '''
    video_id = extract_video_id(url)
    if video_id is None:
        log.error("Could not extract video ID from URL")
        return None

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ""
        for line in transcript:
            full_transcript += " " + line['text']
        return full_transcript
    except Exception as e:
        log.error("An error occurred:", e)
        return None


##################################################
# Download

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


def normalize_local_uri(uri):
    if uri.startswith('file://'):
        uri = uri[len('file://'):]
    uri = os.path.expanduser(uri)
    return uri


def is_local_uri(uri):
    uri = normalize_local_uri(uri)
    if os.path.exists(uri):
        return True
    else:
        return False


def read_from_disk(path: Path) -> List[Tuple[Path, BytesIO]]:
    files = []

    # If path is a directory
    if path.is_dir():
        for file_path in path.iterdir():
            if file_path.suffix in FILETYPES:
                with open(file_path, 'rb') as fp:
                    buf = BytesIO(fp.read())
                    files.append((file_path, buf))

    # If path is a file
    elif path.is_file() and path.suffix in FILETYPES:
        with open(path, 'rb') as fp:
            buf = BytesIO(fp.read())
            files.append((path, buf))

    # Path is neither a file nor a directory (or it's an unsupported filetype)
    else:
        log.warning(f"Unsupported path or filetype: {path}")

    return files


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
        # a lookup won't work in cases where, eg,
        #   Content-Type: text/plain; charset=utf-8
        for k, ext in mapping.items():
            if content_type in k:
                return ext
        return '.txt'

    def _preprocess_url(self, url: str):
        ''' Some URLs deserve special treatment. '''
        if 'arxiv.org' in url:
            url = url.replace('abs', 'pdf')
        return url

    def fetch(self, title: Union[str, None], uri: str) -> List[BytesIO]:
        ''' Fetches bytes of a file (either from disk or online) based on the provided URI. '''
        if is_local_uri(uri):
            return self.fetch_offline(uri)
        else:
            return self.fetch_online(title, uri)

    def fetch_offline(self, uri: str) -> List[BytesIO]:
        ''' Fetches bytes of a file from the local disk based on the provided URI. '''
        uri = normalize_local_uri(uri)
        dir_path = Path(uri)

        if not dir_path.exists():
            log.error(f'Local path does not exist: {uri}')
            return []

        return read_from_disk(dir_path)

    def fetch_online(self, title: Union[str, None], url: str) -> List[BytesIO]:
        ''' Fetches the bytes of a file, reads from cache if it already exists,
        or downloads otherwise.

        Notice it returns a List because in cases like a git repo, there are
        many files represented by one URL.'''
        # Preprocess
        url = self._preprocess_url(url)

        url_sane = sanitize(url)  # instead of hashing
        dir_path = self.cache_dir / url_sane

        # Save if not cached
        if not dir_path.exists():
            log.info(f'URL hasnt been cached, fetching')
            # Github
            if 'github.com' in url:
                log.info(f'Downloading Github URL: {url}')
                os.makedirs(dir_path, exist_ok=True)
                Repo.clone_from(url, dir_path)

            # Youtube
            elif 'youtube.com' in url or 'youtu.be' in url:
                log.info(f'Downloading Youtube URL: {url}')
                video_id = extract_video_id(url)
                if video_id is None:
                    log.error("Could not extract video ID from URL")
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

                        os.makedirs(dir_path, exist_ok=True)
                        with open(file_path, 'w') as fp:
                            fp.write(full_transcript)
                    except Exception as e:
                        log.error("An error occurred:", e)

            # Other
            else:
                try:
                    response = requests.get(url, headers=headers)
                except requests.exceptions.SSLError as e:
                    log.error(f'SSLError: {e}')
                    # response = requests.get(url, headers=headers, verify=False)
                    return []

                if response.status_code == 200:
                    file_ext = self._get_extension(response.headers.get('Content-Type', ''))

                    # HTML
                    if file_ext == '.html':
                        log.info(f'Downloading URL as HTML: {url}')
                        content = response.text
                        soup = BeautifulSoup(content, 'html.parser')

                        # infer title
                        if title is None:
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
                            title = url_sane

                        file_path = dir_path / f'{title}{file_ext}'
                        os.makedirs(dir_path, exist_ok=True)
                        with open(file_path, 'w') as fp:
                            log.debug(f'writing: {file_path}')
                            fp.write(soup.get_text(separator=' '))

                    # PDF
                    elif file_ext == '.pdf':
                        log.info(f'Downloading URL as PDF: {url}')
                        if title is None:
                            title = url_sane
                        file_path = dir_path / f'{title}{file_ext}'
                        os.makedirs(dir_path, exist_ok=True)
                        with open(file_path, 'wb') as fp:  # note the 'wb' here
                            log.debug(f'writing: {file_path}')
                            fp.write(response.content)  # note the use of content instead of text here

                    # OTHER
                    else:
                        log.info(f'Downloading URL as "Other": {url}')
                        content = response.text
                        if title is None:
                            title = url_sane
                        file_path = dir_path / f'{title}{file_ext}'
                        os.makedirs(dir_path, exist_ok=True)
                        with open(file_path, 'w') as fp:
                            log.debug(f'writing: {file_path}')
                            fp.write(content)
                else:
                    log.error(f'NOT FOUND: {url}, {response.status_code}, {response}')

        files = read_from_disk(dir_path)
        return files

    def fetch_utf8(self, title: Union[str, None], url: str) -> List[Tuple[str, str]]:
        ''' Fetches and converts files to unicode. Reads from cache if it
        already exists, or downloads otherwise. '''
        files = self.fetch(title, url)
        contents = []
        for file_path, buf in files:
            contents.append((file_path, read_unicode(file_path, buf)))
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

def save_url(title: Union[str, None], url: str, dl: Downloader, db_path):
    files = dl.fetch_utf8(title, url)
    with sqlite3.connect(db_path) as conn:
        for path, content in files:
            if title is None:
                title = path.stem
            # print(f'{title}, {path.stem}, {url[:20]}, {str(path)[-20:]}, {content[:100]}')
            insert_to_database(conn, url, title, str(path), content)
        return files


def save_docs(references, dl, db_path):
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for title, url in references:
            if url is None:
                log.error("No URL provided")
                continue

            futures.append(executor.submit(save_url, title, url, dl, db_path))

        # Wait for all downloads to finish
        out = []
        for future in tqdm(futures, desc="Fetching"):
            files = future.result()
            out += files
        return out
