'''

Make web resources easy to interrogate.

1.) Fetch some files/webpages/git repos, and save them to disk

2.) Load them back up into a dict


----------
DEPS

pip install pypdf
pip install beautifulsoup4
pip install gitpython

'''

from pypdf import PdfReader
from bs4 import BeautifulSoup
import git
import os
import requests
import shutil
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import sqlite3
import hashlib
import requests
from git import Repo
from typing import List, Tuple, Union, Any
from pathlib import Path

FILETYPES = [
    '.py',  # '.ipynb',
    '.txt', '.md', '.org',
    '.java',
    '.c', '.cpp',
    '.js', '.html', '.css',
    '.hs',
]


references = {
    "Prompting is programming: A query language for large language models": 'https://arxiv.org/abs/2212.06094',
    "Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs": "https://arxiv.org/abs/2306.03081",
    "Guidance": "https://github.com/microsoft/guidance",
    "Synchromesh: Reliable code generation from pre-trained language models": "https://arxiv.org/abs/2201.11227",
    "Abstract syntax networks for code generation and semantic parsing": "https://arxiv.org/abs/1704.07535",
    "Language models are unsupervised multitask learners": 'http://cs.brown.edu/courses/cs146/assets/papers/language_models_are_unsupervised_multitask_learners.pdf',

    "PICARD: Parsing incrementally for constrained auto-regressive decoding from language models": "https://arxiv.org/abs/2109.05093",
    "Neural machine translation of rare words with subword units": "https://arxiv.org/abs/1508.07909",
    "Attention is all you need": 'https://arxiv.org/abs/1706.03762',
    "Controllable Neural Text Generation": "https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/",

    "r2d4/rellm": "https://github.com/r2d4/rellm",
    'r2d4/parserllm': 'https://github.com/r2d4/parserllm',
    'normal-computing/outlines': 'https://github.com/normal-computing/outlines',
    'shreyar/guardrails': 'https://github.com/shreyar/guardrails',
    'eth-sri/lmql': 'https://github.com/eth-sri/lmql',
    '1rgs/jsonformer': 'https://github.com/1rgs/jsonformer',
    'Shopify/torch-grammar/': 'https://github.com/Shopify/torch-grammar/',
    'ggerganov/llama.cpp': 'https://github.com/ggerganov/llama.cpp',
}


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

from io import BytesIO

class Download:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)

    def _get_url_hash(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    def _get_extension(self, content_type: str) -> str:
        mapping = {'text/html': '.html',
                   'application/json': '.json',
                   'text/plain': '.txt',
                   'application/pdf': '.pdf'
                   }
        return mapping.get(content_type, '.bin')

    def fetch(self, url: str) -> List[Tuple[str, BytesIO]]:
        url_hash = self._get_url_hash(url)
        dir_path = self.cache_dir / url_hash

        files = []
        if not dir_path.exists():
            os.makedirs(dir_path, exist_ok=True)
            if 'github.com' in url:
                Repo.clone_from(url, dir_path)
            else:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    file_ext = self._get_extension(response.headers.get('Content-Type', ''))
                    file_path = dir_path / f'{url_hash}{file_ext}'
                    with open(file_path, 'wb') as fp:
                        response.raw.decode_content = True
                        shutil.copyfileobj(response.raw, fp)

        for file_path in dir_path.iterdir():
            with open(file_path, 'rb') as fp:
                buf = BytesIO(fp.read())
                relative_path = str(file_path.relative_to(dir_path))
                files.append((relative_path, buf))

        return files

    def fetch_utf8(self, url: str) -> List[Tuple[str, str]]:
        files = self.fetch(url)
        contents = []
        for file_path, buf in files:
            file_content = buf.getvalue()
            if file_path.endswith('.pdf'):
                pdf = PdfReader(BytesIO(file_content))
                contents.append((file_path, '\n'.join(page.extract_text()
                                                      for page in pdf.pages)))
            elif file_path.endswith('.html'):
                soup = BeautifulSoup(file_content, "html.parser")
                contents.append((file_path, soup.get_text()))
            elif file_path.endswith('.txt') or file_path.endswith('.json'):
                contents.append((file_path, file_content.decode('utf-8')))
            else:
                contents.append((file_path, file_content.decode('utf-8', errors='ignore')))

        return contents


def write_to_disk(file_path, content):
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f'Successfully written to: {file_path}')
    except Exception as e:
        print(f'Error writing to file: {file_path}. Error: {e}')


def initialize_database():
    connection = sqlite3.connect('resources.db')
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resources (
            id INTEGER PRIMARY KEY,
            title TEXT,
            url TEXT,
            file_path TEXT,
            content TEXT
        )
    """)
    connection.commit()
    return connection


def download_pdf(title, url, output_path, force_redownload):
    try:
        print(f'Saving: {title}')
        file_path = os.path.join(output_path, f"{title}.pdf")
        if not force_redownload and os.path.exists(file_path):
            print(f"File {file_path} already exists, skip downloading.")
            return

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as fp:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, fp)
    except Exception as e:
        print(f'Error on Title={title}, URL={url}. Error: {e}')


def download_github(title, url, output_path, force_redownload):
    try:
        print(f'Saving: {title}')
        if not force_redownload and os.path.exists(os.path.join(output_path, title)):
            print(f"Repository {title} has already been cloned, skip downloading.")
        else:
            git.Repo.clone_from(url, os.path.join(output_path, title))
    except Exception as e:
        print(f'Error on Title={title}, URL={url}. Error: {e}')


def download_html(title, url, output_path, force_redownload):
    try:
        print(f'Saving: {title}')
        file_path = os.path.join(output_path, f"{title}.txt")
        if not force_redownload and os.path.exists(file_path):
            print(f"File {file_path} already exists, skip downloading.")
            return
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        with open(file_path, "w") as f:
            f.write(soup.get_text())
    except Exception as e:
        print(f'Error on Title={title}, URL={url}. Error: {e}')


def save_docs(references, output_path, force_redownload=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for title, url in references.items():
            title = sanitize(title)
            if url is None:
                print(f"No URL provided for {title}")
                continue

            # PDF
            path = urlparse(url).path
            if path.endswith('.pdf'):
                futures.append(executor.submit(download_pdf, title, url, output_path, force_redownload))

            # ARXIV
            elif "arxiv.org" in url:
                # Replace 'abs' with 'pdf' in arXiv URL
                url = url.replace("abs", "pdf")
                futures.append(executor.submit(download_pdf, title, url, output_path, force_redownload))

            # GITHUB
            elif "github.com" in url:
                futures.append(executor.submit(download_github, title, url, output_path, force_redownload))

            # HTML or others
            else:
                futures.append(executor.submit(download_html, title, url, output_path, force_redownload))

        # Wait for all downloads to finish
        for future in tqdm(futures, desc="Downloading"):
            future.result()


def load_docs(path):
    def extract_text_from_pdf(file_path):
        pdf = PdfReader(file_path)
        text = ''
        for page in pdf.pages:
            text += '\n' + page.extract_text()
        return text

    def read_text_from_file(file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def walk_directory(directory):
        texts = []
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if filename.endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                elif any([filename.endswith(typ) for typ in FILETYPES]):
                    text = read_text_from_file(file_path)
                else:
                    continue  # if the file type is not supported, skip this file
                texts.append((filename, text))  # save as a tuple
        return texts

    data_dict = {}
    for entry in os.scandir(path):
        if entry.is_file() and (entry.name.endswith('.pdf') or any([entry.name.endswith(typ) for typ in FILETYPES])):
            if entry.name.endswith('.pdf'):
                text = extract_text_from_pdf(entry.path)
            else:
                text = read_text_from_file(entry.path)
            data_dict[entry.name] = [(entry.name, text)]
        elif entry.is_dir():
            data_dict[entry.name] = walk_directory(entry.path)

    return data_dict


if True or __name__ == '__main__':
    # args
    ACTION = 'load'  # {save, load}
    OUTPUT_PATH = "./t12_outputs"

    if ACTION == 'save':
        print('SAVING')
        save_docs(references, OUTPUT_PATH, force_redownload=False)

    if ACTION == 'load':
        print('LOADING')
        loaded = load_docs(OUTPUT_PATH)
