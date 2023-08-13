'''

Stitch together `download` and `embed`. This file should be deleted eventually.

'''

from typing import Union
import sqlite3


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
