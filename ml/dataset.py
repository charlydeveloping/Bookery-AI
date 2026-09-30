import json
import re
import unicodedata
from pathlib import Path


def normalize_text(value):
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def tokens(value):
    return re.findall(r"\w+", normalize_text(value).casefold(), re.UNICODE)


def load_books(path):
    books = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(books, list) or not books:
        raise ValueError("Catalog must be a nonempty JSON array")
    ids = set()
    for book in books:
        if not all(book.get(key) for key in ("id", "title", "author", "description", "genres", "language", "searchable_text")):
            raise ValueError(f"Incomplete book: {book.get('id', '?')}")
        if book["id"] in ids:
            raise ValueError(f"Duplicate book ID: {book['id']}")
        ids.add(book["id"])
    return books


def matches(book, language=None, genre=None, author=None):
    if language and book["language"].casefold() != language.casefold():
        return False
    if genre and not any(genre.casefold() in item.casefold() for item in book["genres"]):
        return False
    if author and author.casefold() not in book["author"].casefold():
        return False
    return True
