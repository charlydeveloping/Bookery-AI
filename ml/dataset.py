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


def referenced_book_ids(query, books):
    """Skip a named reference title in 'similar to this book' requests."""
    query_words = f" {' '.join(tokens(query))} "
    markers = (" parecido a ", " parecida a ", " similar a ", " como ", " tipo ")
    if not any(marker in query_words for marker in markers):
        return set()
    excluded = set()
    for book in books:
        for title in [book["title"], *book.get("reference_aliases", [])]:
            if f" {' '.join(tokens(title))} " in query_words:
                excluded.add(book["id"])
                break
    return excluded


def expand_reference_query(query, books):
    """Use catalog facts about a named book to make similarity requests specific."""
    referenced = referenced_book_ids(query, books)
    if not referenced:
        return query
    details = [f"{', '.join(book['genres'])}. {book['description']}"
               for book in books if book["id"] in referenced]
    return f"{query}. {' '.join(details)}"


def reference_genres(books, referenced):
    return {genre.casefold() for book in books if book["id"] in referenced
            for genre in book["genres"]}
