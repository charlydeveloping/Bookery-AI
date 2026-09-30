import json
import re
import unicodedata
from pathlib import Path


def normalize_text(value):
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def tokens(value):
    return re.findall(r"\w+", normalize_text(value).casefold(), re.UNICODE)


def folded_words(value):
    plain = "".join(character for character in unicodedata.normalize("NFKD", normalize_text(value).casefold())
                    if not unicodedata.combining(character))
    return " ".join(tokens(plain))


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
    """Resolve a named author or title when the query asks for similar books."""
    query_words = f" {folded_words(query)} "
    markers = (" parecido a ", " parecida a ", " parecidos a ", " similares a ",
               " similar a ", " como ", " tipo ", " libros de ", " obras de ", " estilo de ")
    if not any(marker in query_words for marker in markers):
        return set()
    excluded = set()
    authors = {book["author"] for book in books}
    for author in authors:
        author_phrase = f" {folded_words(author)} "
        if author_phrase in query_words:
            excluded.update(book["id"] for book in books if book["author"] == author)
    for book in books:
        for title in [book["title"], *book.get("reference_aliases", [])]:
            if f" {folded_words(title)} " in query_words:
                excluded.add(book["id"])
                break
    return excluded


def expand_reference_query(query, books):
    """Use catalog facts about a named book to make similarity requests specific."""
    referenced = referenced_book_ids(query, books)
    if not referenced:
        return query
    reference_books = [book for book in books if book["id"] in referenced]
    profile_topics = list(dict.fromkeys(topic for book in reference_books for topic in book.get("reference_topics", [])))
    if profile_topics:
        details = [f"Temas de referencia: {', '.join(profile_topics)}."]
    else:
        details = [f"{', '.join(book['genres'])}. {book['description']}" for book in reference_books]
    return f"{query}. {' '.join(details)}"


def reference_genres(books, referenced):
    return {genre.casefold() for book in books if book["id"] in referenced
            for genre in book["genres"]}


def reference_context(query, books):
    referenced = referenced_book_ids(query, books)
    if not referenced:
        return None
    reference_books = [book for book in books if book["id"] in referenced]
    authors = {book["author"] for book in reference_books}
    query_words = f" {folded_words(query)} "
    author = next((name for name in authors if f" {folded_words(name)} " in query_words), None)
    topics = list(dict.fromkeys(topic for book in reference_books for topic in book.get("reference_topics", [])))
    genres = sorted({genre for book in reference_books for genre in book["genres"]})
    return {"label": author or reference_books[0]["title"], "book_ids": referenced,
            "topics": topics or genres, "genres": genres}
