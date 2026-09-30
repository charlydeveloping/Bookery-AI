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
    matched_titles = []
    for book in books:
        for title in [book["title"], *book.get("reference_aliases", [])]:
            if f" {folded_words(title)} " in query_words:
                excluded.add(book["id"])
                matched_titles.append(book)
                break
    if matched_titles:
        reference_authors = {book["author"] for book in matched_titles}
        excluded.update(book["id"] for book in books if book["author"] in reference_authors)
    return excluded


def expand_reference_query(query, books):
    """Use catalog facts about a named book to make similarity requests specific."""
    referenced = referenced_book_ids(query, books)
    if not referenced:
        return query
    reference_books = [book for book in books if book["id"] in referenced]
    named_book = find_title_in_query(query, reference_books)
    if named_book and not find_author_in_query(query, reference_books):
        reference_books = [named_book]
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
    named_book = find_title_in_query(query, reference_books)
    profile_books = reference_books if author or not named_book else [named_book]
    topics = list(dict.fromkeys(topic for book in profile_books for topic in book.get("reference_topics", [])))
    genres = sorted({genre for book in profile_books for genre in book["genres"]})
    return {"label": author or (named_book or reference_books[0])["title"], "book_ids": referenced,
            "topics": topics or genres, "genres": genres}


def infer_genre(query, books):
    """Apply explicit catalog categories and unambiguous Spanish synonyms."""
    words = f" {folded_words(query)} "
    synonyms = {"programacion": "Tecnología", "programar": "Tecnología",
                "cocinar": "Cocina", "recetas": "Cocina",
                "viajar": "Viajes", "viaje": "Viajes"}
    for term, genre in synonyms.items():
        if f" {term} " in words:
            return genre
    categories = sorted({genre for book in books for genre in book["genres"]}, key=len, reverse=True)
    for genre in categories:
        if len(folded_words(genre)) >= 5 and f" {folded_words(genre)} " in words:
            return genre
    return None


def availability_query(query):
    """Detect direct questions asking whether a specific catalog item is available."""
    query_words = f" {folded_words(query)} "
    if similarity_reference_query(query):
        return False
    return bool(re.search(r"\b(?:tienen|tienes|hay|venden|disponible|cuentan con)\b", query_words)) and bool(
        re.search(r"\b(?:el|la|este|esta)\s+(?:libro|novela|titulo)\b", query_words)
        or find_title_phrase(query))


def similarity_reference_query(query):
    """Detect a request for books similar to a named book or author."""
    query_words = f" {folded_words(query)} "
    markers = (" parecido a ", " parecida a ", " parecidos a ", " parecidas a ",
               " similar a ", " similares a ", " semejante a ", " semejantes a ")
    return any(marker in query_words for marker in markers)


def descriptive_similarity_query(query):
    """Allow comparisons to a genre or theme instead of assuming a named book."""
    reference = extract_similarity_reference(query)
    return bool(reference and re.match(r"^(?:una?\s+(?:novela|libro|historia|cuento)\s+de|historias\s+de)\b",
                                       folded_words(reference)))


def find_title_in_query(query, books):
    """Find the longest exact catalog title or alias mentioned in a query."""
    query_words = f" {folded_words(query)} "
    found = []
    for book in books:
        for title in [book["title"], *book.get("reference_aliases", [])]:
            folded_title = folded_words(title)
            if folded_title and f" {folded_title} " in query_words:
                found.append((len(folded_title), book))
    return max(found, key=lambda item: item[0])[1] if found else None


def find_author_in_query(query, books):
    query_words = f" {folded_words(query)} "
    authors = {book["author"] for book in books}
    found = [author for author in authors if f" {folded_words(author)} " in query_words]
    return max(found, key=len) if found else None


def find_title_phrase(query):
    """Extract a named title from common Spanish catalog questions."""
    normalized = normalize_text(query).strip(" \t\r\n?¿¡.,;:")
    patterns = (
        r"(?:tienen|tienes|hay|venden|esta disponible|conoces)\s+(?:el|la|este|esta)\s+(?:libro|novela|titulo)\s+(.+)$",
        r"(?:de que trata|quien escribio|quien es el autor de|informacion sobre|resumen de|sinopsis de|autor de)\s+(?:(?:el|la)\s+(?:libro|novela)\s+)?(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, folded_words(normalized), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def direct_book_query(query, books):
    """Return a known title or a named title whose details the user asks about."""
    if similarity_reference_query(query):
        return None
    query_words = f" {folded_words(query)} "
    asks_about_book = availability_query(query) or bool(re.search(
        r"\b(?:de que trata|quien escribio|quien es el autor de|informacion sobre|resumen de|sinopsis de|autor de)\b", query_words))
    book = find_title_in_query(query, books)
    if book and (asks_about_book or folded_words(query) == folded_words(book["title"])
                 or bool(re.search(r"\b(?:tienen|tienes|hay|venden|disponible)\b", query_words))):
        return {"book": book, "requested_title": book["title"]}
    requested_title = find_title_phrase(query)
    if requested_title and asks_about_book:
        return {"book": None, "requested_title": requested_title}
    return None


def extract_similarity_reference(query):
    """Extract text after a similarity phrase, excluding the question punctuation."""
    normalized = normalize_text(query)
    match = re.search(r"\b(?:parecid[oa]s?|similar(?:es)?|semejant(?:e|es))\s+a\s+(.+?)[?.!,;]*$",
                      normalized, flags=re.IGNORECASE)
    if not match:
        return None
    reference = match.group(1).strip(" \t\r\n?¿¡.,;:")
    reference = re.sub(r"^(?:los\s+libros\s+de|las\s+obras\s+de|el\s+libro|la\s+novela|los\s+libros)\s+",
                       "", reference, flags=re.IGNORECASE)
    return reference or None
