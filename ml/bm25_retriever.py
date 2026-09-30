import math
from collections import Counter

from ml.dataset import expand_reference_query, matches, reference_genres, referenced_book_ids, tokens


class BM25Retriever:
    def __init__(self, books, k1=1.5, b=0.75):
        self.books = books
        self.k1 = k1
        self.b = b
        self.documents = [Counter(tokens(book["searchable_text"])) for book in books]
        self.lengths = [sum(doc.values()) for doc in self.documents]
        self.average_length = sum(self.lengths) / len(books) if books else 0
        self.document_frequency = Counter(term for doc in self.documents for term in doc)

    def search(self, query, k=5, language=None, genre=None, author=None):
        if not query.strip():
            raise ValueError("Query cannot be empty")
        terms = set(tokens(expand_reference_query(query, self.books)))
        excluded = referenced_book_ids(query, self.books)
        shared_genres = reference_genres(self.books, excluded) if not genre else set()
        ranked = []
        for book, doc, length in zip(self.books, self.documents, self.lengths):
            if book["id"] in excluded or not matches(book, language, genre, author):
                continue
            if shared_genres and not any(item.casefold() in shared_genres for item in book["genres"]):
                continue
            score = 0.0
            for term in terms:
                frequency = doc[term]
                if frequency:
                    df = self.document_frequency[term]
                    idf = math.log(1 + (len(self.books) - df + 0.5) / (df + 0.5))
                    score += idf * frequency * (self.k1 + 1) / (frequency + self.k1 * (1 - self.b + self.b * length / self.average_length))
            if score > 0:
                ranked.append((book, score))
        ranked.sort(key=lambda pair: (-sum(item.casefold() in shared_genres for item in pair[0]["genres"]),
                                      -pair[1], pair[0]["id"]))
        return ranked[:k]
