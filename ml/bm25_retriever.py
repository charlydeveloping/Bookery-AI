import math
from collections import Counter

from ml.dataset import matches, tokens


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
        terms = set(tokens(query))
        ranked = []
        for book, doc, length in zip(self.books, self.documents, self.lengths):
            if not matches(book, language, genre, author):
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
        ranked.sort(key=lambda pair: (-pair[1], pair[0]["id"]))
        return ranked[:k]
