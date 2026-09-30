import math

from ml.dataset import matches


def ndcg_at_k(ids, relevance, k=5):
    def dcg(values):
        return sum((2 ** value - 1) / math.log2(index + 2) for index, value in enumerate(values))
    observed = dcg([relevance.get(book_id, 0) for book_id in ids[:k]])
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return observed / ideal if ideal else 0.0


def precision_at_k(ids, relevance, k=5):
    return sum(relevance.get(book_id, 0) >= 2 for book_id in ids[:k]) / k


def preference_compliance(books, language=None, genre=None, author=None):
    if not any((language, genre, author)):
        return None
    return sum(matches(book, language, genre, author) for book in books) / len(books) if books else 1.0
