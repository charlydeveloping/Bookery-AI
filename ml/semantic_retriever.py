import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.dataset import expand_reference_query, load_books, matches, reference_genres, referenced_book_ids

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def catalog_hash(books):
    payload = json.dumps([(book["id"], book["searchable_text"]) for book in books], ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def build_embeddings(books, output, model_name=MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, revision=MODEL_REVISION)
    vectors = model.encode([book["searchable_text"] for book in books], normalize_embeddings=True, show_progress_bar=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, vectors)
    output.with_suffix(".json").write_text(json.dumps({"catalog_hash": catalog_hash(books), "model": model_name, "revision": MODEL_REVISION,
        "dimension": int(vectors.shape[1]), "count": len(books)}, indent=2), encoding="utf-8")


class SemanticRetriever:
    def __init__(self, books, embedding_path, model_name=MODEL_NAME, model=None):
        self.books = books
        self.model_name = model_name
        path = Path(embedding_path)
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata["catalog_hash"] != catalog_hash(books) or metadata["model"] != model_name or metadata["revision"] != MODEL_REVISION:
            raise ValueError("Embeddings do not match catalog/model; regenerate them")
        self.vectors = np.load(path)
        if self.vectors.shape != (len(books), metadata["dimension"]):
            raise ValueError("Invalid embedding matrix shape")
        self.model = model or self._load_model()

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.model_name, revision=MODEL_REVISION)

    def search(self, query, k=5, language=None, genre=None, author=None):
        if not query.strip():
            raise ValueError("Query cannot be empty")
        vector = np.asarray(self.model.encode([expand_reference_query(query, self.books)], normalize_embeddings=True)[0])
        scores = np.einsum("ij,j->i", self.vectors, vector)
        excluded = referenced_book_ids(query, self.books)
        shared_genres = reference_genres(self.books, excluded) if not genre else set()
        ranked = [(book, float(scores[index])) for index, book in enumerate(self.books)
                  if book["id"] not in excluded and matches(book, language, genre, author)
                  and (not shared_genres or any(item.casefold() in shared_genres for item in book["genres"]))]
        ranked.sort(key=lambda pair: (-sum(item.casefold() in shared_genres for item in pair[0]["genres"]),
                                      -pair[1], pair[0]["id"]))
        return ranked[:k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/processed/books.json")
    parser.add_argument("--output", default="data/processed/embeddings.npy")
    args = parser.parse_args()
    build_embeddings(load_books(args.catalog), args.output)
