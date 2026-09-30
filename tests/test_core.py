import json

import numpy as np
import pytest

from ml.bm25_retriever import BM25Retriever
from ml.build_curated_catalog import build_catalog
from ml.expand_popular_catalog import build_entry, collect, spanish_edition
from ml.dataset import infer_genre, load_books, referenced_book_ids
from ml.metrics import ndcg_at_k, precision_at_k, preference_compliance
from ml.prepare_dataset import clean
from ml.semantic_retriever import MODEL_NAME, MODEL_REVISION, SemanticRetriever, catalog_hash


RAW = [
    {"id": "a", "volumeInfo": {"title": "Ciudad futura", "authors": ["A. Escritora"], "categories": ["Ciencia ficción"], "language": "es", "description": "Una sociedad está gobernada por cámaras y decisiones automatizadas. Una mujer lucha por recuperar su libertad frente a la maquinaria del control social."}},
    {"id": "b", "volumeInfo": {"title": "El huerto", "authors": ["B. Escritor"], "categories": ["Drama familiar"], "language": "es", "description": "Una familia vuelve a un huerto rural tras muchos años de separación. Sus miembros reconstruyen sus relaciones y recuerdan el pasado compartido."}},
]


def test_clean_and_valid_bm25_case():
    books = clean(RAW)
    assert len(books) == 2
    assert BM25Retriever(books).search("cámaras decisiones automatizadas")[0][0]["id"] == "a"


def test_difficult_semantic_query_has_no_forced_bm25_success():
    books = clean(RAW)
    results = BM25Retriever(books).search("vigilancia digital")
    assert results == []


def test_semantic_retrieval_handles_vocabulary_difference(tmp_path):
    books = clean(RAW)
    path = tmp_path / "embeddings.npy"
    np.save(path, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    path.with_suffix(".json").write_text(json.dumps({"catalog_hash": catalog_hash(books),
        "model": MODEL_NAME, "revision": MODEL_REVISION, "dimension": 2, "count": 2}))

    class FakeModel:
        def encode(self, queries, normalize_embeddings=True):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    results = SemanticRetriever(books, path, model=FakeModel()).search("vigilancia digital")
    assert results[0][0]["id"] == "a"


def test_invalid_query_and_filters():
    books = clean(RAW)
    with pytest.raises(ValueError):
        BM25Retriever(books).search("  ")
    assert BM25Retriever(books).search("ciudad", language="en") == []
    assert preference_compliance([books[0]], genre="Ciencia") == 1.0


def test_spanish_catalog_excludes_named_reference():
    annotations = [{"id": "a", "title": "Harry Potter y la piedra filosofal",
                    "reference_aliases": ["Harry Potter"], "genres": ["Fantasía"],
                    "description": "Un estudiante descubre la magia y encuentra amistades nuevas mientras investiga un misterio en una escuela de hechicería."},
                   {"id": "b", "title": "El hobbit", "genres": ["Fantasía"],
                    "description": "Un viajero abandona su hogar y participa en una aventura fantástica junto a un grupo que busca un tesoro custodiado por un dragón."}]
    sources = [{"id": key, "author": "Autor " + key, "edition_id": key + "M",
                "edition_language": "spa", "source_url": "https://openlibrary.org/books/" + key + "M"}
               for key in ("a", "b")]
    books = build_catalog(annotations, sources)
    assert all(book["language"] == "es" for book in books)
    assert referenced_book_ids("Algo parecido a Harry Potter", books) == {"a"}
    assert [book["id"] for book, _ in BM25Retriever(books).search("Parecido a Harry Potter con fantasía")] == ["b"]


def test_metrics():
    assert ndcg_at_k(["a", "b"], {"a": 3, "b": 2}) == 1
    assert precision_at_k(["a", "b"], {"a": 3, "b": 2}) == 0.4


def test_popular_import_requires_verified_spanish_edition_and_deduplicates():
    doc = {"key": "/works/OL999W", "title": "Sample Work", "author_name": ["Autor Ejemplo"],
           "readinglog_count": 100, "subject": ["Magic", "Adventure"],
           "editions": {"docs": [
               {"key": "/books/OL1M", "title": "English title", "language": ["eng"]},
               {"key": "/books/OL2M", "title": "Título español", "language": ["spa"]}]}}
    assert spanish_edition(doc) == ("OL2M", "Título español")
    annotation, source = build_entry(doc, "fantasy", "Fantasía")
    assert annotation["title"] == "Título español"
    assert "Fantasía" in annotation["genres"]
    assert source["edition_language"] == "spa"
    assert build_entry({**doc, "editions": {"docs": [{"key": "/books/OL1M",
                        "title": "English title", "language": ["eng"]}]}}, "fantasy", "Fantasía") is None

    def fetch(_url):
        return {"docs": [doc]}

    annotations, sources, counts = collect(set(), quota=1, fetch=fetch, pause=0)
    assert len(annotations) == len(sources) == 1
    assert sum(counts.values()) == 1


def test_expanded_catalog_covers_categories_and_excludes_same_author_references():
    books = load_books("data/processed/books.json")
    assert len(books) >= 200
    assert all(book["language"] == "es" and book.get("source_url") for book in books)
    genres = {genre for book in books for genre in book["genres"]}
    assert {"Fantasía", "Ciencia ficción", "Misterio", "Romance", "Terror",
            "Poesía", "Cocina", "Tecnología", "Viajes", "Cómic"} <= genres
    assert infer_genre("Quiero aprender programación", books) == "Tecnología"
    excluded = referenced_book_ids("Algo parecido a Harry Potter", books)
    assert excluded
    assert all(book["id"] in excluded for book in books if book["author"] == "J. K. Rowling")
