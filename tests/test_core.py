import json

import numpy as np
import pytest

from ml.bm25_retriever import BM25Retriever
from ml.metrics import ndcg_at_k, precision_at_k, preference_compliance
from ml.prepare_dataset import clean
from ml.semantic_retriever import MODEL_NAME, MODEL_REVISION, SemanticRetriever, catalog_hash


RAW = [
    {"id": "a", "volumeInfo": {"title": "Future City", "authors": ["A. Writer"], "categories": ["Science fiction"], "language": "en", "description": "A society is governed by pervasive cameras and automated decisions. A woman struggles to recover her freedom from the machinery of social control."}},
    {"id": "b", "volumeInfo": {"title": "The Orchard", "authors": ["B. Writer"], "categories": ["Literary fiction"], "language": "en", "description": "A family returns to a rural orchard after many years apart. They rebuild their relationships across the changing seasons and remember their shared past."}},
]


def test_clean_and_valid_bm25_case():
    books = clean(RAW)
    assert len(books) == 2
    assert BM25Retriever(books).search("cameras automated decisions")[0][0]["id"] == "a"


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
    assert BM25Retriever(books).search("city", language="es") == []
    assert preference_compliance([books[0]], genre="Science") == 1.0


def test_metrics():
    assert ndcg_at_k(["a", "b"], {"a": 3, "b": 2}) == 1
    assert precision_at_k(["a", "b"], {"a": 3, "b": 2}) == 0.4
