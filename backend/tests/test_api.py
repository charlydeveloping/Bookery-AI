from fastapi.testclient import TestClient

from backend.app.main import create_app
from tests.test_core import RAW
from ml.prepare_dataset import clean


class FakeSemantic:
    def __init__(self, books):
        self.books = books

    def search(self, query, k=5, language=None, genre=None, author=None):
        from ml.dataset import matches
        return [(book, 0.8) for book in self.books if matches(book, language, genre, author)][:k]


def test_api_end_to_end_contract():
    books = clean(RAW)
    with TestClient(create_app(books, FakeSemantic(books))) as client:
        response = client.post("/api/recommend", json={"query": "Quiero ciencia ficción", "genre": "Science", "limit": 5})
        assert response.status_code == 200
        payload = response.json()
        assert [item["id"] for item in payload["recommendations"]] == ["a"]
        assert client.get("/api/options").json()["languages"] == ["en"]
        assert client.post("/api/recommend", json={"query": "   "}).status_code == 422
        assert client.post("/api/recommend", json={"query": "books", "limit": 0}).status_code == 422
        assert client.post("/api/search/bm25", json={"query": "cameras"}).json()["recommendations"][0]["id"] == "a"
