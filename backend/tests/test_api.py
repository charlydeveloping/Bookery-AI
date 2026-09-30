from fastapi.testclient import TestClient
import httpx

from backend.app.main import create_app
from backend.app.llm import LLMExplainer
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


def test_llm_explains_only_retrieved_books():
    books = clean(RAW)

    def reply(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"choices": [{"message": {"content":
            '{"reasons":[{"id":"a","reason":"Explora una idea científica presente en la descripción."}]}'}}]})

    llm = LLMExplainer("https://example.test/v1", "test-model", "test-key",
                       client=httpx.Client(transport=httpx.MockTransport(reply)))
    with TestClient(create_app(books, FakeSemantic(books), llm)) as client:
        payload = client.post("/api/recommend", json={"query": "ciencia ficción", "genre": "Science"}).json()
        assert payload["explanation_mode"] == "llm"
        assert [item["id"] for item in payload["recommendations"]] == ["a"]
        assert payload["recommendations"][0]["reason"].startswith("Explora")
        baseline = client.post("/api/search/bm25", json={"query": "cameras"}).json()
        assert baseline["explanation_mode"] == "basic"


def test_llm_rejects_unknown_book_and_uses_basic_explanation():
    books = clean(RAW)

    def reply(_request):
        return httpx.Response(200, json={"choices": [{"message": {"content":
            '{"reasons":[{"id":"invented","reason":"Una recomendación falsa."}]}'}}]})

    llm = LLMExplainer("https://example.test/v1", "test-model",
                       client=httpx.Client(transport=httpx.MockTransport(reply)))
    with TestClient(create_app(books, FakeSemantic(books), llm)) as client:
        payload = client.post("/api/recommend", json={"query": "ciencia ficción", "genre": "Science"}).json()
        assert payload["explanation_mode"] == "basic"
        assert [item["id"] for item in payload["recommendations"]] == ["a"]
        assert "similitud semántica" in payload["recommendations"][0]["reason"]
