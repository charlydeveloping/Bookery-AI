from fastapi.testclient import TestClient
import httpx

from backend.app.main import create_app
from backend.app.llm import LLMExplainer
from ml.dataset import referenced_book_ids
from tests.test_core import RAW
from ml.prepare_dataset import clean


class FakeSemantic:
    def __init__(self, books):
        self.books = books

    def search(self, query, k=5, language=None, genre=None, author=None):
        from ml.dataset import matches, reference_genres
        excluded = referenced_book_ids(query, self.books)
        shared = reference_genres(self.books, excluded) if not genre else set()
        return [(book, 0.8) for book in self.books
                if book["id"] not in excluded and matches(book, language, genre, author)
                and (not shared or any(value.casefold() in shared for value in book["genres"]))][:k]


def test_api_end_to_end_contract():
    books = clean(RAW)
    with TestClient(create_app(books, FakeSemantic(books))) as client:
        response = client.post("/api/recommend", json={"query": "Quiero ciencia ficción", "genre": "Ciencia", "limit": 5})
        assert response.status_code == 200
        payload = response.json()
        assert [item["id"] for item in payload["recommendations"]] == ["a"]
        assert client.get("/api/options").json()["languages"] == ["es"]
        assert client.post("/api/recommend", json={"query": "   "}).status_code == 422
        assert client.post("/api/recommend", json={"query": "books", "limit": 0}).status_code == 422
        assert client.post("/api/search/bm25", json={"query": "cámaras"}).json()["recommendations"][0]["id"] == "a"


def test_availability_query_answers_from_catalog_instead_of_recommending_unrelated_books():
    books = clean(RAW)
    with TestClient(create_app(books, FakeSemantic(books))) as client:
        payload = client.post("/api/recommend", json={"query": "¿Tienen el libro Pueblo enfermo?", "language": "es"}).json()
        assert payload["recommendations"] == []
        assert payload["explanation_mode"] == "catalog"
        assert "No encontré «pueblo enfermo»" in payload["answer"]

        known_title = books[0]["title"]
        payload = client.post("/api/recommend", json={"query": f"¿Tienen el libro {known_title}?", "language": "es"}).json()
        assert [item["id"] for item in payload["recommendations"]] == [books[0]["id"]]
        assert "está en el catálogo académico" in payload["answer"]


def test_direct_book_questions_and_author_listing_use_catalog_evidence():
    books = clean(RAW)

    class UnexpectedExplainer:
        def explain(self, _query, _books):
            raise AssertionError("Catalog questions must not be sent to the LLM")

    with TestClient(create_app(books, FakeSemantic(books), UnexpectedExplainer())) as client:
        known = client.post("/api/recommend", json={"query": "¿De qué trata Ciudad futura?"}).json()
        assert [item["id"] for item in known["recommendations"]] == ["a"]
        assert books[0]["description"] in known["answer"]
        missing = client.post("/api/recommend", json={"query": "¿Quién escribió Pueblo enfermo?"}).json()
        assert missing["recommendations"] == []
        assert "No encontré «pueblo enfermo»" in missing["answer"]
        short = client.post("/api/recommend", json={"query": "¿Tienen Ciudad futura?"}).json()
        assert [item["id"] for item in short["recommendations"]] == ["a"]
        author = client.post("/api/recommend", json={"query": "¿Qué libros de A. Escritora tienes?"}).json()
        assert [item["id"] for item in author["recommendations"]] == ["a"]
        assert author["explanation_mode"] == "catalog"


def test_generic_book_request_still_uses_retriever():
    books = clean(RAW)
    with TestClient(create_app(books, FakeSemantic(books))) as client:
        payload = client.post("/api/recommend", json={"query": "¿Tienes algún libro de ciencia ficción?"}).json()
        assert payload["recommendations"]
        assert payload["explanation_mode"] == "basic"
        descriptive = client.post("/api/recommend", json={"query": "Algo similar a una novela de ciencia ficción"}).json()
        assert descriptive["recommendations"]


def test_unknown_similarity_reference_does_not_return_accidental_matches_or_call_llm():
    books = clean(RAW)

    class UnexpectedExplainer:
        def explain(self, _query, _books):
            raise AssertionError("Unknown reference queries must not be sent to the LLM")

    with TestClient(create_app(books, FakeSemantic(books), UnexpectedExplainer())) as client:
        payload = client.post("/api/recommend", json={
            "query": "tienes algun libro similar a pueblo enfermo?", "language": "es"
        }).json()
        assert payload["recommendations"] == []
        assert payload["explanation_mode"] == "catalog"
        assert "no puedo determinar" in payload["answer"]
        assert "pueblo enfermo" in payload["answer"]


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
        payload = client.post("/api/recommend", json={"query": "ciencia ficción", "genre": "Ciencia"}).json()
        assert payload["explanation_mode"] == "llm"
        assert [item["id"] for item in payload["recommendations"]] == ["a"]
        assert payload["recommendations"][0]["reason"].startswith("Explora")
        baseline = client.post("/api/search/bm25", json={"query": "cámaras"}).json()
        assert baseline["explanation_mode"] == "basic"


def test_llm_rejects_unknown_book_and_uses_basic_explanation():
    books = clean(RAW)

    def reply(_request):
        return httpx.Response(200, json={"choices": [{"message": {"content":
            '{"reasons":[{"id":"invented","reason":"Una recomendación falsa."}]}'}}]})

    llm = LLMExplainer("https://example.test/v1", "test-model",
                       client=httpx.Client(transport=httpx.MockTransport(reply)))
    with TestClient(create_app(books, FakeSemantic(books), llm)) as client:
        payload = client.post("/api/recommend", json={"query": "ciencia ficción", "genre": "Ciencia"}).json()
        assert payload["explanation_mode"] == "basic"
        assert [item["id"] for item in payload["recommendations"]] == ["a"]
        assert "similitud semántica" in payload["recommendations"][0]["reason"]


def test_author_similarity_uses_catalog_topics_and_skips_llm():
    books = clean(RAW)
    books[0]["genres"] = ["Psicología", "Crecimiento personal"]
    books[0]["title"] = "Inteligencia emocional"
    books[0]["author"] = "Daniel Goleman"
    books[0]["searchable_text"] = "Inteligencia emocional. Daniel Goleman. Psicología. Gestión emocional."
    books.append({"id": "mre-1", "title": "Recupera tu mente, reconquista tu vida",
                  "author": "Marian Rojas Estapé", "description": "Una guía sobre atención, emociones y bienestar personal.",
                  "genres": ["Psicología", "Gestión emocional", "Crecimiento personal"],
                  "reference_topics": ["atención", "gestión emocional", "bienestar"], "language": "es",
                  "searchable_text": "Recupera tu mente. atención gestión emocional bienestar."})

    class UnexpectedExplainer:
        def explain(self, _query, _books):
            raise AssertionError("Author reference explanations should use catalog evidence")

    with TestClient(create_app(books, FakeSemantic(books), UnexpectedExplainer())) as client:
        payload = client.post("/api/recommend", json={
            "query": "Quiero leer algo parecido a los libros de Marian Rojas Estape", "language": "es"
        }).json()
        assert [item["id"] for item in payload["recommendations"]] == ["a"]
        assert payload["explanation_mode"] == "reference"
        assert "Psicología" in payload["recommendations"][0]["reason"]
        assert "Marian Rojas Estapé" in payload["answer"]
