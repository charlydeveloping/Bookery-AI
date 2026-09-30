import json

from fastapi.testclient import TestClient

from backend.app.evaluation import EvaluationStore
from backend.app.main import create_app
from ml.prepare_dataset import clean
from tests.test_core import RAW


class FakeSemantic:
    def __init__(self, books):
        self.books = books

    def search(self, query, k=5, language=None, genre=None, author=None):
        return [(book, 0.8) for book in reversed(self.books)][:k]


def test_human_evaluation_saves_zero_scores_and_calculates_only_when_complete(tmp_path):
    cases_path = tmp_path / "queries.json"
    cases_path.write_text(json.dumps([{"query": "ciudad huerto", "language": "es"},
                                      {"query": "cámaras", "language": "es"}]))
    judgments_path = tmp_path / "reviewed_queries.json"
    results_path = tmp_path / "evaluation_reviewed.csv"
    books = clean(RAW)
    store = EvaluationStore(cases_path, judgments_path, results_path)

    with TestClient(create_app(books, FakeSemantic(books), evaluation_store=store)) as client:
        overview = client.get("/api/evaluation/cases").json()
        assert overview["completed"] == 0
        first = client.get("/api/evaluation/cases/1").json()
        assert [book["id"] for book in first["books"]] == ["a", "b"]
        assert all(book["relevance"] is None for book in first["books"])
        assert client.post("/api/evaluation/calculate").status_code == 409
        assert client.post("/api/evaluation/cases/1/judgments", json={"judgments": [
            {"book_id": "a", "relevance": 3}]}).status_code == 422
        assert client.post("/api/evaluation/cases/1/judgments", json={"judgments": [
            {"book_id": "a", "relevance": 4}, {"book_id": "b", "relevance": 0}]}).status_code == 422

        first_scores = [{"book_id": "a", "relevance": 3}, {"book_id": "b", "relevance": 0}]
        assert client.post("/api/evaluation/cases/1/judgments", json={"judgments": first_scores}).json()["completed"] == 1
        assert client.get("/api/evaluation/cases/1").json()["books"][1]["relevance"] == 0
        assert client.post("/api/evaluation/cases/2/judgments", json={"judgments": first_scores}).json()["completed"] == 2
        result = client.post("/api/evaluation/calculate").json()
        assert [row["method"] for row in result["results"]] == ["BM25", "Semantic"]
        assert len(json.loads(judgments_path.read_text())) == 2
        assert "ndcg_at_5" in results_path.read_text()
