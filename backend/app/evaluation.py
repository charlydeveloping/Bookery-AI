"""Collect human relevance judgments for the fixed evaluation queries."""

import csv
import json
import os
from pathlib import Path
from threading import Lock

from fastapi import HTTPException

from ml.evaluate import evaluate


class EvaluationStore:
    def __init__(self, cases_path, judgments_path, results_path):
        self.cases_path = Path(cases_path)
        self.judgments_path = Path(judgments_path)
        self.results_path = Path(results_path)
        self.lock = Lock()

    def cases(self):
        return json.loads(self.cases_path.read_text(encoding="utf-8"))

    def saved(self):
        if not self.judgments_path.exists():
            return []
        return json.loads(self.judgments_path.read_text(encoding="utf-8"))

    def case(self, case_id):
        cases = self.cases()
        if case_id < 1 or case_id > len(cases):
            raise HTTPException(status_code=404, detail="Consulta de evaluación inexistente")
        return cases[case_id - 1]

    def candidates(self, case, bm25, semantic):
        filters = {key: case.get(key) for key in ("language", "genre", "author")}
        results = (bm25.search(case["query"], k=5, **filters),
                   semantic.search(case["query"], k=5, **filters))
        books = {book["id"]: book for ranking in results for book, _ in ranking}
        return sorted(books.values(), key=lambda book: (book["title"].casefold(), book["id"]))

    def overview(self):
        saved = {item["query"] for item in self.saved()}
        cases = self.cases()
        return {"total": len(cases), "completed": len(saved & {case["query"] for case in cases}),
                "cases": [{"id": index, "query": case["query"], "completed": case["query"] in saved}
                          for index, case in enumerate(cases, start=1)]}

    def detail(self, case_id, bm25, semantic):
        case = self.case(case_id)
        saved = next((item for item in self.saved() if item["query"] == case["query"]), None)
        ratings = {item["book_id"]: item["relevance"] for item in saved["relevant_books"]} if saved else {}
        books = self.candidates(case, bm25, semantic)
        return {"id": case_id, "query": case["query"], "completed": saved is not None,
                "books": [{"id": book["id"], "title": book["title"], "author": book["author"],
                           "genres": book["genres"], "description": book["description"],
                           "source_url": book.get("source_url"), "relevance": ratings.get(book["id"])}
                          for book in books]}

    def save(self, case_id, judgments, bm25, semantic):
        case = self.case(case_id)
        candidates = self.candidates(case, bm25, semantic)
        expected = {book["id"] for book in candidates}
        received = [item.book_id for item in judgments]
        if len(received) != len(expected) or set(received) != expected:
            raise HTTPException(status_code=422, detail="Califica todos los libros de esta consulta una sola vez")
        reviewed = {key: case[key] for key in ("query", "language", "genre", "author") if key in case}
        reviewed["relevant_books"] = [{"book_id": book["id"],
                                       "relevance": next(item.relevance for item in judgments if item.book_id == book["id"])}
                                      for book in candidates]
        with self.lock:
            saved = {item["query"]: item for item in self.saved()}
            saved[case["query"]] = reviewed
            ordered = [saved[item["query"]] for item in self.cases() if item["query"] in saved]
            self._write_json(self.judgments_path, ordered)
        return self.overview()

    def calculate(self, bm25, semantic, catalog_ids):
        cases = self.cases()
        saved = {item["query"]: item for item in self.saved()}
        if len(saved) != len(cases) or any(case["query"] not in saved for case in cases):
            raise HTTPException(status_code=409, detail="Primero califica todas las consultas")
        for case in cases:
            expected = {book["id"] for book in self.candidates(case, bm25, semantic)}
            judged = {item["book_id"] for item in saved[case["query"]]["relevant_books"]}
            if expected != judged:
                raise HTTPException(status_code=409, detail="Cambió el catálogo o el modelo; revisa las calificaciones")
        rows = evaluate({"BM25": bm25, "Semantic": semantic},
                        [saved[case["query"]] for case in cases], catalog_ids)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return {"results": rows, "judgments_file": str(self.judgments_path),
                "results_file": str(self.results_path)}

    @staticmethod
    def _write_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
