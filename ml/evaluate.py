import argparse
import csv
import json
import time
from pathlib import Path

from ml.bm25_retriever import BM25Retriever
from ml.dataset import load_books
from ml.metrics import ndcg_at_k, precision_at_k, preference_compliance
from ml.semantic_retriever import SemanticRetriever


def evaluate(retrievers, cases, catalog_ids):
    if not cases:
        raise ValueError("Evaluation file has no labeled queries")
    rows = []
    for method, retriever in retrievers.items():
        measures = {"ndcg_at_5": [], "precision_at_5": [], "preference_compliance": [], "response_time_ms": []}
        for case in cases:
            relevance = {item["book_id"]: item["relevance"] for item in case["relevant_books"]}
            if not relevance or not set(relevance) <= catalog_ids or any(value not in (0, 1, 2, 3) for value in relevance.values()):
                raise ValueError(f"Invalid relevance judgments: {case['query']}")
            filters = {key: case.get(key) for key in ("language", "genre", "author")}
            start = time.perf_counter()
            results = retriever.search(case["query"], k=5, **filters)
            measures["response_time_ms"].append((time.perf_counter() - start) * 1000)
            ids = [book["id"] for book, _ in results]
            measures["ndcg_at_5"].append(ndcg_at_k(ids, relevance))
            measures["precision_at_5"].append(precision_at_k(ids, relevance))
            compliance = preference_compliance([book for book, _ in results], **filters)
            if compliance is not None:
                measures["preference_compliance"].append(compliance)
        row = {"method": method, "queries": len(cases)}
        for key, values in measures.items():
            row[key] = round(sum(values) / len(values), 4) if values else None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/processed/books.json")
    parser.add_argument("--embeddings", default="data/processed/embeddings.npy")
    parser.add_argument("--cases", default="data/evaluation/queries.json")
    parser.add_argument("--output", default="results/evaluation.csv")
    args = parser.parse_args()
    books = load_books(args.catalog)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    rows = evaluate({"BM25": BM25Retriever(books), "Semantic": SemanticRetriever(books, args.embeddings)}, cases, {book["id"] for book in books})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Method | nDCG@5 | Precision@5 | Preference Compliance | Avg Response Time (ms)")
    for row in rows:
        print(f"{row['method']} | {row['ndcg_at_5']} | {row['precision_at_5']} | {row['preference_compliance']} | {row['response_time_ms']}")


if __name__ == "__main__":
    main()
