import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StrictInt, field_validator
from dotenv import load_dotenv

from ml.bm25_retriever import BM25Retriever
from ml.dataset import (descriptive_similarity_query, direct_book_query, extract_similarity_reference, find_author_in_query,
                        infer_genre, load_books, matches, reference_context, similarity_reference_query)
from ml.semantic_retriever import SemanticRetriever
from backend.app.llm import LLMExplainer
from backend.app.jev import JevSelector
from backend.app.evaluation import EvaluationStore

load_dotenv()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    language: str | None = None
    genre: str | None = None
    author: str | None = None
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("La consulta no puede estar vacía")
        return value.strip()


class Judgment(BaseModel):
    book_id: str
    relevance: StrictInt = Field(ge=0, le=3)


class JudgmentRequest(BaseModel):
    judgments: list[Judgment]


def create_app(books=None, semantic=None, explainer=None, jev=None, evaluation_store=None):
    @asynccontextmanager
    async def lifespan(app):
        if not hasattr(app.state, "books"):
            app.state.books = load_books(os.getenv("BOOKERY_CATALOG", "data/processed/books.json"))
            app.state.bm25 = BM25Retriever(app.state.books)
            app.state.semantic = SemanticRetriever(app.state.books, os.getenv("BOOKERY_EMBEDDINGS", "data/processed/embeddings.npy"))
            app.state.explainer = LLMExplainer.from_env()
            app.state.jev = JevSelector.from_env()
        yield

    app = FastAPI(title="Bookery AI", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_methods=["GET", "POST"], allow_headers=["*"])
    app.state.evaluation = evaluation_store or EvaluationStore(
        os.getenv("BOOKERY_EVALUATION_CASES", "data/evaluation/queries.json"),
        os.getenv("BOOKERY_REVIEWED_CASES", "data/evaluation/reviewed_queries.json"),
        os.getenv("BOOKERY_REVIEWED_RESULTS", "results/evaluation_reviewed.csv"))
    if books is not None:
        app.state.books = books
        app.state.bm25 = BM25Retriever(books)
        app.state.semantic = semantic
        app.state.explainer = explainer
        app.state.jev = jev

    @app.get("/health")
    def health():
        return {"status": "ok", "catalog_size": len(app.state.books)}

    @app.get("/api/options")
    def options():
        return {"languages": sorted({book["language"] for book in app.state.books}),
                "genres": sorted({genre for book in app.state.books for genre in book["genres"]})}

    @app.get("/api/evaluation/cases")
    def evaluation_cases():
        return app.state.evaluation.overview()

    @app.get("/api/evaluation/cases/{case_id}")
    def evaluation_case(case_id: int):
        return app.state.evaluation.detail(case_id, app.state.bm25, app.state.semantic)

    @app.post("/api/evaluation/cases/{case_id}/judgments")
    def save_judgments(case_id: int, request: JudgmentRequest):
        return app.state.evaluation.save(case_id, request.judgments, app.state.bm25, app.state.semantic)

    @app.post("/api/evaluation/calculate")
    def calculate_evaluation():
        return app.state.evaluation.calculate(app.state.bm25, app.state.semantic,
                                              {book["id"] for book in app.state.books})

    def run(request, method):
        start = time.perf_counter()
        use_jev = method == "semantic_jev"
        direct = direct_book_query(request.query, app.state.books)
        if direct:
            book = direct["book"]
            recommendations = []
            if book and matches(book, request.language, request.genre, request.author):
                recommendations = [{"id": book["id"], "title": book["title"], "author": book["author"],
                    "genres": book["genres"], "language": book["language"], "description": book["description"],
                    "score": 1.0, "source_url": book.get("source_url"),
                    "reason": "Ficha registrada en el catálogo académico."}]
                answer = (f"«{book['title']}», de {book['author']}, está en el catálogo académico. "
                          f"Según su ficha: {book['description']}")
            elif book:
                answer = f"«{book['title']}» está en el catálogo, pero no cumple los filtros seleccionados."
            else:
                answer = (f"No encontré «{direct['requested_title']}» en el catálogo académico. "
                          "No puedo confirmar su contenido ni la disponibilidad en la librería.")
            return {"query": request.query, "method": method, "recommendations": recommendations,
                    "answer": answer, "explanation_mode": "catalog",
                    "response_time_ms": round((time.perf_counter() - start) * 1000, 2)}
        if similarity_reference_query(request.query):
            reference = reference_context(request.query, app.state.books)
            if reference is None and not descriptive_similarity_query(request.query):
                named_reference = extract_similarity_reference(request.query) or "esa obra"
                return {"query": request.query, "method": method, "recommendations": [],
                        "answer": (f"No tengo información sobre «{named_reference}» en la base de conocimiento, "
                                   "así que no puedo determinar qué libros del catálogo son similares. "
                                   "Prueba con un título conocido o describe los temas que buscas."),
                        "explanation_mode": "catalog",
                        "response_time_ms": round((time.perf_counter() - start) * 1000, 2)}
        author = find_author_in_query(request.query, app.state.books)
        if author and not similarity_reference_query(request.query) and any(
                phrase in f" {request.query.casefold()} " for phrase in ("libros de", "obras de", "escritos por")):
            author_books = [book for book in app.state.books if book["author"] == author
                            and matches(book, request.language, request.genre, request.author)][:request.limit]
            recommendations = [{"id": book["id"], "title": book["title"], "author": book["author"],
                "genres": book["genres"], "language": book["language"], "description": book["description"],
                "score": 1.0, "source_url": book.get("source_url"),
                "reason": "Obra de este autor registrada en el catálogo académico."} for book in author_books]
            answer = (f"Encontré {len(recommendations)} libro(s) de {author} en el catálogo académico."
                      if recommendations else f"No encontré libros de {author} con los filtros seleccionados.")
            return {"query": request.query, "method": method, "recommendations": recommendations,
                    "answer": answer, "explanation_mode": "catalog",
                    "response_time_ms": round((time.perf_counter() - start) * 1000, 2)}
        retriever = app.state.semantic if method != "bm25" else app.state.bm25
        if retriever is None:
            raise HTTPException(status_code=503, detail="Índice semántico no disponible")
        reference = reference_context(request.query, app.state.books)
        genre = request.genre or (None if reference else infer_genre(request.query, app.state.books))
        results = retriever.search(request.query, k=request.limit, language=request.language, genre=genre, author=request.author)
        selection_mode = "semantic"
        selection_confidence = None
        if use_jev and results and app.state.jev:
            decision = app.state.jev.choose(request.query, [book for book, _ in results])
            if decision and decision["choice"] != "ninguno" and decision["confidence"] >= 0.35:
                chosen_id = decision["choice"]
                results = sorted(results, key=lambda pair: (pair[0]["id"] != chosen_id,
                    -decision["probabilities"][pair[0]["id"]]))
                selection_mode = "jev"
                selection_confidence = decision["confidence"]
        books = [book for book, _ in results]
        described_books = [book for book in books if "sin sinopsis argumental" not in book.get("description_origin", "")]
        explanations = (app.state.explainer.explain(request.query, described_books)
                        if method != "bm25" and app.state.explainer and described_books and not reference else None)
        recommendations = []
        for book, score in results:
            shared_topics = sorted(set(book["genres"]) & set(reference["genres"])) if reference else []
            if reference and shared_topics:
                reason = (f"Comparte las categorías {', '.join(shared_topics)} registradas para los libros de "
                          f"{reference['label']}.")
            elif "sin sinopsis argumental" in book.get("description_origin", ""):
                reason = (f"Clasificado como {', '.join(book['genres'][:2])} en este catálogo. "
                          "La ficha no permite confirmar detalles de la trama.")
            else:
                reason = (explanations or {}).get(book["id"]) or \
                         f"Coincide con tu búsqueda según {'similitud semántica' if method == 'semantic' else 'términos del catálogo'}. Género: {', '.join(book['genres'][:2])}."
            recommendations.append({"id": book["id"], "title": book["title"], "author": book["author"],
                "genres": book["genres"], "language": book["language"], "description": book["description"],
                "score": round(float(score), 5), "source_url": book.get("source_url"),
                "reason": reason})
        if reference:
            answer = (f"Tomé como referencia los libros de {reference['label']} y busqué obras de otros autores "
                      f"que comparten categorías del catálogo: {', '.join(reference['genres'][:5])}."
                      if recommendations else f"No encontré alternativas a {reference['label']} que cumplan los filtros en este catálogo.")
            explanation_mode = "reference"
        else:
            answer = (f"Encontré {len(recommendations)} libro(s) del catálogo que podrían interesarte."
                      if recommendations else "No encontré libros del catálogo con esos criterios. Prueba con otra descripción o filtros.")
            explanation_mode = "llm" if explanations else "basic"
        return {"query": request.query, "method": method, "recommendations": recommendations,
                "answer": answer, "explanation_mode": explanation_mode,
                "selection_mode": selection_mode, "selection_confidence": selection_confidence,
                "response_time_ms": round((time.perf_counter() - start) * 1000, 2)}

    @app.post("/api/recommend")
    def recommend(request: SearchRequest):
        return run(request, "semantic")

    @app.post("/api/recommend/jev")
    def recommend_jev(request: SearchRequest):
        return run(request, "semantic_jev")

    @app.post("/api/search/bm25")
    def bm25(request: SearchRequest):
        return run(request, "bm25")

    return app


app = create_app()
