import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from ml.bm25_retriever import BM25Retriever
from ml.dataset import load_books, reference_context
from ml.semantic_retriever import SemanticRetriever
from backend.app.llm import LLMExplainer

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


def create_app(books=None, semantic=None, explainer=None):
    @asynccontextmanager
    async def lifespan(app):
        if not hasattr(app.state, "books"):
            app.state.books = load_books(os.getenv("BOOKERY_CATALOG", "data/processed/books.json"))
            app.state.bm25 = BM25Retriever(app.state.books)
            app.state.semantic = SemanticRetriever(app.state.books, os.getenv("BOOKERY_EMBEDDINGS", "data/processed/embeddings.npy"))
            app.state.explainer = LLMExplainer.from_env()
        yield

    app = FastAPI(title="Bookery AI", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_methods=["GET", "POST"], allow_headers=["*"])
    if books is not None:
        app.state.books = books
        app.state.bm25 = BM25Retriever(books)
        app.state.semantic = semantic
        app.state.explainer = explainer

    @app.get("/health")
    def health():
        return {"status": "ok", "catalog_size": len(app.state.books)}

    @app.get("/api/options")
    def options():
        return {"languages": sorted({book["language"] for book in app.state.books}),
                "genres": sorted({genre for book in app.state.books for genre in book["genres"]})}

    def run(request, method):
        start = time.perf_counter()
        retriever = app.state.semantic if method == "semantic" else app.state.bm25
        if retriever is None:
            raise HTTPException(status_code=503, detail="Índice semántico no disponible")
        results = retriever.search(request.query, k=request.limit, language=request.language, genre=request.genre, author=request.author)
        books = [book for book, _ in results]
        reference = reference_context(request.query, app.state.books)
        explanations = (app.state.explainer.explain(request.query, books)
                        if method == "semantic" and app.state.explainer and books and not reference else None)
        recommendations = []
        for book, score in results:
            shared_topics = sorted(set(book["genres"]) & set(reference["genres"])) if reference else []
            if reference and shared_topics:
                reason = (f"Comparte los temas {', '.join(shared_topics)} asociados a los libros de "
                          f"{reference['label']}.")
            else:
                reason = (explanations or {}).get(book["id"]) or \
                         f"Coincide con tu búsqueda según {'similitud semántica' if method == 'semantic' else 'términos del catálogo'}. Género: {', '.join(book['genres'][:2])}."
            recommendations.append({"id": book["id"], "title": book["title"], "author": book["author"],
                "genres": book["genres"], "language": book["language"], "description": book["description"],
                "score": round(float(score), 5), "source_url": book.get("source_url"),
                "reason": reason})
        if reference:
            answer = (f"Tomé como referencia los libros de {reference['label']} y busqué obras de otros autores "
                      f"que comparten estos temas: {', '.join(reference['topics'][:5])}.")
            explanation_mode = "reference"
        else:
            answer = (f"Encontré {len(recommendations)} libro(s) del catálogo que podrían interesarte."
                      if recommendations else "No encontré libros del catálogo con esos criterios. Prueba con otra descripción o filtros.")
            explanation_mode = "llm" if explanations else "basic"
        return {"query": request.query, "method": method, "recommendations": recommendations,
                "answer": answer, "explanation_mode": explanation_mode,
                "response_time_ms": round((time.perf_counter() - start) * 1000, 2)}

    @app.post("/api/recommend")
    def recommend(request: SearchRequest):
        return run(request, "semantic")

    @app.post("/api/search/bm25")
    def bm25(request: SearchRequest):
        return run(request, "bm25")

    return app


app = create_app()
