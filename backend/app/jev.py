"""Optional TypeSafe Jev selection over books already retrieved from the catalog."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)


class JevSelector:
    def __init__(self, api_key, model="jev-latest", base_url="https://api.typesafe.ai/v1/systemone", client=None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.client = client or httpx.Client(timeout=5.0)

    @classmethod
    def from_env(cls):
        key = os.getenv("TYPESAFE_API_KEY", "").strip()
        return cls(key, os.getenv("BOOKERY_JEV_MODEL", "jev-latest")) if key else None

    def choose(self, query, books):
        if len(books) < 2:
            return None
        candidates = {book["id"]: book for book in books}
        state = {"consulta": query, "candidatos": [
            {"id": book["id"], "titulo": book["title"], "autor": book["author"],
             "generos": book["genres"], "descripcion": book["description"][:900]}
            for book in books]}
        payload = {"state": state, "model": self.model, "questions": {
            "libro_indicado": {"type": "choice",
                "instructions": "¿Qué libro satisface mejor la petición del lector según las fichas? Elige ninguno si falta evidencia suficiente.",
                "criteria": {**{book_id: f"{book['title']} — {book['author']}" for book_id, book in candidates.items()},
                             "ninguno": "Ninguna ficha satisface suficientemente la petición"}}}}
        try:
            response = self.client.post(self.base_url, json=payload,
                                        headers={"Authorization": f"Bearer {self.api_key}"}, timeout=5.0)
            response.raise_for_status()
            answer = response.json()["answers"]["libro_indicado"]
            choice = answer["choice"]
            confidence = answer["confidence"]
            probabilities = answer["probabilities"]
            if (choice not in candidates and choice != "ninguno") or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError("Invalid Jev choice")
            if not isinstance(probabilities, dict) or set(probabilities) != set(candidates) | {"ninguno"}:
                raise ValueError("Invalid Jev probabilities")
            if any(not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in probabilities.values()):
                raise ValueError("Invalid Jev probability")
            return {"choice": choice, "confidence": float(confidence), "probabilities": probabilities}
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            logger.warning("Jev selection unavailable: %s", type(error).__name__)
            return None
