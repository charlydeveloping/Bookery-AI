"""Freeze manually reviewed popular works into the catalog input snapshots."""

import json
from pathlib import Path


def review(candidates, sources, selections):
    candidate_by_id = {item["id"]: item for item in candidates}
    source_by_id = {item["id"]: item for item in sources}
    if len(candidate_by_id) != len(candidates) or len(source_by_id) != len(sources):
        raise ValueError("Duplicate candidate ID")
    selected_annotations, selected_sources = [], []
    seen = set()
    for selection in selections:
        book_id = selection["id"]
        genre = selection["genre"].strip()
        if book_id in seen or book_id not in candidate_by_id or book_id not in source_by_id or not genre:
            raise ValueError(f"Invalid reviewed book: {book_id}")
        seen.add(book_id)
        candidate = candidate_by_id[book_id]
        source = source_by_id[book_id]
        if source["edition_language"] != "spa":
            raise ValueError(f"No Spanish edition: {book_id}")
        topics = candidate["genres"][1:]
        topic_text = (f" La ficha temática de Open Library menciona {', '.join(topics)}."
                      if topics else "")
        description = (f"Edición en español de «{candidate['title']}», obra de {source['author']}. "
                       f"Esta colección la clasifica como {genre.casefold()}.{topic_text} "
                       "No hay una sinopsis argumental revisada para esta ficha.")
        selected_annotations.append({**candidate, "genres": [genre], "description": description,
                                     "description_origin": "resumen de metadatos revisado manualmente; sin sinopsis argumental",
                                     "reviewed": True})
        selected_sources.append(source)
    return selected_annotations, selected_sources


def main():
    raw = Path("data/raw")
    curated = Path("data/curated")
    candidates = json.loads((raw / "popular_candidates_annotations_es.json").read_text(encoding="utf-8"))
    sources = json.loads((raw / "popular_candidates_editions_es.json").read_text(encoding="utf-8"))
    selections = json.loads((curated / "popular_review_es.json").read_text(encoding="utf-8"))
    annotations, editions = review(candidates, sources, selections)
    for filename, values in (("popular_annotations_es.json", annotations),
                             ("popular_editions_es.json", editions)):
        (curated / filename).write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Reviewed {len(annotations)} Spanish-edition works")


if __name__ == "__main__":
    main()
