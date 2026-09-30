"""Build the Spanish demonstration catalog from reviewed annotations and source metadata."""

import argparse
import json
from pathlib import Path

from ml.dataset import normalize_text


def build_catalog(annotations, sources):
    if not isinstance(annotations, list) or not isinstance(sources, list):
        raise ValueError("Catalog inputs must be lists")
    source_by_id = {item["id"]: item for item in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("Duplicate source ID")
    books = []
    seen = set()
    for item in annotations:
        book_id = item["id"]
        if book_id in seen:
            raise ValueError(f"Duplicate annotation: {book_id}")
        seen.add(book_id)
        source = source_by_id.get(book_id, {})
        if source:
            if source.get("edition_language") != "spa" or not source.get("edition_id"):
                raise ValueError(f"Spanish edition missing: {book_id}")
            metadata_source = "Open Library"
        else:
            source = {"author": item.get("author"), "first_publish_year": item.get("published_date"),
                      "edition_id": item.get("edition_id"), "source_url": item.get("source_url")}
            if not all((item.get("author"), item.get("source_url"), item.get("source_name"), item.get("edition_id"))):
                raise ValueError(f"Source metadata missing: {book_id}")
            metadata_source = item["source_name"]
        title = normalize_text(item.get("title"))
        author = normalize_text(source.get("author"))
        description = normalize_text(item.get("description"))
        genres = [normalize_text(genre) for genre in item.get("genres", [])]
        if not title or not author or len(description) < 80 or not genres or not all(genres):
            raise ValueError(f"Incomplete annotation: {book_id}")
        aliases = [normalize_text(alias) for alias in item.get("reference_aliases", [])]
        if not all(aliases):
            raise ValueError(f"Invalid reference alias: {book_id}")
        books.append({"id": book_id, "title": title, "author": author,
                      "description": description, "genres": genres, "language": "es",
                      "reference_aliases": aliases,
                      "published_date": source.get("first_publish_year"),
                      "edition_id": source["edition_id"], "source_url": source["source_url"],
                      "metadata_source": metadata_source,
                      "description_origin": "redacción original para demostración académica",
                      "reference_topics": [normalize_text(topic) for topic in item.get("reference_topics", [])],
                      "searchable_text": f"{title}. {author}. {', '.join(genres)}. {description}"})
    if not books:
        raise ValueError("Catalog cannot be empty")
    return sorted(books, key=lambda book: book["id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/curated/annotations_es.json")
    parser.add_argument("--sources", default="data/curated/openlibrary_editions_es.json")
    parser.add_argument("--output", default="data/processed/books.json")
    args = parser.parse_args()
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    books = build_catalog(annotations, sources)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(books, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built {len(books)} Spanish catalog records -> {output}")


if __name__ == "__main__":
    main()
