"""Create a reproducible Spanish-edition catalog expansion from Open Library."""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from ml.dataset import folded_words, normalize_text


CATEGORIES = {
    "fantasy": "Fantasía",
    "science_fiction": "Ciencia ficción",
    "mystery": "Misterio",
    "romance": "Romance",
    "horror": "Terror",
    "historical_fiction": "Novela histórica",
    "young_adult_fiction": "Literatura juvenil",
    "children's_literature": "Literatura infantil",
    "classic_literature": "Clásicos",
    "poetry": "Poesía",
    "philosophy": "Filosofía",
    "psychology": "Psicología",
    "self_help": "Crecimiento personal",
    "business": "Negocios",
    "history": "Historia",
    "biography": "Biografía",
    "science": "Ciencia",
    "technology": "Tecnología",
    "travel": "Viajes",
    "cooking": "Cocina",
    "art": "Arte",
    "comics": "Cómic",
}

TARGET_TITLES = {
    "Around the World in Eighty Days": "Viajes",
    "On the Road": "Viajes",
    "Into the Wild": "Viajes",
    "Clean Code": "Tecnología",
    "The Pragmatic Programmer": "Tecnología",
    "Twenty Love Poems and a Song of Despair": "Poesía",
}

SEARCH_FIELDS = "key,title,author_name,first_publish_year,subject,readinglog_count,editions,editions.key,editions.title,editions.language"

TOPIC_WORDS = {
    "adventure": "aventura", "magic": "magia", "witch": "brujería",
    "space": "espacio", "alien": "vida extraterrestre", "future": "futuro",
    "crime": "crimen", "detective": "investigación", "murder": "asesinato",
    "love": "amor", "relationship": "relaciones", "family": "familia",
    "ghost": "fantasmas", "vampire": "vampiros", "war": "guerra",
    "history": "historia", "politic": "política", "school": "escuela",
    "child": "infancia", "teen": "adolescencia", "friend": "amistad",
    "poem": "poesía", "ethic": "ética", "mind": "mente",
    "emotion": "emociones", "habit": "hábitos", "leader": "liderazgo",
    "entrepreneur": "emprendimiento", "life": "vida", "physics": "física",
    "computer": "computación", "programming": "programación", "food": "alimentación",
    "recipe": "recetas", "paint": "pintura", "music": "música",
    "graphic": "arte gráfico", "comic": "historieta", "travel": "viajes",
}


def get_json(url, opener=urllib.request.urlopen):
    request = urllib.request.Request(url, headers={"User-Agent": "BookeryAI-AcademicCatalog/1.0"})
    for attempt in range(3):
        try:
            with opener(request, timeout=30) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def spanish_edition(doc):
    editions = (doc.get("editions") or {}).get("docs") or []
    for edition in editions:
        key = edition.get("key", "")
        title = normalize_text(edition.get("title"))
        if (key.startswith("/books/OL") and key.endswith("M") and
                "spa" in (edition.get("language") or []) and len(folded_words(title)) >= 6):
            return key.rsplit("/", 1)[-1], title
    return None


def topic_tags(doc):
    subjects = [folded_words(value) for value in (doc.get("subject") or [])[:80]]
    tags = []
    for word, label in TOPIC_WORDS.items():
        if any(word in subject for subject in subjects) and label not in tags:
            tags.append(label)
    return tags[:3]


def build_entry(doc, category, label):
    key = doc.get("key", "")
    authors = doc.get("author_name") or []
    edition = spanish_edition(doc)
    if not key.startswith("/works/OL") or not key.endswith("W") or not authors or not edition:
        return None
    book_id = key.rsplit("/", 1)[-1]
    author = normalize_text(authors[0])
    edition_id, title = edition
    tags = topic_tags(doc)
    genres = [label, *tags]
    topics = ", ".join(tags) if tags else label.casefold()
    description = (f"Edición en español de una obra de {label.casefold()} de {author}. "
                   f"Open Library la relaciona con {topics}. "
                   "La ficha importada no contiene una sinopsis argumental verificada en español.")
    annotation = {"id": book_id, "title": title, "genres": genres,
                  "description": description,
                  "description_origin": "resumen de metadatos de Open Library; sin sinopsis argumental",
                  "popularity_signal": doc.get("readinglog_count", 0),
                  "selection_category": category}
    source = {"id": book_id, "work_title": normalize_text(doc.get("title")),
              "author": author, "edition_id": edition_id, "edition_title": title,
              "edition_language": "spa", "first_publish_year": doc.get("first_publish_year"),
              "source_url": f"https://openlibrary.org/books/{edition_id}",
              "readinglog_count": doc.get("readinglog_count", 0)}
    return annotation, source


def collect(existing_ids, quota=10, search_limit=40, fetch=get_json, pause=1.1):
    annotations, sources = [], []
    seen = set(existing_ids)
    counts = {}
    for category, label in CATEGORIES.items():
        params = urllib.parse.urlencode({"q": f"subject:{category} language:spa",
            "sort": "readinglog", "limit": search_limit, "lang": "es",
            "fields": SEARCH_FIELDS})
        data = fetch("https://openlibrary.org/search.json?" + params)
        count = 0
        for doc in data.get("docs", []):
            entry = build_entry(doc, category, label)
            if not entry or entry[0]["id"] in seen:
                continue
            annotation, source = entry
            annotations.append(annotation)
            sources.append(source)
            seen.add(annotation["id"])
            count += 1
            if count >= quota:
                break
        counts[category] = count
        if pause:
            time.sleep(pause)
    return annotations, sources, counts


def collect_targeted(existing_ids, fetch=get_json, pause=1.1):
    annotations, sources = [], []
    seen = set(existing_ids)
    for title, label in TARGET_TITLES.items():
        params = urllib.parse.urlencode({"q": f'title:"{title}" language:spa',
            "sort": "readinglog", "limit": 8, "lang": "es", "fields": SEARCH_FIELDS})
        data = fetch("https://openlibrary.org/search.json?" + params)
        for doc in data.get("docs", []):
            entry = build_entry(doc, "targeted:" + title, label)
            if entry and entry[0]["id"] not in seen:
                annotations.append(entry[0])
                sources.append(entry[1])
                seen.add(entry[0]["id"])
                break
        if pause:
            time.sleep(pause)
    return annotations, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quota", type=int, default=10)
    parser.add_argument("--search-limit", type=int, default=40)
    parser.add_argument("--targeted-only", action="store_true")
    args = parser.parse_args()
    if args.quota < 1 or args.search_limit < args.quota:
        parser.error("Use a positive quota and a search limit at least as large as the quota")
    existing = json.loads(Path("data/curated/annotations_es.json").read_text(encoding="utf-8"))
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.targeted_only:
        annotations = json.loads((output_dir / "popular_candidates_annotations_es.json").read_text(encoding="utf-8"))
        sources = json.loads((output_dir / "popular_candidates_editions_es.json").read_text(encoding="utf-8"))
        counts = {}
    else:
        annotations, sources, counts = collect({book["id"] for book in existing}, args.quota, args.search_limit)
    targeted_annotations, targeted_sources = collect_targeted(
        {book["id"] for book in existing} | {book["id"] for book in annotations})
    annotations.extend(targeted_annotations)
    sources.extend(targeted_sources)
    if not annotations:
        raise SystemExit("No Spanish-edition records were found")
    for filename, values in (("popular_candidates_annotations_es.json", annotations),
                             ("popular_candidates_editions_es.json", sources)):
        (output_dir / filename).write_text(
            json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {len(annotations)} Spanish-edition works; category counts: {counts}")


if __name__ == "__main__":
    main()
