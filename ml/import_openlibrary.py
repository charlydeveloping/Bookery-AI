"""Import real Open Library works with nonempty descriptions."""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from ml.dataset import normalize_text

QUERIES = ("science fiction", "fantasy", "mystery", "history", "romance", "dystopia")
LAST_REQUEST = 0.0


def get_json(url):
    global LAST_REQUEST
    wait = 1.1 - (time.monotonic() - LAST_REQUEST)
    if wait > 0:
        time.sleep(wait)
    request = urllib.request.Request(url, headers={"User-Agent": "BookeryAI-Academic/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    finally:
        LAST_REQUEST = time.monotonic()


def import_books(limit_per_query=20):
    books = {}
    raw = []
    for query in QUERIES:
        params = urllib.parse.urlencode({"q": query, "limit": limit_per_query,
            "fields": "key,title,author_name,subject,language,first_publish_year,number_of_pages_median"})
        search = get_json("https://openlibrary.org/search.json?" + params)
        raw.append({"query": query, "response": search})
        for doc in search.get("docs", []):
            key = doc.get("key", "")
            if not key.startswith("/works/") or key in books:
                continue
            try:
                work = get_json("https://openlibrary.org" + key + ".json")
            except (OSError, ValueError):
                continue
            description = work.get("description", "")
            if isinstance(description, dict):
                description = description.get("value", "")
            description = normalize_text(description)
            title = normalize_text(doc.get("title"))
            authors = doc.get("author_name") or []
            subjects = [normalize_text(x) for x in (doc.get("subject") or [])[:8]]
            languages = doc.get("language") or []
            if len(description) < 80 or not title or not authors or not subjects or not languages:
                continue
            author = ", ".join(normalize_text(x) for x in authors[:3])
            book_id = key.split("/")[-1]
            books[key] = {"id": book_id, "title": title, "author": author,
                "description": description, "genres": subjects, "language": "en" if "eng" in languages else languages[0],
                "published_date": doc.get("first_publish_year"), "page_count": doc.get("number_of_pages_median"),
                "source_url": "https://openlibrary.org" + key,
                "searchable_text": f"{title}. {author}. {', '.join(subjects)}. {description}"}
    return raw, sorted(books.values(), key=lambda book: book["id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-query", type=int, default=20)
    parser.add_argument("--output", default="data/processed/books.json")
    args = parser.parse_args()
    if args.limit_per_query < 1:
        parser.error("--limit-per-query must be positive")
    raw, books = import_books(args.limit_per_query)
    if not books:
        raise SystemExit("No books with descriptions returned")
    raw_path = Path("data/raw/openlibrary_search.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(books, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Retained {len(books)} real Open Library works -> {output}")


if __name__ == "__main__":
    main()
