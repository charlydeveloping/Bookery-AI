"""Download Google Books metadata and prepare a fixed, inspectable catalog."""
import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from ml.dataset import normalize_text

QUERIES = ("subject:fiction", "subject:fantasy", "subject:science fiction", "subject:mystery", "subject:history", "subject:romance")


def download(pages, raw_path):
    items = []
    for query in QUERIES:
        for page in range(pages):
            params = urllib.parse.urlencode({"q": query, "startIndex": page * 40, "maxResults": 40, "printType": "books", "projection": "full"})
            request = urllib.request.Request("https://www.googleapis.com/books/v1/volumes?" + params, headers={"User-Agent": "BookeryAI-Academic/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                items.extend(json.load(response).get("items", []))
            time.sleep(1)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return items


def clean(items):
    books = {}
    for item in items:
        info = item.get("volumeInfo", {})
        description = re.sub(r"<[^>]+>", " ", html.unescape(info.get("description", "")))
        description = normalize_text(re.sub(r"\s+", " ", description))
        title = normalize_text(info.get("title"))
        authors = info.get("authors") or []
        genres = [normalize_text(x) for x in info.get("categories", []) if normalize_text(x)]
        language = normalize_text(info.get("language")).lower()
        book_id = normalize_text(item.get("id"))
        if not book_id or not title or not authors or len(description) < 80 or not genres or not language:
            continue
        author = ", ".join(normalize_text(x) for x in authors)
        books[book_id] = {"id": book_id, "title": title, "author": author,
            "description": description, "genres": genres, "language": language,
            "published_date": info.get("publishedDate"), "page_count": info.get("pageCount"),
            "source_url": info.get("infoLink"),
            "searchable_text": f"{title}. {author}. {', '.join(genres)}. {description}"}
    return sorted(books.values(), key=lambda book: book["id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--raw", default="data/raw/google_books.json")
    parser.add_argument("--output", default="data/processed/books_google_unreviewed.json")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    raw_path = Path(args.raw)
    if args.pages < 1:
        parser.error("--pages must be positive")
    items = json.loads(raw_path.read_text(encoding="utf-8")) if args.offline else download(args.pages, raw_path)
    books = clean(items)
    if not books:
        raise SystemExit("No usable books found; inspect source data")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(books, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Downloaded {len(items)} records; retained {len(books)} books -> {output}")


if __name__ == "__main__":
    main()
