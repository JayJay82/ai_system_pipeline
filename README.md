
# ai_system_pipeline — Crawler + Ingestor

> **Goal:** Crawl any public website, convert it to Markdown, embed the text with OpenAI models, and upsert the vectors into a Qdrant collection.

## Project structure

| Folder | Description |
| ------ | ----------- |
| `crawler/`  | FastAPI micro‑service that fetches pages (with Playwright), cleans HTML and returns Markdown. |
| `ingestor/` | CLI / Docker service that calls the crawler, chunks the Markdown, generates embeddings and writes them to Qdrant. |

```
ai_system_pipeline/
├─ crawler/
│  ├─ main.py        # FastAPI entry‑point
│  └─ ...            # crawler utils
├─ ingestor/
│  ├─ main.py        # script we tweaked together
│  └─ crawler_ingest.py
├─ Dockerfile        # builds the ingestor image
├─ pyproject.toml    # Poetry config (extras: crawler, ingestor)
└─ README.md         # ← this file
```

## Quick start (local)

```bash
# 1) clone + install Poetry deps
poetry install --with crawler,ingestor --sync

# 2) set your OpenAI key (PowerShell)
$Env:OPENAI_API_KEY = "sk‑...long‑key..."

# 3) launch crawler API (optional depth/playwright flags)
poetry run run-crawler

# 4) run ingestor (defaults: https://www.example.com, depth 3)
poetry run run-ingestor
```

## Environment variables

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `OPENAI_API_KEY` | _none_ | OpenAI credential (mandatory) |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model (1536‑dim) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `md_embeddings` | Collection name |
| `CRAWLER_API_URL` | `http://localhost:8000/crawl` | Crawling service |
| `CRAWL_SITE_URL` | `https://www.example.com` | Root URL to ingest |
| `CRAWL_DEPTH` | `3` | Max link depth |
| `CRAWL_TIMEOUT` | `30` | Seconds per request |

Place them in a `.env` or pass `-e VAR=value`.

## Docker

```bash
# Build the ingestor image
docker build -t ingestor:latest -f Dockerfile .

# Run against local Qdrant + Crawler
docker run --rm   -e OPENAI_API_KEY=sk-...   -e QDRANT_URL=http://host.docker.internal:6333   -e CRAWLER_API_URL=http://host.docker.internal:8000/crawl   ingestor:latest
```

### docker‑compose example

```yaml
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: [ "6333:6333" ]

  crawler:
    build:
      context: .
      dockerfile: Dockerfile.crawler  # if you have one
    environment:
      - PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium
    ports: [ "8000:8000" ]

  ingestor:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - CRAWLER_API_URL=http://crawler:8000/crawl
    depends_on: [ qdrant, crawler ]
```

## Troubleshooting

| Symptom | Fix |
| ------- | ---- |
| `Illegal header value b'Bearer '` | `OPENAI_API_KEY` missing/empty |
| `Vector dimension error` (400) | Collection size ≠ embedding dim. Delete or recreate with correct `size`. |
| Retry `APIConnectionError` | Check network/proxy or key permissions. |

---

© 2025 — Your Name
