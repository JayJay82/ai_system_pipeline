# ai_system_pipeline — Crawler + Ingestor + Retriever

**Goal:** Crawl any public website, convert it to Markdown, embed the text with OpenAI models, upsert the vectors into a Qdrant collection, and finally perform QA retrieval.

---

## Project structure

| Folder               | Description                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------------|
| `crawler/`           | FastAPI micro-service that fetches pages (with Playwright), cleans HTML and returns Markdown.   |
| `ingestor/`          | CLI / Docker service that calls the crawler, chunks the Markdown, generates embeddings, and upserts them into Qdrant. |
| `retriever/`         | CLI script that queries the Qdrant-backed index via Llama-Index for question-answering.         |
| `Dockerfile`         | Builds the ingestor image.                                                                      |
| `pyproject.toml`     | Poetry config (extras: crawler, ingestor, retrieve).                                            |
| `README.md`          | ← this file                                                                                     |

---

## Quick start (local)

1. **Clone & install Poetry deps**
   ```bash
   poetry install --with crawler,ingestor,retrieve --sync
   ```
2. **Set your OpenAI key** (PowerShell)
   ```powershell
   $Env:OPENAI_API_KEY = "sk-...long-key..."
   ```
3. **Launch crawler API** (optional depth/playwright flags)
   ```bash
   poetry run run-crawler
   ```
4. **Run ingestor** (defaults: `https://www.example.com`, depth 3)
   ```bash
   poetry run run-ingestor
   ```
5. **Run retriever** (QA on the ingested site)
   ```bash
   poetry run retrieve --question "Qual è la missione di OpenAI?"
   ```

---

## Environment variables

| Variable               | Default                                 | Purpose                                                      |
|------------------------|-----------------------------------------|--------------------------------------------------------------|
| `OPENAI_API_KEY`       | _none_                                  | OpenAI credential (mandatory)                                |
| `EMBED_MODEL`          | `text-embedding-3-small`                | Embedding model (1536-dim)                                   |
| `LLM_MODEL`            | `gpt-4o-mini`                           | LLM model for generation                                     |
| `QDRANT_URL`           | `http://localhost:6333`                 | Qdrant endpoint                                              |
| `QDRANT_COLLECTION`    | `md_embeddings`                         | Collection name                                              |
| `CRAWLER_API_URL`      | `http://localhost:8000/crawl`          | Crawling service                                             |
| `CRAWL_SITE_URL`       | `https://www.example.com`               | Root URL to ingest                                           |
| `CRAWL_DEPTH`          | `3`                                     | Max link depth                                               |
| `CRAWL_TIMEOUT`        | `30`                                    | Seconds per crawl request                                    |

> Place them in a `.env` file or pass `-e VAR=value` when running Docker.

---

## Docker

1. **Build the ingestor image**
   ```bash
   docker build -t ingestor:latest -f Dockerfile .
   ```
2. **Run against local Qdrant + Crawler**
   ```bash
   docker run --rm      -e OPENAI_API_KEY=sk-...      -e QDRANT_URL=http://host.docker.internal:6333      -e CRAWLER_API_URL=http://host.docker.internal:8000/crawl      ingestor:latest
   ```

---

## docker-compose example

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

  retriever:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - ./:/app
    command: poetry run retrieve --question "<Your question>"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_COLLECTION=md_embeddings
    depends_on: [ qdrant ]
```

---

## Troubleshooting

| Symptom                               | Fix                                                                                         |
|---------------------------------------|---------------------------------------------------------------------------------------------|
| Illegal header value `b'Bearer '`     | `OPENAI_API_KEY` missing/empty                                                              |
| Vector dimension error (400)          | Collection size ≠ embedding dim. Delete or recreate with correct size.                      |
| `AttributeError: load_index_from_storage` | Use `VectorStoreIndex.from_vector_store(...)` instead of `load_index_from_storage`.        |
| Retry `APIConnectionError`            | Check network/proxy or key permissions.                                                     |

---

© 2025 — Your Name
