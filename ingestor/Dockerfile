# ------------------------------------------------------------
# Dockerfile — Ingestor service (Llama‑Index + Qdrant)
# ------------------------------------------------------------
# Build context: root of the monorepo (contains pyproject.toml)
# Secrets: **NON** inserire OPENAI_API_KEY nell’immagine – passala
#          al run‑time con `-e OPENAI_API_KEY=sk-…` o Docker secrets.
# ------------------------------------------------------------

FROM python:3.11-slim AS base

# ---- System deps ---------------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# ---- Poetry --------------------------------------------------------------
ARG POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry \
    POETRY_CACHE_DIR=/var/cache/pypoetry
RUN curl -sSL https://install.python-poetry.org | python - \
 && ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry

# ---- App source ----------------------------------------------------------
WORKDIR /app
COPY pyproject.toml poetry.lock* ./

# Install only the *ingestor* extra to keep the image small
RUN poetry install --only ingestor --no-root --sync

# Copy sources
COPY . .

# ---- Runtime env ---------------------------------------------------------
ENV \
    CRAWLER_API_URL=http://localhost:8000/crawl \
    CRAWL_SITE_URL=https://www.esempio.com \
    CRAWL_DEPTH=3 \
    CRAWL_TIMEOUT=30 \
    QDRANT_URL=http://qdrant:6333 \
    QDRANT_COLLECTION=md_embeddings \
    EMBED_MODEL=text-embedding-3-small

# ❗  OPENAI_API_KEY va passato al run‑time:
#     docker run -e OPENAI_API_KEY=sk-… ingestor:latest

CMD ["poetry", "run", "run-ingestor"]