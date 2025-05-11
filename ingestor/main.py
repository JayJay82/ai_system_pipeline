#!/usr/bin/env python3
"""Ingest a site (crawl ➜ markdown) into a Qdrant collection using **Llama‑Index 0.10.x**.

* No local docstore → all node text remains in Qdrant payload.
* Vector‑size is **auto‑matched** to the embedding model (handles 1536 vs 3072).

Install (Poetry extra *ingestor*):
    poetry add --group=ingestor \
        llama-index==0.10.3 \
        "llama-index-vector-stores-qdrant>=0.2.15,<0.3" \
        qdrant-client openai requests python-dotenv

Required env vars (.env next to this file):
    OPENAI_API_KEY          – OpenAI key

Optional env vars (defaults shown):
    EMBED_MODEL             – text-embedding-3-small   # 1536‑dim  (large = 3072)
    LLM_MODEL               – gpt4o-mini
    QDRANT_URL              – http://localhost:6333
    QDRANT_COLLECTION       – md_embeddings
    CRAWLER_API_URL         – http://localhost:8000/crawl
    CRAWL_SITE_URL          – https://www.esempio.com
    CRAWL_DEPTH             – 3
    CRAWL_TIMEOUT           – 30
"""

import os
from pathlib import Path
import requests
from dotenv import load_dotenv

# ── Llama‑Index imports ──────────────────────────────────────────────
from llama_index.core import Document, ServiceContext, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

from qdrant_client import QdrantClient

# ── Load .env located in the same folder as this script ─────────────
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, verbose=True)  # prints a warning if missing


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
DIM_MAP = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def get_embed_dim(model_name: str) -> int:
    """Return the known dimension or fallback to 1536."""
    return DIM_MAP.get(model_name, 1536)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    # 📌 Models from env -------------------------------------------------
    embed_model_name = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    llm_model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # 1️⃣  Connect or create Qdrant collection --------------------------
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "md_embeddings")

    client = QdrantClient(url=qdrant_url)

    expected_dim = get_embed_dim(embed_model_name)

    def _get_current_dim(col_name: str) -> int | None:
        """Return existing vector size or None if not found/unknown."""
        try:
            info = client.get_collection(collection_name=col_name)
            # http / grpc response both expose .result.config.params.vectors
            vectors = info.result.config.params.vectors  # type: ignore[attr-defined]
            if isinstance(vectors, list):
                return vectors[0].size  # multi‑vector config
            return vectors.size  # single vector
        except Exception:
            return None

    current_dim = _get_current_dim(collection)

    if current_dim is not None and current_dim != expected_dim:
        print(
            f"[!] Collection '{collection}' dim={current_dim} ≠ expected {expected_dim}. "
            "Recreating…"
        )
        client.recreate_collection(
            collection_name=collection,
            vectors_config={"size": expected_dim, "distance": "Cosine"},
        )
    elif current_dim is None:
        # collection does not exist
        client.recreate_collection(
            collection_name=collection,
            vectors_config={"size": expected_dim, "distance": "Cosine"},
        )

    # 2️⃣  Llama‑Index service context (LLM + embeddings + parser) ------  Llama‑Index service context (LLM + embeddings + parser) ------
    embed_model = OpenAIEmbedding(
        model=embed_model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    llm = OpenAI(
        model=llm_model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0,
    )

    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )

    # 3️⃣  Call crawler API (markdown) ----------------------------------
    crawl_api = os.getenv("CRAWLER_API_URL", "http://localhost:8000/crawl")
    site_url = os.getenv("CRAWL_SITE_URL", "https://www.esempio.com")
    depth = int(os.getenv("CRAWL_DEPTH", 3))
    timeout = int(os.getenv("CRAWL_TIMEOUT", 30))

    resp = requests.post(
        crawl_api,
        json={"url": site_url, "depth": depth, "timeout": timeout},
        timeout=timeout + 5,
    )
    resp.raise_for_status()
    md_text = resp.content.decode("utf-8")

    # 4️⃣  Build Document ----------------------------------------------
    doc = Document(
        text=md_text,
        doc_id=site_url,
        metadata={"source": site_url, "depth": depth},
    )

    # 5️⃣  Storage context (only vector store, no local docstore) ------
    vector_store = QdrantVectorStore(client=client, collection_name=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 6️⃣  Index + upsert ----------------------------------------------
    VectorStoreIndex.from_documents(
        [doc],
        storage_context=storage_context,
        service_context=service_context,
    )

    print(
        f"[✓] Ingestione completata: {site_url} → {collection} "
        f"(dim={expected_dim}, model={embed_model_name})"
    )


if __name__ == "__main__":
    main()
