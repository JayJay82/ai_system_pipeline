#!/usr/bin/env python3
"""
Retrieve da Qdrant con Llama-Index 0.10.x.

Esempio:
    poetry run python retrieve.py \
        --question "Qual è la missione di OpenAI?"
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# ── Load .env ──────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

def parse_args():
    p = argparse.ArgumentParser(
        description="Retrieval QA da Qdrant usando Llama-Index"
    )
    p.add_argument(
        "--question", "-q",
        required=True,
        help="Testo della domanda da porre all'indice"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # ── Config da env ────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Servono le credenziali OPENAI_API_KEY in .env")

    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    llm_model   = os.getenv("LLM_MODEL", "gpt-4o-mini")
    qdrant_url  = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection  = os.getenv("QDRANT_COLLECTION", "md_embeddings")

    # ── Setup client Qdrant ───────────────────────────────────────────
    client = QdrantClient(url=qdrant_url)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection
    )
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # ── Setup models Llama-Index ──────────────────────────────────────
    embed = OpenAIEmbedding(
        model=embed_model,
        openai_api_key=openai_key
    )
    llm = OpenAI(
        model=llm_model,
        api_key=openai_key,
        temperature=0.0
    )
    service_context = ServiceContext.from_defaults(
        embed_model=embed,
        llm=llm,
    )
    # ── Carica l'indice direttamente da Qdrant ────────────────────────
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        service_context=service_context
    )

    # ── Crea il motore di query ──────────────────────────────────────
    query_engine = index.as_query_engine(
        similarity_top_k=5,  # quanti frammenti recuperare
        streaming=False  # True per streaming token-by-token
    )

    # ── Esegui la query ───────────────────────────────────────────────
    response = query_engine.query(args.question)
    print("\n=== RISPOSTA ===\n")
    print(response)

if __name__ == "__main__":
    main()
