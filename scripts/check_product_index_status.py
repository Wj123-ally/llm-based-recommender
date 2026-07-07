from __future__ import annotations

import pickle
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from pymilvus import MilvusClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def main() -> None:
    conn = sqlite3.connect(config.DATABASE_PATH)
    product_count = conn.execute("SELECT count(*) FROM products").fetchone()[0]
    print("sqlite_products", product_count)

    print("bm25_exists", config.BM25_INDEX_PATH.exists())
    print("bm25_size", config.BM25_INDEX_PATH.stat().st_size if config.BM25_INDEX_PATH.exists() else 0)
    if config.BM25_INDEX_PATH.exists():
        with config.BM25_INDEX_PATH.open("rb") as file:
            payload = pickle.load(file)
        print("bm25_documents", len(payload.get("documents", [])) if isinstance(payload, dict) else "legacy")

    client = MilvusClient(uri=config.MILVUS_URI)
    collections = client.list_collections()
    print("milvus_collections", collections)
    if config.MILVUS_TEXT_COLLECTION_NAME not in collections:
        print("milvus_text_exists", False)
    else:
        print_collection_status(client, config.MILVUS_TEXT_COLLECTION_NAME, "milvus_text")

    if config.MILVUS_IMAGE_COLLECTION_NAME not in collections:
        print("milvus_image_exists", False)
    else:
        print_collection_status(client, config.MILVUS_IMAGE_COLLECTION_NAME, "milvus_image")


def print_collection_status(client: MilvusClient, collection_name: str, label: str) -> None:
    print(f"{label}_exists", True)
    client.flush(collection_name)
    print(f"{label}_stats", client.get_collection_stats(collection_name))
    client.load_collection(collection_name)
    sample = client.query(
        collection_name=collection_name,
        filter='id != ""',
        output_fields=["id"],
        limit=5,
    )
    print(f"{label}_sample_count", len(sample))
    print(f"{label}_sample_ids", [row.get("id") for row in sample])
    all_rows = client.query(
        collection_name=collection_name,
        filter='id != ""',
        output_fields=["id"],
        limit=2000,
    )
    ids = [str(row.get("id") or "") for row in all_rows]
    duplicates = [item for item in Counter(ids).items() if item[1] > 1]
    print(f"{label}_query_rows", len(all_rows))
    print(f"{label}_query_unique_ids", len(set(ids)))
    print(f"{label}_duplicate_ids", duplicates[:10])


if __name__ == "__main__":
    main()
