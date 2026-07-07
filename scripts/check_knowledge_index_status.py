from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pymilvus import MilvusClient


def main() -> None:
    client = MilvusClient(uri=config.MILVUS_URI)
    collection_name = config.MILVUS_KNOWLEDGE_COLLECTION_NAME
    collections = client.list_collections()
    print("milvus_collections", collections)
    print("knowledge_exists", collection_name in collections)
    if collection_name not in collections:
        return

    client.flush(collection_name)
    print("knowledge_stats", client.get_collection_stats(collection_name))
    client.load_collection(collection_name)
    rows = client.query(
        collection_name=collection_name,
        filter='id != ""',
        output_fields=["id", "metadata"],
        limit=1000,
    )
    ids = [str(row.get("id") or "") for row in rows]
    filenames = [
        str((row.get("metadata") or {}).get("source_filename") or "")
        for row in rows
    ]
    print("knowledge_query_rows", len(rows))
    print("knowledge_unique_ids", len(set(ids)))
    print("knowledge_duplicate_ids", [item for item in Counter(ids).items() if item[1] > 1][:10])
    print("knowledge_files", Counter(filenames).most_common())
    print("knowledge_sample", [(row.get("id"), (row.get("metadata") or {}).get("source_filename")) for row in rows[:10]])


if __name__ == "__main__":
    main()
