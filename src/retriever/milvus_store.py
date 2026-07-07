from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

import config as settings


VECTOR_FIELD = "vector"
CONTENT_FIELD = "page_content"
METADATA_FIELD = "metadata"
MAX_CONTENT_LENGTH = 60000


@lru_cache(maxsize=1)
def get_milvus_client():
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError(
            "pymilvus is required. Install it in rag_env before building Milvus indexes."
        ) from exc

    kwargs: dict[str, Any] = {"uri": settings.MILVUS_URI}
    if settings.MILVUS_TOKEN:
        kwargs["token"] = settings.MILVUS_TOKEN
    return MilvusClient(**kwargs)


def reset_collection(collection_name: str, dimension: int) -> None:
    client = get_milvus_client()
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    create_collection(collection_name, dimension)


def create_collection(collection_name: str, dimension: int) -> None:
    from pymilvus import DataType

    client = get_milvus_client()
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(CONTENT_FIELD, DataType.VARCHAR, max_length=65535)
    schema.add_field(METADATA_FIELD, DataType.JSON)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD,
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def upsert_documents(
    collection_name: str,
    documents: list[Document],
    embeddings: list[list[float]],
) -> None:
    client = get_milvus_client()
    rows = []
    for document, embedding in zip(documents, embeddings):
        product_id = str(document.metadata.get("id") or document.id or "")
        if not product_id:
            continue
        rows.append(
            {
                "id": product_id,
                VECTOR_FIELD: [float(value) for value in embedding],
                CONTENT_FIELD: _truncate(document.page_content),
                METADATA_FIELD: _json_metadata(document.metadata or {}),
            }
        )

    if rows:
        client.upsert(collection_name=collection_name, data=rows)


def search_documents(
    collection_name: str,
    query_embedding: list[float],
    top_k: int,
) -> list[tuple[Document, float | None]]:
    client = get_milvus_client()
    _load_collection(collection_name)
    results = client.search(
        collection_name=collection_name,
        data=[[float(value) for value in query_embedding]],
        anns_field=VECTOR_FIELD,
        limit=top_k,
        output_fields=[CONTENT_FIELD, METADATA_FIELD],
        search_params={"metric_type": "COSINE"},
    )

    documents: list[tuple[Document, float | None]] = []
    for hit in results[0] if results else []:
        entity = hit.get("entity") or {}
        metadata = dict(entity.get(METADATA_FIELD) or {})
        product_id = str(hit.get("id") or metadata.get("id") or "")
        if product_id:
            metadata["id"] = metadata.get("id") or product_id
        documents.append(
            (
                Document(
                    page_content=str(entity.get(CONTENT_FIELD) or ""),
                    metadata=metadata,
                    id=product_id or None,
                ),
                float(hit["distance"]) if "distance" in hit else None,
            )
        )
    return documents


def count_collection(collection_name: str) -> int:
    client = get_milvus_client()
    if not client.has_collection(collection_name):
        return 0
    stats = client.get_collection_stats(collection_name)
    return int(stats.get("row_count") or stats.get("num_rows") or 0)


def _load_collection(collection_name: str) -> None:
    try:
        get_milvus_client().load_collection(collection_name)
    except Exception:
        pass


def _truncate(value: str) -> str:
    text = value or ""
    return text[:MAX_CONTENT_LENGTH]


def _json_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
    return cleaned
