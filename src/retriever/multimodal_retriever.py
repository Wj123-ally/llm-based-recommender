"""
Text-to-image product retrieval using CLIP embeddings.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as settings  # noqa: E402

logger = logging.getLogger(__name__)


def resolve_local_model_path(model_name: str) -> str:
    snapshot_root = (
        settings.CLIP_MODEL_CACHE_DIR
        / "hub"
        / f"models--{model_name.replace('/', '--')}"
        / "snapshots"
    )
    if snapshot_root.exists():
        for snapshot in snapshot_root.iterdir():
            if (snapshot / "modules.json").exists():
                return str(snapshot)
    return model_name


@lru_cache(maxsize=1)
def load_clip_text_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        resolve_local_model_path(settings.CLIP_TEXT_MODEL_NAME),
        cache_folder=str(settings.CLIP_MODEL_CACHE_DIR),
        local_files_only=settings.CLIP_LOCAL_FILES_ONLY,
    )


@lru_cache(maxsize=1)
def load_multimodal_collection_name() -> str:
    return settings.MILVUS_IMAGE_COLLECTION_NAME


def retrieve_from_product_images(query: str, top_k: int | None = None) -> list[tuple[Document, float | None]]:
    if top_k is None:
        top_k = settings.MULTIMODAL_RETRIEVER_TOP_K

    text_model = load_clip_text_model()
    query_embedding = text_model.encode(query, normalize_embeddings=True).tolist()
    from src.retriever.milvus_store import search_documents

    return search_documents(
        load_multimodal_collection_name(),
        query_embedding,
        top_k,
    )
