"""
Build a CLIP image index for text-to-image product retrieval.

Product side: local image file -> CLIP image embedding.
Query side: user text -> multilingual CLIP text embedding.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as settings  # noqa: E402
from src.database.product_repo import ProductRepo  # noqa: E402
from src.retriever.milvus_store import (  # noqa: E402
    count_collection,
    reset_collection,
    upsert_documents,
)

logger = logging.getLogger(__name__)
IMAGE_MANIFEST_PATH = settings.DATA_DIR / "image_manifest.csv"


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
def load_clip_image_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        resolve_local_model_path(settings.CLIP_IMAGE_MODEL_NAME),
        cache_folder=str(settings.CLIP_MODEL_CACHE_DIR),
        local_files_only=settings.CLIP_LOCAL_FILES_ONLY,
    )


def load_local_image(image_path: Path) -> Image.Image | None:
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.warning("Skip local image %s: %s", image_path, exc)
        return None


def _metadata(row: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {}
    for key, value in row.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    return metadata


def _fetch_products(limit: int | None = None) -> list[dict[str, Any]]:
    repo = ProductRepo()
    sql = "SELECT * FROM products WHERE image_url IS NOT NULL AND image_url != '' ORDER BY id"
    params: list[Any] = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [dict(row) for row in repo.conn.execute(sql, params).fetchall()]


def _load_image_manifest() -> dict[str, Path]:
    if not IMAGE_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Image manifest not found: {IMAGE_MANIFEST_PATH}")

    image_paths: dict[str, Path] = {}
    with IMAGE_MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("status") not in {"downloaded", "exists"}:
                continue
            product_id = row.get("product_id")
            image_path = row.get("image_path")
            if not product_id or not image_path:
                continue

            normalized_image_path = image_path.replace("\\", "/")
            path = Path(normalized_image_path)
            if not path.is_absolute():
                path = settings.PROJECT_ROOT / path
            image_paths[product_id] = path

    return image_paths


def build_image_index(limit: int | None = None, reset: bool = True) -> int:
    image_paths = _load_image_manifest()
    products = [
        product
        for product in _fetch_products(limit=limit)
        if str(product.get("id")) in image_paths
    ]
    if not products:
        logger.warning("No products with local images found")
        return 0

    model = load_clip_image_model()

    indexed = 0
    collection_ready = not reset
    for product in products:
        product_id = str(product["id"])
        image_path = image_paths[product_id]
        image = load_local_image(image_path)
        if image is None:
            continue

        embedding = model.encode(image, normalize_embeddings=True).tolist()
        metadata = _metadata(product)
        metadata["image_path"] = str(image_path.relative_to(settings.PROJECT_ROOT))
        if not collection_ready:
            reset_collection(settings.MILVUS_IMAGE_COLLECTION_NAME, len(embedding))
            collection_ready = True
        upsert_documents(
            settings.MILVUS_IMAGE_COLLECTION_NAME,
            [
                Document(
                    page_content=product.get("content_text")
                    or product.get("title")
                    or "",
                    metadata=metadata,
                    id=product_id,
                )
            ],
            [embedding],
        )
        indexed += 1

        if indexed % 50 == 0:
            logger.info("Indexed image embeddings: %s/%s", indexed, len(products))

    logger.info(
        "Built Milvus image index: %s/%s products indexed in collection '%s' (count=%s)",
        indexed,
        len(products),
        settings.MILVUS_IMAGE_COLLECTION_NAME,
        count_collection(settings.MILVUS_IMAGE_COLLECTION_NAME),
    )
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_image_index(limit=args.limit, reset=not args.no_reset)


if __name__ == "__main__":
    main()
