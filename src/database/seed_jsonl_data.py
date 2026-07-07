"""
Seed SQLite products from the processed JSONL dataset.

The SQLite table keeps fields aligned with the JSONL data while retaining the
generic columns used by the existing retriever.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as settings  # noqa: E402
from src.database.connection import close_connection, get_connection  # noqa: E402
from src.database.product_repo import ProductRepo  # noqa: E402

logger = logging.getLogger(__name__)


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _display_text(value: Any, separator: str = " ") -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return separator.join(_display_text(item, separator) for item in value if item not in (None, ""))
    if isinstance(value, dict):
        return " ".join(f"{key} {_display_text(val, separator)}" for key, val in value.items())
    return str(value).strip()


def _first_text(*values: Any) -> str:
    for value in values:
        text = _display_text(value)
        if text:
            return text
    return ""


def _category_parts(item: dict[str, Any]) -> tuple[str, str, str, str]:
    category = _display_text(item.get("category"))
    parts = [part.strip() for part in category.split(" / ") if part.strip()]

    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    fallback = [
        _display_text(raw.get("doc_industry_name")),
        _display_text(raw.get("doc_cate1_name")),
        _display_text(raw.get("doc_cate2_name")),
        _display_text(raw.get("doc_cate3_name") or raw.get("doc_cate4_name")),
    ]

    while len(parts) < 4:
        parts.append(fallback[len(parts)] if len(parts) < len(fallback) else "")

    return tuple(parts[:4])  # type: ignore[return-value]


def build_content_text(item: dict[str, Any], product: dict[str, Any]) -> str:
    fields = [
        ("title", product.get("title")),
        ("description", item.get("description")),
        ("category_l1", product.get("category_l1")),
        ("category_l2", product.get("category_l2")),
        ("category_l3", product.get("category_l3")),
        ("category_l4", product.get("category_l4")),
        ("brand", product.get("brand")),
        ("shoe_type", item.get("shoe_type")),
        ("heel_type", item.get("heel_type")),
        ("closure_type", item.get("closure_type")),
        ("color", item.get("color")),
        ("material", item.get("material")),
        ("season", item.get("season")),
        ("style", item.get("style")),
        ("gender", item.get("gender")),
        ("target_user", item.get("target_user")),
        ("usage_scene", item.get("usage_scene")),
        ("functionality", item.get("functionality")),
        ("tags", item.get("tags")),
        ("attributes", item.get("attributes")),
    ]
    return "\n".join(
        f"{label}: {_display_text(value)}"
        for label, value in fields
        if _display_text(value)
    )


def convert_item(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    category_l1, category_l2, category_l3, category_l4 = _category_parts(item)

    product = {
        "id": _first_text(item.get("product_id"), raw.get("id")),
        "title": _first_text(item.get("title"), raw.get("doc_title")),
        "image_url": _first_text(item.get("image_url"), raw.get("doc_image")),
        "category_l1": category_l1,
        "category_l2": category_l2,
        "category_l3": category_l3,
        "category_l4": category_l4,
        "source_dataset": _display_text(item.get("source_dataset")),
        "description": _display_text(item.get("description")),
        "brand": _display_text(item.get("brand")),
        "material": _json_text(item.get("material", [])),
        "season": _json_text(item.get("season", [])),
        "style": _json_text(item.get("style", [])),
        "color": _json_text(item.get("color", [])),
        "shoe_type": _display_text(item.get("shoe_type")),
        "heel_type": _display_text(item.get("heel_type")),
        "closure_type": _display_text(item.get("closure_type")),
        "gender": _display_text(item.get("gender")),
        "target_user": _json_text(item.get("target_user", [])),
        "usage_scene": _json_text(item.get("usage_scene", [])),
        "functionality": _json_text(item.get("functionality", [])),
        "text_color": _json_text(item.get("text_color", [])),
        "image_color": _json_text(item.get("image_color", [])),
        "color_source": _json_text(item.get("color_source", [])),
        "color_confidence": _display_text(item.get("color_confidence")),
        "color_conflict": 1 if item.get("color_conflict") else 0,
        "color_detail": _json_text(item.get("color_detail", {})),
        "tags": _json_text(item.get("tags", [])),
        "rating": round(random.uniform(3.8, 5.0), 1),
        "sales_count": int(random.expovariate(1 / 400) + 20),
        "stock_status": random.choices(["有货", "预售", "售罄"], weights=[0.85, 0.10, 0.05])[0],
        "attributes_raw": _display_text(item.get("attributes", {})),
        "attributes_json": _json_text(item.get("attributes", {})),
        "raw_json": _json_text(raw),
        "content_text": "",
    }
    product["content_text"] = build_content_text(item, product)
    return product


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Processed JSONL dataset not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def reset_products_tables() -> None:
    conn = get_connection()
    conn.execute("DELETE FROM products")
    conn.execute("INSERT INTO products_fts(products_fts) VALUES ('rebuild')")
    conn.commit()


def seed_from_jsonl(path: Path | None = None, reset: bool = True) -> int:
    jsonl_path = path or settings.PROCESSED_JSONL_DATA_PATH
    items = load_jsonl(jsonl_path)
    products = [convert_item(item) for item in items]
    products = [product for product in products if product["id"] and product["title"]]

    repo = ProductRepo()
    if reset:
        reset_products_tables()

    inserted = repo.insert_batch(products)
    total = repo.count()
    logger.info("Seeded %s products from %s; database total=%s", inserted, jsonl_path, total)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        seed_from_jsonl(reset=True)
    finally:
        close_connection()
