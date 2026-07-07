from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
JSONL_PATH = ROOT / "src/indexing/data/data/processed/shoe_products_only_shoes_cleaned.jsonl"
DB_PATH = ROOT / "src/database/enriched_products.db"
IMAGE_MANIFEST_PATH = ROOT / "src/indexing/data/image_manifest.csv"

REQUIRED_FIELDS = [
    "product_id",
    "title",
    "description",
    "image_url",
    "category",
    "source_dataset",
    "brand",
    "color",
    "material",
    "shoe_type",
    "style",
    "season",
    "heel_type",
    "closure_type",
    "gender",
    "target_user",
    "usage_scene",
    "functionality",
    "tags",
    "attributes",
    "raw",
]
LIST_FIELDS = [
    "color",
    "material",
    "style",
    "season",
    "target_user",
    "usage_scene",
    "functionality",
    "tags",
]
TEXT_PROBE_TERMS = ["劳保鞋", "拖鞋", "女", "男", "夏", "冬", "板鞋", "皮鞋", "跑步鞋", "高跟鞋", "防滑", "透气"]
MOJIBAKE_PROBE_TERMS = ["鍔", "闉", "鐢", "濂", "鏃", "绌", "杞", "澶", "鍐", "鎷", "閫"]


def nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def walk_text(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from walk_text(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_text(item)


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    rows: list[dict[str, Any]] = []
    invalid: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid.append((line_number, str(exc)))
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows, invalid


def summarize_jsonl(rows: list[dict[str, Any]], invalid: list[tuple[int, str]]) -> dict[str, Any]:
    missing = Counter()
    empty = Counter()
    bad_type = Counter()
    cleaning_notes = Counter()
    final_check_rows = 0

    for row in rows:
        for field in REQUIRED_FIELDS:
            if field not in row:
                missing[field] += 1
            elif not nonempty(row.get(field)):
                empty[field] += 1

        for field in LIST_FIELDS:
            if field in row and nonempty(row.get(field)) and not isinstance(row.get(field), list):
                bad_type[field] += 1

        notes = row.get("cleaning_notes") or {}
        if isinstance(notes, dict):
            for field in notes:
                cleaning_notes[field] += 1
            if notes.get("final_check"):
                final_check_rows += 1

    ids = [str(row.get("product_id") or "") for row in rows]
    titles = [str(row.get("title") or "") for row in rows]
    all_text = [" ".join(walk_text(row)) for row in rows]

    return {
        "rows": len(rows),
        "invalid_json": len(invalid),
        "unique_ids": len(set(item for item in ids if item)),
        "duplicate_ids": sum(1 for _, count in Counter(ids).items() if count > 1),
        "duplicate_titles": sum(1 for _, count in Counter(titles).items() if count > 1),
        "missing_required": dict(missing),
        "empty_required": dict(empty),
        "bad_list_types": dict(bad_type),
        "normal_chinese_hit_rows": sum(any(term in text for term in TEXT_PROBE_TERMS) for text in all_text),
        "mojibake_hit_rows": sum(any(term in text for term in MOJIBAKE_PROBE_TERMS) for text in all_text),
        "replacement_char_count": sum(text.count("\ufffd") for text in all_text),
        "cleaning_notes_top": cleaning_notes.most_common(15),
        "final_check_rows": final_check_rows,
        "gender": Counter(str(row.get("gender") or "") for row in rows).most_common(),
        "shoe_type_top": Counter(str(row.get("shoe_type") or "") for row in rows).most_common(15),
        "season_top": Counter(
            item for row in rows for item in (row.get("season") or [])
        ).most_common(15),
    }


def summarize_sqlite(json_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not DB_PATH.exists():
        return {"exists": False}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    db_rows = [dict(row) for row in conn.execute("SELECT * FROM products").fetchall()]
    json_by_id = {str(row.get("product_id")): row for row in json_rows}

    empty = Counter()
    for row in db_rows:
        for field in [
            "title",
            "description",
            "image_url",
            "brand",
            "material",
            "season",
            "style",
            "color",
            "shoe_type",
            "heel_type",
            "closure_type",
            "gender",
            "target_user",
            "usage_scene",
            "functionality",
            "tags",
            "content_text",
        ]:
            if not nonempty(row.get(field)):
                empty[field] += 1

    changed = Counter()
    filled = Counter()
    emptied = Counter()
    examples: list[tuple[str, str, str, str]] = []
    for db_row in db_rows:
        json_row = json_by_id.get(str(db_row.get("id")))
        if not json_row:
            continue
        for field in ["brand", "shoe_type", "heel_type", "closure_type", "gender"]:
            old = str(db_row.get(field) or "")
            new = str(json_row.get(field) or "")
            if old == new:
                continue
            changed[field] += 1
            if not old and new:
                filled[field] += 1
            if old and not new:
                emptied[field] += 1
            if len(examples) < 12:
                examples.append((str(db_row.get("id")), field, old, new))

    all_text = [" ".join(str(value or "") for value in row.values()) for row in db_rows]
    json_ids = {str(row.get("product_id")) for row in json_rows}
    db_ids = {str(row.get("id")) for row in db_rows}

    return {
        "exists": True,
        "rows": len(db_rows),
        "ids_overlap": len(json_ids & db_ids),
        "only_jsonl": len(json_ids - db_ids),
        "only_sqlite": len(db_ids - json_ids),
        "empty_fields": dict(empty),
        "mojibake_hit_rows": sum(any(term in text for term in MOJIBAKE_PROBE_TERMS) for text in all_text),
        "replacement_char_count": sum(text.count("\ufffd") for text in all_text),
        "gender": Counter(str(row.get("gender") or "") for row in db_rows).most_common(),
        "shoe_type_top": Counter(str(row.get("shoe_type") or "") for row in db_rows).most_common(15),
        "jsonl_vs_sqlite_changed": dict(changed),
        "jsonl_filled_where_sqlite_empty": dict(filled),
        "jsonl_emptied_where_sqlite_had_value": dict(emptied),
        "examples": examples,
    }


def summarize_image_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not IMAGE_MANIFEST_PATH.exists():
        return {"exists": False}

    with IMAGE_MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        manifest_rows = list(csv.DictReader(file))
    usable_ids = {
        row.get("product_id")
        for row in manifest_rows
        if row.get("status") in {"downloaded", "exists"}
    }
    json_ids = {str(row.get("product_id")) for row in rows}
    return {
        "exists": True,
        "manifest_rows": len(manifest_rows),
        "status": dict(Counter(row.get("status", "") for row in manifest_rows)),
        "usable_images": len(usable_ids),
        "current_covered": len(json_ids & usable_ids),
        "current_total": len(json_ids),
    }


def main() -> None:
    rows, invalid = load_jsonl(JSONL_PATH)

    print("=== CURRENT_JSONL ===")
    print("path", JSONL_PATH)
    for key, value in summarize_jsonl(rows, invalid).items():
        print(key, value)
    print("sample_titles")
    for row in rows[:5]:
        print(row.get("product_id"), str(row.get("title") or "")[:100])

    print("=== SQLITE_LOADED_DATA ===")
    for key, value in summarize_sqlite(rows).items():
        print(key, value)

    print("=== IMAGE_COVERAGE ===")
    for key, value in summarize_image_coverage(rows).items():
        print(key, value)


if __name__ == "__main__":
    main()
