"""Fill missing product fields from existing product text.

The script is intentionally rule-based and conservative:
- only fills empty fields;
- never overwrites existing cleaned values;
- updates both SQLite and the processed JSONL source;
- rebuilds content_text from the updated fields.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as settings  # noqa: E402

DB_PATH = settings.DATABASE_PATH
JSONL_PATH = settings.PROCESSED_JSONL_DATA_PATH


LIST_FIELDS = {
    "material",
    "season",
    "style",
    "target_user",
    "usage_scene",
    "functionality",
    "text_color",
    "image_color",
    "color_source",
    "tags",
}


RULES: dict[str, list[tuple[str, str]]] = {
    "shoe_type": [
        ("运动鞋", "运动鞋"), ("跑步鞋", "跑步鞋"), ("跑鞋", "跑步鞋"),
        ("板鞋", "板鞋"), ("皮鞋", "皮鞋"), ("乐福鞋", "乐福鞋"),
        ("拖鞋", "拖鞋"), ("凉拖", "拖鞋"), ("凉鞋", "凉鞋"),
        ("马丁靴", "马丁靴"), ("雪地靴", "雪地靴"), ("雨鞋", "雨鞋"),
        ("帆布鞋", "帆布鞋"), ("高跟鞋", "高跟鞋"), ("劳保鞋", "劳保鞋"),
        ("工作鞋", "工作鞋"), ("登山鞋", "登山鞋"), ("徒步鞋", "徒步鞋"),
        ("篮球鞋", "篮球鞋"), ("训练鞋", "训练鞋"), ("足球鞋", "足球鞋"),
        ("舞蹈鞋", "舞蹈鞋"), ("骑行鞋", "骑行鞋"), ("棉拖", "拖鞋"),
        ("单鞋", "单鞋"), ("小白鞋", "运动鞋"), ("老爹鞋", "运动鞋"),
        ("豆豆鞋", "豆豆鞋"), ("布鞋", "布鞋"),
    ],
    "material": [
        ("头层牛皮", "头层牛皮"), ("牛皮", "牛皮"), ("真皮", "真皮"),
        ("软皮", "软皮"), ("帆布", "帆布"), ("网面", "网面"),
        ("透气网面", "网面"), ("麂皮", "麂皮"), ("绒面", "绒面"),
        ("翻毛皮", "翻毛皮"), ("橡胶底", "橡胶底"), ("橡胶", "橡胶"),
        ("EVA", "EVA"), ("eva", "EVA"), ("加绒", "加绒"),
        ("羊毛", "羊毛"), ("软底", "软底"),
    ],
    "season": [
        ("春夏", "春夏"), ("秋冬", "秋冬"), ("春秋", "春秋"),
        ("四季", "四季"), ("夏季", "夏季"), ("冬季", "冬季"),
        ("春季", "春季"), ("秋季", "秋季"), ("夏天", "夏季"),
        ("冬天", "冬季"), ("春款", "春季"), ("夏款", "夏季"),
        ("秋款", "秋季"), ("冬款", "冬季"),
    ],
    "style": [
        ("休闲", "休闲"), ("商务", "商务"), ("正装", "正装"),
        ("英伦", "英伦"), ("复古", "复古"), ("潮流", "潮流"),
        ("时尚", "时尚"), ("百搭", "百搭"), ("仙女", "仙女"),
        ("学院", "学院"), ("工装", "工装"), ("运动", "运动"),
    ],
    "functionality": [
        ("透气", "透气"), ("防滑", "防滑"), ("防水", "防水"),
        ("保暖", "保暖"), ("轻便", "轻便"), ("舒适", "舒适"),
        ("增高", "增高"), ("显瘦", "显瘦"), ("耐磨", "耐磨"),
        ("防砸", "防砸"), ("防刺穿", "防刺穿"), ("缓震", "缓震"),
        ("减震", "缓震"), ("防臭", "防臭"),
    ],
    "heel_type": [
        ("厚底", "厚底"), ("平底", "平底"), ("高跟", "高跟"),
        ("低跟", "低跟"), ("中跟", "中跟"), ("粗跟", "粗跟"),
        ("细跟", "细跟"), ("坡跟", "坡跟"), ("松糕", "厚底"),
    ],
    "closure_type": [
        ("系带", "系带"), ("一脚蹬", "一脚蹬"), ("拉链", "拉链"),
        ("魔术贴", "魔术贴"), ("套脚", "套脚"), ("搭扣", "搭扣"),
        ("扣带", "搭扣"), ("鞋带", "系带"), ("懒人鞋", "一脚蹬"),
    ],
    "target_user": [
        ("儿童", "儿童"), ("女童", "儿童"), ("男童", "儿童"),
        ("学生", "学生"), ("妈妈", "妈妈"), ("老人", "老人"),
        ("老年", "老人"), ("爸爸", "老人"), ("情侣", "情侣"),
        ("女鞋", "女"), ("女士", "女"), ("女款", "女"), ("女 ", "女"),
        ("男鞋", "男"), ("男士", "男"), ("男款", "男"), ("男 ", "男"),
    ],
    "usage_scene": [
        ("跑步", "跑步"), ("户外", "户外"), ("居家", "居家"),
        ("浴室", "浴室"), ("洗澡", "浴室"), ("通勤", "通勤"),
        ("工地", "工地"), ("电焊", "工地"), ("婚礼", "婚礼"),
        ("结婚", "婚礼"), ("正装", "正装"), ("校园", "校园"),
        ("学生", "校园"), ("健身", "健身"), ("训练", "健身"),
        ("登山", "户外"), ("徒步", "户外"), ("篮球", "运动"),
        ("足球", "运动"), ("运动", "运动"), ("商务", "通勤"),
        ("休闲", "日常"), ("百搭", "日常"), ("外穿", "日常"),
    ],
    "text_color": [
        ("黑白", "拼色"), ("蓝白", "拼色"), ("撞色", "拼色"), ("拼色", "拼色"),
        ("黑色", "黑色"), ("劲黑", "黑色"), ("小黑", "黑色"),
        ("白色", "白色"), ("小白鞋", "白色"),
        ("米白", "米色"), ("米色", "米色"), ("杏色", "米色"),
        ("棕色", "棕色"), ("咖色", "棕色"), ("卡其", "棕色"),
        ("灰色", "灰色"), ("烟灰", "灰色"),
        ("红色", "红色"), ("粉色", "粉色"), ("绿色", "绿色"),
        ("蓝色", "蓝色"), ("黄色", "黄色"), ("金色", "黄色"),
        ("紫色", "紫色"), ("橙色", "橙色"),
    ],
}


def has_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text and text not in {"[]", "{}", "null", "None", "nan"})


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not has_value(value):
        return [] 
    try:
        return json.loads(str(value))
    except Exception:
        return value


def to_json_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def display(value: Any) -> str:
    value = parse_jsonish(value)
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(display(item) for item in value if has_value(item))
    if isinstance(value, dict):
        return " ".join(f"{key} {display(val)}" for key, val in value.items())
    return str(value).strip()


def existing_list(value: Any) -> list[str]:
    parsed = parse_jsonish(value)
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if has_value(item)]
    if has_value(parsed):
        return [str(parsed).strip()]
    return []


def collect_text(product: dict[str, Any]) -> str:
    raw = parse_jsonish(product.get("raw_json") or product.get("raw") or {})
    attrs = parse_jsonish(product.get("attributes_json") or product.get("attributes") or {})
    parts = [
        product.get("title"),
        product.get("description"),
        product.get("attributes_raw"),
        display(attrs),
    ]
    if isinstance(raw, dict):
        parts.extend([raw.get("doc_attributes"), raw.get("query_title"), raw.get("doc_title")])
    return " ".join(display(part) for part in parts if has_value(part)).lower()


def infer_values(field: str, text: str) -> list[str]:
    values: list[str] = []
    for term, normalized in RULES[field]:
        if term.lower() in text and normalized not in values:
            values.append(normalized)
    return values


def infer_gender(text: str) -> str:
    if any(term in text for term in ["女童", "公主", "女鞋", "女士", "女款", "女 "]):
        return "女"
    if any(term in text for term in ["男童", "男鞋", "男士", "男款", "男 "]):
        return "男"
    if "儿童" in text or "童鞋" in text:
        return "儿童"
    if "情侣" in text:
        return "情侣"
    return ""


def infer_brand(product: dict[str, Any]) -> str:
    attrs = parse_jsonish(product.get("attributes_json") or product.get("attributes") or {})
    if isinstance(attrs, dict):
        brand = attrs.get("品牌")
        text = display(brand)
        if text:
            return text
    return ""


def default_usage_scene(product: dict[str, Any]) -> list[str]:
    text = collect_text(product)
    shoe_type = display(product.get("shoe_type"))
    values: list[str] = []
    mappings = [
        (["跑步鞋", "跑鞋"], "跑步"),
        (["运动鞋", "训练鞋", "篮球鞋", "足球鞋"], "运动"),
        (["拖鞋", "凉鞋"], "居家"),
        (["皮鞋", "乐福鞋"], "通勤"),
        (["马丁靴", "雪地靴", "登山鞋", "徒步鞋", "雨鞋"], "户外"),
        (["劳保鞋", "工作鞋"], "工地"),
        (["高跟鞋"], "正装"),
        (["板鞋", "帆布鞋", "单鞋", "豆豆鞋", "布鞋"], "日常"),
    ]
    for terms, value in mappings:
        if any(term in shoe_type or term in text for term in terms) and value not in values:
            values.append(value)
    return values


def default_closure_type(product: dict[str, Any]) -> str:
    text = collect_text(product)
    shoe_type = display(product.get("shoe_type"))
    if any(term in text for term in ["拖鞋", "凉拖", "棉拖", "一字拖"]):
        return "套脚"
    if any(term in shoe_type for term in ["拖鞋"]):
        return "套脚"
    if any(term in text for term in ["乐福鞋", "豆豆鞋", "懒人鞋", "一脚蹬"]):
        return "一脚蹬"
    if any(term in shoe_type for term in ["乐福鞋", "豆豆鞋"]):
        return "一脚蹬"
    if any(term in text for term in ["板鞋", "运动鞋", "跑步鞋", "跑鞋", "帆布鞋", "篮球鞋", "足球鞋"]):
        return "系带"
    return ""


def default_target_user(product: dict[str, Any]) -> list[str]:
    text = collect_text(product)
    values = infer_values("target_user", text)
    gender = display(product.get("gender"))
    if gender in {"男", "女", "儿童", "情侣"} and gender not in values:
        values.append(gender)
    return values


def build_content_text(product: dict[str, Any]) -> str:
    fields = [
        ("title", product.get("title")),
        ("description", product.get("description")),
        ("category_l1", product.get("category_l1")),
        ("category_l2", product.get("category_l2")),
        ("category_l3", product.get("category_l3")),
        ("category_l4", product.get("category_l4")),
        ("brand", product.get("brand")),
        ("shoe_type", product.get("shoe_type")),
        ("heel_type", product.get("heel_type")),
        ("closure_type", product.get("closure_type")),
        ("color", product.get("color")),
        ("text_color", product.get("text_color")),
        ("image_color", product.get("image_color")),
        ("material", product.get("material")),
        ("season", product.get("season")),
        ("style", product.get("style")),
        ("gender", product.get("gender")),
        ("target_user", product.get("target_user")),
        ("usage_scene", product.get("usage_scene")),
        ("functionality", product.get("functionality")),
        ("tags", product.get("tags")),
        ("attributes", product.get("attributes_raw") or product.get("attributes_json") or product.get("attributes")),
    ]
    return "\n".join(f"{label}: {display(value)}" for label, value in fields if display(value))


def enrich_product(product: dict[str, Any]) -> tuple[dict[str, Any], int]:
    updated = dict(product)
    changes = 0
    text = collect_text(updated)

    for field in ["material", "season", "style", "functionality", "target_user", "usage_scene", "text_color"]:
        if not existing_list(updated.get(field)):
            values = infer_values(field, text)
            if values:
                updated[field] = values
                changes += 1

    if not existing_list(updated.get("target_user")):
        values = default_target_user(updated)
        if values:
            updated["target_user"] = values
            changes += 1

    if not existing_list(updated.get("usage_scene")):
        values = default_usage_scene(updated)
        if values:
            updated["usage_scene"] = values
            changes += 1

    for field in ["shoe_type", "heel_type", "closure_type"]:
        if not has_value(updated.get(field)):
            values = infer_values(field, text)
            if values:
                updated[field] = values[0]
                changes += 1

    if not has_value(updated.get("closure_type")):
        value = default_closure_type(updated)
        if value:
            updated["closure_type"] = value
            changes += 1

    if not has_value(updated.get("gender")):
        gender = infer_gender(text)
        if gender:
            updated["gender"] = gender
            changes += 1

    if not has_value(updated.get("brand")):
        brand = infer_brand(updated)
        if brand:
            updated["brand"] = brand
            changes += 1

    updated["content_text"] = build_content_text(updated)
    return updated, changes


def db_to_product(row: sqlite3.Row) -> dict[str, Any]:
    product = dict(row)
    for field in LIST_FIELDS:
        product[field] = existing_list(product.get(field))
    return product


def product_to_db(product: dict[str, Any]) -> dict[str, Any]:
    db_product = dict(product)
    for field in LIST_FIELDS:
        db_product[field] = to_json_text(existing_list(db_product.get(field)))
    return db_product


def update_sqlite() -> tuple[int, int]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM products").fetchall()
    updated_count = 0
    field_changes = 0
    update_columns = [
        "brand", "material", "season", "style", "shoe_type", "heel_type",
        "closure_type", "gender", "target_user", "usage_scene",
        "functionality", "text_color", "content_text",
    ]
    with conn:
        for row in rows:
            enriched, changes = enrich_product(db_to_product(row))
            if changes:
                field_changes += changes
                updated_count += 1
                db_product = product_to_db(enriched)
                assignments = ", ".join(f"{col}=?" for col in update_columns)
                conn.execute(
                    f"UPDATE products SET {assignments}, updated_at=datetime('now') WHERE id=?",
                    [db_product.get(col) for col in update_columns] + [db_product["id"]],
                )
        try:
            conn.execute("INSERT INTO products_fts(products_fts) VALUES ('rebuild')")
        except sqlite3.OperationalError:
            pass
    conn.close()
    return updated_count, field_changes


def update_jsonl() -> tuple[int, int]:
    rows = [
        json.loads(line)
        for line in JSONL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    updated_count = 0
    field_changes = 0
    output = []
    for item in rows:
        item.pop("price", None)
        item.pop("currency", None)
        item.pop("size", None)
        normalized = dict(item)
        normalized["id"] = item.get("product_id") or item.get("id")
        normalized["category_l1"], normalized["category_l2"], normalized["category_l3"], normalized["category_l4"] = (
            [part.strip() for part in str(item.get("category", "")).split(" / ") if part.strip()] + ["", "", "", ""]
        )[:4]
        normalized["attributes_json"] = item.get("attributes", {})
        normalized["raw_json"] = item.get("raw", {})
        normalized["attributes_raw"] = display(item.get("attributes", {}))

        enriched, changes = enrich_product(normalized)
        if changes:
            updated_count += 1
            field_changes += changes
        for field in [
            "brand", "material", "season", "style", "shoe_type", "heel_type",
            "closure_type", "gender", "target_user", "usage_scene", "functionality",
            "text_color",
        ]:
            if field in LIST_FIELDS:
                item[field] = existing_list(enriched.get(field))
            else:
                item[field] = enriched.get(field, "")
        item["description"] = enriched.get("description", item.get("description", ""))
        output.append(item)

    JSONL_PATH.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in output),
        encoding="utf-8",
    )
    return updated_count, field_changes


def main() -> None:
    sqlite_rows, sqlite_changes = update_sqlite()
    jsonl_rows, jsonl_changes = update_jsonl()
    print(f"sqlite_updated_rows={sqlite_rows} sqlite_field_changes={sqlite_changes}")
    print(f"jsonl_updated_rows={jsonl_rows} jsonl_field_changes={jsonl_changes}")


if __name__ == "__main__":
    main()
