"""
建表 DDL 与索引创建。

表设计：
- products:       商品主表（结构化字段 + 检索文本）
- products_fts:   FTS5 全文搜索虚拟表（中文分词由 jieba 外部处理）
"""

import logging

from .connection import get_connection

logger = logging.getLogger(__name__)

CREATE_PRODUCTS_TABLE = """
CREATE TABLE IF NOT EXISTS products (
    id              TEXT PRIMARY KEY,
    title           TEXT    NOT NULL,
    image_url       TEXT,
    category_l1     TEXT    NOT NULL,
    category_l2     TEXT,
    category_l3     TEXT,
    category_l4     TEXT,
    source_dataset  TEXT,
    description     TEXT,
    brand           TEXT,
    material        TEXT,
    season          TEXT,
    style           TEXT,
    color           TEXT,
    shoe_type       TEXT,
    heel_type       TEXT,
    closure_type    TEXT,
    gender          TEXT,
    target_user     TEXT,
    usage_scene     TEXT,
    functionality   TEXT,
    text_color      TEXT,
    image_color     TEXT,
    color_source    TEXT,
    color_confidence TEXT,
    color_conflict  INTEGER,
    color_detail    TEXT,
    tags            TEXT,
    rating          REAL,
    sales_count     INTEGER,
    stock_status    TEXT    DEFAULT '有货',
    attributes_raw  TEXT,
    attributes_json TEXT,
    raw_json        TEXT,
    content_text    TEXT    NOT NULL,
    created_at      TEXT    DEFAULT (datetime('now')),
    updated_at      TEXT    DEFAULT (datetime('now'))
);
"""

PRODUCT_COLUMN_DEFINITIONS = {
    "source_dataset": "TEXT",
    "description": "TEXT",
    "shoe_type": "TEXT",
    "heel_type": "TEXT",
    "closure_type": "TEXT",
    "target_user": "TEXT",
    "usage_scene": "TEXT",
    "functionality": "TEXT",
    "text_color": "TEXT",
    "image_color": "TEXT",
    "color_source": "TEXT",
    "color_confidence": "TEXT",
    "color_conflict": "INTEGER",
    "color_detail": "TEXT",
    "tags": "TEXT",
    "attributes_json": "TEXT",
    "raw_json": "TEXT",
}

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_products_category_l1 ON products(category_l1);",
    "CREATE INDEX IF NOT EXISTS idx_products_category_l2 ON products(category_l2);",
    "CREATE INDEX IF NOT EXISTS idx_products_rating ON products(rating DESC);",
    "CREATE INDEX IF NOT EXISTS idx_products_season ON products(season);",
    "CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand);",
    "CREATE INDEX IF NOT EXISTS idx_products_material ON products(material);",
    "CREATE INDEX IF NOT EXISTS idx_products_gender ON products(gender);",
    "CREATE INDEX IF NOT EXISTS idx_products_stock ON products(stock_status);",
    "CREATE INDEX IF NOT EXISTS idx_products_shoe_type ON products(shoe_type);",
    "CREATE INDEX IF NOT EXISTS idx_products_color_confidence ON products(color_confidence);",
]

CREATE_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
    title,
    brand,
    material,
    style,
    content_text,
    content=products,
    content_rowid=rowid
);
"""

CREATE_FTS_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS products_ai AFTER INSERT ON products BEGIN
        INSERT INTO products_fts(rowid, title, brand, material, style, content_text)
        VALUES (new.rowid, new.title, new.brand, new.material, new.style, new.content_text);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS products_ad AFTER DELETE ON products BEGIN
        INSERT INTO products_fts(products_fts, rowid, title, brand, material, style, content_text)
        VALUES ('delete', old.rowid, old.title, old.brand, old.material, old.style, old.content_text);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS products_au AFTER UPDATE ON products BEGIN
        INSERT INTO products_fts(products_fts, rowid, title, brand, material, style, content_text)
        VALUES ('delete', old.rowid, old.title, old.brand, old.material, old.style, old.content_text);
        INSERT INTO products_fts(rowid, title, brand, material, style, content_text)
        VALUES (new.rowid, new.title, new.brand, new.material, new.style, new.content_text);
    END;
    """,
]


def _ensure_product_columns(cursor) -> None:
    existing_columns = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(products)").fetchall()
    }

    for column_name, definition in PRODUCT_COLUMN_DEFINITIONS.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE products ADD COLUMN {column_name} {definition}")


def init_db() -> None:
    """
    初始化数据库：建表 + 索引 + FTS5。

    幂等操作 —— 表已存在时跳过。
    """
    conn = get_connection()
    cursor = conn.cursor()

    logger.info("创建 products 主表 ...")
    cursor.execute(CREATE_PRODUCTS_TABLE)
    _ensure_product_columns(cursor)

    logger.info("创建索引 ...")
    for index_sql in CREATE_INDEXES:
        cursor.execute(index_sql)

    logger.info("创建 FTS5 全文搜索表及触发器 ...")
    cursor.execute(CREATE_FTS_TABLE)
    for trigger_sql in CREATE_FTS_TRIGGERS:
        cursor.execute(trigger_sql)

    conn.commit()
    logger.info("数据库初始化完成")
