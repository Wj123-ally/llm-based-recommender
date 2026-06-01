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
    price           REAL,
    brand           TEXT,
    material        TEXT,
    season          TEXT,
    style           TEXT,
    color           TEXT,
    gender          TEXT,
    rating          REAL,
    sales_count     INTEGER,
    stock_status    TEXT    DEFAULT '有货',
    attributes_raw  TEXT,
    content_text    TEXT    NOT NULL,
    created_at      TEXT    DEFAULT (datetime('now')),
    updated_at      TEXT    DEFAULT (datetime('now'))
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_products_category_l1 ON products(category_l1);",
    "CREATE INDEX IF NOT EXISTS idx_products_category_l2 ON products(category_l2);",
    "CREATE INDEX IF NOT EXISTS idx_products_price ON products(price);",
    "CREATE INDEX IF NOT EXISTS idx_products_rating ON products(rating DESC);",
    "CREATE INDEX IF NOT EXISTS idx_products_season ON products(season);",
    "CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand);",
    "CREATE INDEX IF NOT EXISTS idx_products_material ON products(material);",
    "CREATE INDEX IF NOT EXISTS idx_products_gender ON products(gender);",
    "CREATE INDEX IF NOT EXISTS idx_products_stock ON products(stock_status);",
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

def init_db() -> None:
    """
    初始化数据库：建表 + 索引 + FTS5。

    幂等操作 —— 表已存在时跳过。
    """
    conn = get_connection()
    cursor = conn.cursor()

    logger.info("创建 products 主表 ...")
    cursor.execute(CREATE_PRODUCTS_TABLE)

    logger.info("创建索引 ...")
    for index_sql in CREATE_INDEXES:
        cursor.execute(index_sql)

    logger.info("创建 FTS5 全文搜索表及触发器 ...")
    cursor.execute(CREATE_FTS_TABLE)
    for trigger_sql in CREATE_FTS_TRIGGERS:
        cursor.execute(trigger_sql)

    conn.commit()
    logger.info("数据库初始化完成")
