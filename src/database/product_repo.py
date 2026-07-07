"""
商品数据访问层。

提供：
- 批量插入 / 更新
- 按条件过滤查询（价格区间、品牌、材质、季节、类目、评分）
- FTS5 关键词搜索
- 组合查询（向量检索 ID 列表 + SQL 条件过滤）
"""

import sqlite3
from typing import Any, Optional

from .connection import get_connection
from .schema import init_db


class ProductRepo:
    """商品仓库。"""

    def __init__(self) -> None:
        init_db()
        self.conn = get_connection()

    # ── 写入 ───────────────────────────────────────────

    def insert_batch(self, products: list[dict[str, Any]]) -> int:
        """
        批量插入商品。已存在的 ID 会被跳过（INSERT OR IGNORE）。

        Returns:
            实际插入的行数。
        """
        if not products:
            return 0

        columns = [
            "id", "title", "image_url",
            "category_l1", "category_l2", "category_l3", "category_l4",
            "source_dataset", "description",
            "brand", "material", "season", "style", "color",
            "shoe_type", "heel_type", "closure_type", "gender",
            "target_user", "usage_scene", "functionality",
            "text_color", "image_color", "color_source",
            "color_confidence", "color_conflict", "color_detail", "tags",
            "rating", "sales_count", "stock_status",
            "attributes_raw", "attributes_json", "raw_json", "content_text",
        ]

        placeholders = ", ".join(["?"] * len(columns))
        col_names = ", ".join(columns)
        sql = f"INSERT OR IGNORE INTO products ({col_names}) VALUES ({placeholders})"

        rows = [
            [product.get(col) for col in columns]
            for product in products
        ]

        cursor = self.conn.cursor()
        cursor.executemany(sql, rows)
        self.conn.commit()
        return cursor.rowcount

    def upsert(self, product: dict[str, Any]) -> None:
        """插入或更新单条商品。"""
        columns = [
            "id", "title", "image_url",
            "category_l1", "category_l2", "category_l3", "category_l4",
            "source_dataset", "description",
            "brand", "material", "season", "style", "color",
            "shoe_type", "heel_type", "closure_type", "gender",
            "target_user", "usage_scene", "functionality",
            "text_color", "image_color", "color_source",
            "color_confidence", "color_conflict", "color_detail", "tags",
            "rating", "sales_count", "stock_status",
            "attributes_raw", "attributes_json", "raw_json", "content_text",
        ]
        set_clause = ", ".join(f"{col}=excluded.{col}" for col in columns if col != "id")
        placeholders = ", ".join(["?"] * len(columns))
        col_names = ", ".join(columns)

        sql = (
            f"INSERT INTO products ({col_names}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {set_clause}"
        )

        self.conn.execute(sql, [product.get(col) for col in columns])
        self.conn.commit()

    # ── 查询 ───────────────────────────────────────────

    def get_by_id(self, product_id: str) -> Optional[dict[str, Any]]:
        """按 ID 查询单条商品。"""
        cursor = self.conn.execute(
            "SELECT * FROM products WHERE id = ?", (product_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_ids(self, product_ids: list[str]) -> list[dict[str, Any]]:
        """按 ID 列表批量查询，保持传入顺序。"""
        if not product_ids:
            return []

        placeholders = ", ".join(["?"] * len(product_ids))
        rows = self.conn.execute(
            f"SELECT * FROM products WHERE id IN ({placeholders})",
            product_ids,
        ).fetchall()

        # 保持 ID 顺序
        row_map = {row["id"]: dict(row) for row in rows}
        return [row_map[pid] for pid in product_ids if pid in row_map]

    def count(self) -> int:
        """商品总数。"""
        return self.conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]

    # ── 过滤查询 ───────────────────────────────────────

    def query(
        self,
        category_l1: Optional[str] = None,
        category_l2: Optional[str] = None,
        brand: Optional[str] = None,
        material: Optional[str] = None,
        season: Optional[str] = None,
        gender: Optional[str] = None,
        rating_min: Optional[float] = None,
        stock_status: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        组合条件查询。

        所有条件都是 AND 关系。keyword 使用 FTS5 全文搜索。
        """
        conditions: list[str] = []
        params: list[Any] = []

        if category_l1:
            conditions.append("category_l1 = ?")
            params.append(category_l1)
        if category_l2:
            conditions.append("category_l2 = ?")
            params.append(category_l2)
        if brand:
            conditions.append("brand = ?")
            params.append(brand)
        if material:
            conditions.append("material LIKE ?")
            params.append(f"%{material}%")
        if season:
            conditions.append("season LIKE ?")
            params.append(f"%{season}%")
        if gender:
            conditions.append("gender = ?")
            params.append(gender)
        if rating_min is not None:
            conditions.append("rating >= ?")
            params.append(rating_min)
        if stock_status:
            conditions.append("stock_status = ?")
            params.append(stock_status)

        if keyword:
            # FTS5 全文搜索
            conditions.append(
                "rowid IN (SELECT rowid FROM products_fts WHERE products_fts MATCH ?)"
            )
            # FTS5 语法：用 * 做前缀匹配
            params.append(f'"{keyword}"')

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = (
            f"SELECT * FROM products WHERE {where_clause} "
            f"ORDER BY rating DESC, sales_count DESC "
            f"LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def filter_by_ids(
        self,
        product_ids: list[str],
        brand: Optional[str] = None,
        material: Optional[str] = None,
        season: Optional[str] = None,
        gender: Optional[str] = None,
        stock_status: Optional[str] = None,
        rating_min: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        在给定的商品 ID 列表上叠加 SQL 条件过滤。

        典型场景：向量检索返回 top-50 候选 → SQL 过滤价格/品牌/季节。

        Returns:
            过滤后的商品列表，保留原 ID 顺序。
        """
        if not product_ids:
            return []

        placeholders = ", ".join(["?"] * len(product_ids))
        conditions = [f"p.id IN ({placeholders})"]
        params: list[Any] = list(product_ids)

        if brand:
            conditions.append("p.brand = ?")
            params.append(brand)
        if material:
            conditions.append("p.material LIKE ?")
            params.append(f"%{material}%")
        if season:
            conditions.append("p.season LIKE ?")
            params.append(f"%{season}%")
        if gender:
            conditions.append("p.gender = ?")
            params.append(gender)
        if stock_status:
            conditions.append("p.stock_status = ?")
            params.append(stock_status)
        if rating_min is not None:
            conditions.append("p.rating >= ?")
            params.append(rating_min)

        where_clause = " AND ".join(conditions)
        sql = f"SELECT * FROM products p WHERE {where_clause}"

        rows = self.conn.execute(sql, params).fetchall()
        row_map = {row["id"]: dict(row) for row in rows}
        return [row_map[pid] for pid in product_ids if pid in row_map]

    # ── 统计 ───────────────────────────────────────────

    def get_categories(self) -> dict[str, int]:
        """各级类目的商品数量统计。"""
        cats: dict[str, int] = {}
        for level in ["category_l1", "category_l2", "category_l3"]:
            rows = self.conn.execute(
                f"SELECT {level}, COUNT(*) as cnt FROM products "
                f"GROUP BY {level} ORDER BY cnt DESC"
            ).fetchall()
            for row in rows:
                if row[level]:
                    cats[f"{level}:{row[level]}"] = row["cnt"]
        return cats
