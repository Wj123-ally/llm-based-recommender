"""
SQLite 数据库连接管理。

单例模式，整个项目共用同一个连接实例。
"""

import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as settings  # noqa: E402


@lru_cache(maxsize=1)
def get_connection() -> sqlite3.Connection:
    """
    获取 SQLite 数据库连接（单例）。

    启用 WAL 模式以支持并发读取，外键约束开启。
    """
    db_path = str(settings.DATABASE_PATH)
    db_path_obj = Path(db_path)
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def close_connection() -> None:
    """关闭数据库连接，清除缓存。"""
    conn = get_connection()
    conn.close()
    get_connection.cache_clear()
