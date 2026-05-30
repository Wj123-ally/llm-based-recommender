"""
共享工厂模块。

统一项目中重复出现的工厂函数：
- create_chat_llm()          → 通义千问 LLM（3 处重复已消除）
- create_embedding_model()   → DashScope embedding（4 处重复已消除）
- create_chroma_collection() → Chroma 向量存储（3 处重复已消除）

所有函数均带 @lru_cache 单例缓存和重试逻辑。
"""

import logging
import os
import time
from functools import lru_cache
from typing import Any

import config as settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Chat LLM
# ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def create_chat_llm(temperature: float = 0.0, max_tokens: int | None = None) -> Any:
    """
    创建通义千问聊天模型（DashScope ChatTongyi）。

    项目中统一使用此函数创建 LLM 实例，不再在各模块中重复。

    Args:
        temperature: 生成温度。分类任务用 0（默认），生成任务用 0.3。
        max_tokens: 最大输出 token 数。None 表示使用模型默认值。

    Returns:
        ChatTongyi 实例。

    Raises:
        EnvironmentError: 未设置 DASHSCOPE_API_KEY 环境变量。
    """
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise EnvironmentError("请先设置环境变量 DASHSCOPE_API_KEY")

    try:
        from langchain_community.chat_models import ChatTongyi
    except ImportError:
        from langchain_community.chat_models.tongyi import ChatTongyi

    kwargs: dict[str, Any] = {
        "model": os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    logger.debug("创建 ChatTongyi: model=%s temperature=%s max_tokens=%s",
                 kwargs["model"], temperature, max_tokens)
    return ChatTongyi(**kwargs)


# ─────────────────────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def create_embedding_model() -> Any:
    """
    创建 DashScope embedding 模型。

    带 3 次重试，每次间隔 2 秒。使用 Config 中配置的模型名。

    Returns:
        DashScopeEmbeddings 实例。

    Raises:
        EnvironmentError: 未设置 DASHSCOPE_API_KEY。
        RuntimeError: 3 次重试后仍初始化失败。
    """
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise EnvironmentError("请先设置环境变量 DASHSCOPE_API_KEY")

    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            try:
                from langchain_community.embeddings import DashScopeEmbeddings
            except ImportError:
                from langchain_dashscope import DashScopeEmbeddings

            try:
                model = DashScopeEmbeddings(model=settings.DASHSCOPE_EMBEDDING_MODEL)
            except TypeError:
                model = DashScopeEmbeddings(
                    model_name=settings.DASHSCOPE_EMBEDDING_MODEL
                )

            logger.debug("Embedding 模型初始化成功: %s", settings.DASHSCOPE_EMBEDDING_MODEL)
            return model

        except Exception as exc:
            last_error = exc
            logger.warning("初始化 embedding 失败，第 %s/3 次重试: %s", attempt, exc)
            if attempt < 3:
                time.sleep(2)

    raise RuntimeError("初始化 embedding 模型失败（3 次重试后）") from last_error


# ─────────────────────────────────────────────────────────────
# Chroma Collection
# ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=4)
def create_chroma_collection(collection_name: str) -> Any:
    """
    创建 Chroma 向量存储 collection。

    使用统一的 embedding 模型和持久化目录，仅通过 collection_name 区分用途。

    Args:
        collection_name: Chroma collection 名称。
            - settings.CHROMA_COLLECTION_NAME → 商品库
            - KNOWLEDGE_COLLECTION_NAME → 知识库

    Returns:
        Chroma 向量存储实例。
    """
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    persist_dir = str(settings.CHROMA_INDEX_PATH)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=create_embedding_model(),
        persist_directory=persist_dir,
    )

    try:
        count = vectorstore._collection.count()
        logger.debug("Chroma collection '%s': %s 条文档", collection_name, count)
    except Exception:
        logger.exception("读取 Chroma collection '%s' 数量失败", collection_name)

    return vectorstore
