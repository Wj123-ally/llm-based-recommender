"""
知识库检索器。

提供与商品检索器一致的接口风格，从知识库 Chroma collection 中检索
与用户查询最相关的文档片段。
"""

import logging
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

import config as settings
from src.knowledge_base.document_processor import KNOWLEDGE_COLLECTION_NAME

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_embedding_model() -> Any:
    """延迟加载 embedding 模型。"""
    import os
    import time

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
                return DashScopeEmbeddings(model=settings.DASHSCOPE_EMBEDDING_MODEL)
            except TypeError:
                return DashScopeEmbeddings(
                    model_name=settings.DASHSCOPE_EMBEDDING_MODEL
                )
        except Exception as exc:
            last_error = exc
            logger.warning("初始化 embedding 失败，第 %s 次重试", attempt)
            if attempt < 3:
                time.sleep(2)

    raise RuntimeError("初始化 embedding 模型失败") from last_error


@lru_cache(maxsize=1)
def _get_knowledge_chroma() -> Any:
    """获取知识库 Chroma collection。"""
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    persist_dir = str(settings.CHROMA_INDEX_PATH)

    return Chroma(
        collection_name=KNOWLEDGE_COLLECTION_NAME,
        embedding_function=_get_embedding_model(),
        persist_directory=persist_dir,
    )


def retrieve_knowledge(
    query: str,
    top_k: int = 3,
    similarity_threshold: float = 0.3,
) -> str:
    """
    从知识库中检索与查询最相关的文档片段。

    Args:
        query: 用户查询文本。
        top_k: 返回的文档片段数量上限，默认 3。
        similarity_threshold: 相似度阈值，低于此值的片段将被过滤，默认 0.3。

    Returns:
        格式化的文档片段文本。如果知识库为空或无相关结果，返回空字符串。
    """
    try:
        chroma = _get_knowledge_chroma()
    except Exception:
        logger.exception("连接知识库 Chroma collection 失败")
        return ""

    # 检查知识库是否有内容
    try:
        count = chroma._collection.count()
    except Exception:
        logger.exception("读取知识库文档数量失败")
        return ""

    if count == 0:
        logger.debug("知识库为空，跳过知识检索")
        return ""

    try:
        if hasattr(chroma, "similarity_search_with_relevance_scores"):
            scored_docs = chroma.similarity_search_with_relevance_scores(
                query,
                k=top_k,
            )
            # 过滤低于阈值的片段
            results: list[Document] = [
                doc
                for doc, score in scored_docs
                if score is not None and float(score) >= similarity_threshold
            ]
        else:
            results = chroma.similarity_search(query, k=top_k)
    except Exception:
        logger.exception("知识库向量检索失败")
        return ""

    if not results:
        logger.debug("未找到相关知识片段")
        return ""

    # 格式化为文本供 RAG prompt 使用
    parts: list[str] = []
    for index, doc in enumerate(results, start=1):
        source = doc.metadata.get("source_filename", "未知来源")
        parts.append(f"[知识片段 {index}，来源：{source}]\n{doc.page_content}")

    formatted = "\n\n".join(parts)
    logger.info("从知识库检索到 %s 个相关片段", len(results))
    return formatted
