"""
知识库检索器。

提供与商品检索器一致的接口风格，从知识库 Chroma collection 中检索
与用户查询最相关的文档片段。
"""

import logging
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from src.knowledge_base.document_processor import KNOWLEDGE_COLLECTION_NAME
from src.shared import create_chroma_collection

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_knowledge_chroma() -> Any:
    """获取知识库 Chroma collection。"""
    return create_chroma_collection(KNOWLEDGE_COLLECTION_NAME)


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
