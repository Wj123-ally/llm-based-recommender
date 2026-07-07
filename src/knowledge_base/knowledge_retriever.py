"""
知识库检索器。

提供与商品检索器一致的接口风格，从知识库 Milvus collection 中检索
与用户查询最相关的文档片段。
"""

import logging
from langchain_core.documents import Document

from src.shared import create_embedding_model, get_knowledge_collection_name

logger = logging.getLogger(__name__)


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
    from src.retriever.milvus_store import (
        count_collection,
        search_documents,
    )

    collection_name = get_knowledge_collection_name()

    # 检查知识库是否有内容
    try:
        count = count_collection(collection_name)
    except Exception:
        logger.exception("读取知识库文档数量失败")
        return ""

    if count == 0:
        logger.debug("知识库为空，跳过知识检索")
        return ""

    # 生成查询向量并执行搜索
    try:
        query_embedding = create_embedding_model().embed_query(query)
        scored_docs = search_documents(
            collection_name,
            query_embedding,
            top_k,
        )
    except Exception:
        logger.exception("知识库向量检索失败")
        return ""

    # 过滤低于阈值的片段
    results: list[Document] = [
        doc
        for doc, score in scored_docs
        if score is not None and float(score) >= similarity_threshold
    ]

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
