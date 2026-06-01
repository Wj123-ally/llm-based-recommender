"""
商品混合检索节点。

根据 state["need_products"] 决定是否执行检索：
- need_products=True → Chroma + BM25 + Cross-Encoder 混合检索
- need_products=False → 跳过检索，标记为 skipped
"""

import logging
from typing import Any

from langchain_core.documents import Document

from src.recommender.state import RecState

logger = logging.getLogger(__name__)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"商品 {index}\n{build_product_content(doc.metadata) or doc.page_content}"
        for index, doc in enumerate(docs, start=1)
    )


def serialize_docs(docs: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "page_content": build_product_content(doc.metadata) or doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


# 延迟导入以避免循环依赖
def build_product_content(metadata: dict[str, Any]) -> str:
    from src.retriever.product_documents import build_product_content as _build

    return _build(metadata)


def self_query_retrieve(state: RecState) -> RecState:
    """
    LangGraph 商品检索节点。

    根据意图分析结果 need_products 决定是否执行检索：
    - True → 执行 Chroma + BM25 + Cross-Encoder 混合检索
    - False → 跳过检索，标记 retrieval_state=skipped
    """
    query = state["query"]
    need_products = state.get("need_products", True)
    previous_query = state.get("previous_query", "")

    logger.info("-" * 50)
    logger.info("[商品检索] 开始")
    logger.info("[商品检索] 查询语句: %s", query)
    logger.info("[商品检索] 上一轮 query: %s", previous_query or "(无)")
    logger.info("[商品检索] 是否需要商品: %s", need_products)

    # 意图分析判定不需要商品，直接跳过
    if not need_products:
        logger.info("[商品检索] 跳过 — 用户问题不涉及商品推荐需求")
        state["retrieval_state"] = "skipped"
        state["retrieval_source"] = "none"
        state["products"] = ""
        state["documents"] = []
        logger.info("-" * 50)
        return state

    # ── 追问检测：当前 query 太短且上一轮有上下文时，合并搜索词 ──
    search_query = query
    if previous_query and len(query.strip()) < 15:
        # 当前 query 像一个追问/过滤条件，不是独立的新问题
        search_query = f"{previous_query} {query}"
        logger.info("[商品检索] 检测到追问 → 合并搜索词: %s", search_query)

    # 执行混合检索
    try:
        from src.retriever.hybrid_retriever import (
            parse_query_filters,
            retrieve_products,
        )

        # 从原始 query 提取过滤条件，但从合并后的 search_query 做语义搜索
        filters = parse_query_filters(query)
        if filters:
            logger.info("[商品检索] 解析到过滤条件: %s", filters)

        results = retrieve_products(search_query, filters=filters if filters else None)

        if not results:
            logger.info("[商品检索] 结果: 未找到匹配商品")
            state["retrieval_state"] = "empty"
            state["retrieval_source"] = "none"
            state["products"] = ""
            state["documents"] = []
            logger.info("-" * 50)
            return state

        # 格式化检索结果
        state["retrieval_state"] = "success"
        state["retrieval_source"] = "hybrid"
        state["products"] = format_docs(results)
        state["documents"] = serialize_docs(results)

        logger.info("[商品检索] 结果: 成功，找到 %s 件商品", len(results))
        for i, doc in enumerate(results, 1):
            title = doc.metadata.get("title", doc.metadata.get("商品标题", "-"))
            meta = doc.metadata.get("_retrieval", {})
            logger.debug(
                "[商品检索]   #%s: %s | chroma=%.3f bm25=%s rrf=%.4f rerank=%.3f",
                i,
                title[:50],
                meta.get("chroma_score") or 0,
                meta.get("bm25_score") or "-",
                meta.get("rrf_score") or 0,
                meta.get("rerank_score") or 0,
            )

        logger.info("-" * 50)
        return state

    except Exception as exc:
        logger.exception("[商品检索] 异常: %s", exc)
        state["retrieval_state"] = "empty"
        state["retrieval_source"] = "none"
        state["error"] = str(exc)
        state["products"] = ""
        state["documents"] = []
        logger.info("-" * 50)
        return state


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 纯推荐场景
    print("\n=== 测试 1: need_products=True ===")
    result = self_query_retrieve({
        "query": "推荐一件小香风外套",
        "need_products": True,
    })
    print(f"状态: {result.get('retrieval_state')}")
    print(f"来源: {result.get('retrieval_source')}")
    print(f"商品数: {len(result.get('documents', []))}")

    # 纯知识场景
    print("\n=== 测试 2: need_products=False ===")
    result = self_query_retrieve({
        "query": "羊毛衫怎么洗",
        "need_products": False,
    })
    print(f"状态: {result.get('retrieval_state')}")
    print(f"来源: {result.get('retrieval_source')}")
