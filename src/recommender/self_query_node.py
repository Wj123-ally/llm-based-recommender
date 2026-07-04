"""
商品混合检索节点。

根据 state["need_products"] 决定是否执行检索：
- need_products=True → Milvus 向量 + BM25 + 图片向量混合检索
- need_products=False → 跳过检索，标记为 skipped
"""

import logging
from typing import Any

from langchain_core.documents import Document

from src.recommender.state import RecState

logger = logging.getLogger(__name__)
MAX_RETRIEVED_PRODUCTS_FOR_LLM = 3
SHOE_TYPE_TERMS = [
    "洞洞鞋",
    "运动鞋",
    "跑鞋",
    "跑步鞋",
    "板鞋",
    "皮鞋",
    "凉鞋",
    "拖鞋",
    "棉拖",
    "靴子",
    "马丁靴",
    "高跟鞋",
    "单鞋",
    "老爹鞋",
    "乐福鞋",
    "玛丽珍",
    "雪地靴",
]
NEW_PRODUCT_REQUEST_TERMS = [
    "有",
    "有没有",
    "还有",
    "推荐",
    "想看",
    "想要",
    "喜欢",
    "换成",
    "换个",
    "来",
]
PREVIOUS_PRODUCT_REFERENCE_TERMS = [
    "上一轮",
    "上轮",
    "刚才",
    "之前",
    "上面",
    "这些",
    "这几个",
    "那几个",
    "里面",
    "这双",
    "那双",
    "这个",
    "那个",
    "这款",
    "那款",
    "第一个",
    "第二个",
    "第三个",
    "第1个",
    "第2个",
    "第3个",
    "商品1",
    "商品2",
    "商品3",
]


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"商品 {index}\n{build_product_content(doc.metadata) or doc.page_content}"
        for index, doc in enumerate(docs, start=1)
    )


def serialize_docs(docs: list[Document]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        metadata = dict(doc.metadata or {})
        metadata["_source_rank"] = index
        serialized.append(
            {
                "page_content": build_product_content(metadata) or doc.page_content,
                "metadata": metadata,
            }
        )
    return serialized


# 延迟导入以避免循环依赖
def build_product_content(metadata: dict[str, Any]) -> str:
    from src.retriever.product_documents import build_product_content as _build

    return _build(metadata)


def has_explicit_previous_product_reference(query: str) -> bool:
    text = (query or "").strip()
    return any(term in text for term in PREVIOUS_PRODUCT_REFERENCE_TERMS)


def has_new_product_request(query: str) -> bool:
    text = (query or "").strip()
    if not any(term in text for term in SHOE_TYPE_TERMS):
        return False
    if any(term in text for term in NEW_PRODUCT_REQUEST_TERMS):
        return True
    return len(text) <= 12


def self_query_retrieve(state: RecState) -> RecState:
    """
    LangGraph 商品检索节点。

    根据意图分析结果 need_products 决定是否执行检索：
    - True → 执行 Milvus 向量 + BM25 + 图片向量混合检索
    - False → 跳过检索，标记 retrieval_state=skipped
    """
    query = state["query"]
    need_products = state.get("need_products", True)
    previous_query = state.get("previous_query", "")
    context_mode = state.get("context_mode", "new_request")

    logger.info("-" * 50)
    logger.info("[商品检索] 开始")
    logger.info("[商品检索] 查询语句: %s", query)
    logger.info("[商品检索] 上一轮 query: %s", previous_query or "(无)")
    logger.info("[商品检索] context_mode: %s", context_mode)
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

    current_query_has_new_product = has_new_product_request(query)
    has_previous_product_reference = has_explicit_previous_product_reference(query)

    # 每个商品推荐轮次都会检索。这里判断的不是“是否检索”，而是“检索词是否继承上一轮”。
    # 只有“女款/不要黑色/便宜点”这类未说明新鞋型的补充条件，才拼接上一轮 query。
    # “有没有拖鞋/有洞洞鞋吗/想看板鞋”这类本轮已说明鞋型的请求，必须只用本轮 query 检索。
    search_query = query
    should_inherit_search_context = (
        previous_query
        and context_mode == "clarification"
        and not current_query_has_new_product
        and not has_previous_product_reference
    )
    if should_inherit_search_context:
        search_query = f"{previous_query} {query}"
        logger.info("[商品检索] 检测到补充条件 → 合并搜索词: %s", search_query)
    else:
        logger.info("[商品检索] 使用本轮 query 检索: %s", search_query)

    # 执行混合检索
    try:
        from src.retriever.hybrid_retriever import (
            merge_product_filters,
            parse_query_filters,
            retrieve_products,
        )

        # 过滤条件要继承上一轮上下文，例如上一轮说“女鞋”，本轮说“不要黑色”。
        filter_query = (
            f"{previous_query} {query}".strip()
            if should_inherit_search_context
            else query
        )
        rule_filters = parse_query_filters(filter_query)
        llm_filters = state.get("product_filters", {})
        filters = merge_product_filters(
            llm_filters if isinstance(llm_filters, dict) else {},
            rule_filters,
        )
        if filters:
            logger.info("[商品检索] 解析到过滤条件: %s", filters)

        # 获取已推荐商品ID列表
        recommended_ids = state.get("recommended_product_ids", [])

        # 如果是全新请求（new_request），清空已推荐ID列表
        if context_mode == "new_request":
            recommended_ids = []
            logger.info("[商品检索] 检测到新请求，清空已推荐商品ID列表")

        # 判断是否需要排除已推荐商品
        # 只有在追问或补充条件时（非新请求），且已有推荐历史时才排除
        should_exclude_previous = (
            context_mode in ["follow_up", "clarification"]
            and len(recommended_ids) > 0
            and not current_query_has_new_product
        )

        exclude_ids = recommended_ids if should_exclude_previous else None
        if exclude_ids:
            logger.info(
                "[商品检索] 排除已推荐商品: %d个ID将被排除",
                len(exclude_ids)
            )

        results = retrieve_products(
            search_query,
            filters=filters if filters else None,
            exclude_ids=exclude_ids
        )

        if not results:
            logger.info("[商品检索] 结果: 未找到匹配商品")
            state["retrieval_state"] = "empty"
            state["retrieval_source"] = "none"
            state["products"] = ""
            state["documents"] = []
            logger.info("-" * 50)
            return state

        # 只把检索排序后的 top3 交给 LLM 和 UI。LLM 不再承担候选集中二次选品。
        selected_results = results[:MAX_RETRIEVED_PRODUCTS_FOR_LLM]

        # 更新已推荐商品ID列表
        new_ids = [
            doc.metadata.get("id")
            for doc in selected_results
            if doc.metadata.get("id")
        ]
        if new_ids:
            all_recommended_ids = list(set(recommended_ids + new_ids))
            state["recommended_product_ids"] = all_recommended_ids
            logger.info(
                "[商品检索] 更新已推荐商品ID: 新增%d个, 累计%d个",
                len(new_ids),
                len(all_recommended_ids)
            )

        # 格式化检索结果
        state["retrieval_state"] = "success"
        state["retrieval_source"] = "hybrid"
        state["products"] = format_docs(selected_results)
        state["documents"] = serialize_docs(selected_results)

        logger.info(
            "[商品检索] 结果: 成功，找到 %s 件商品，传给 LLM top%s 件",
            len(results),
            len(selected_results),
        )
        for i, doc in enumerate(selected_results, 1):
            title = doc.metadata.get("title", doc.metadata.get("商品标题", "-"))
            meta = doc.metadata.get("_retrieval", {})
            logger.debug(
                "[商品检索]   #%s: %s | dense=%.3f bm25=%s rrf=%.4f rerank=%.3f",
                i,
                title[:50],
                meta.get("dense_score") or 0,
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
        "query": "推荐一双黑色通勤皮鞋",
        "need_products": True,
    })
    print(f"状态: {result.get('retrieval_state')}")
    print(f"来源: {result.get('retrieval_source')}")
    print(f"商品数: {len(result.get('documents', []))}")

    # 纯知识场景
    print("\n=== 测试 2: need_products=False ===")
    result = self_query_retrieve({
        "query": "真皮鞋怎么清洁",
        "need_products": False,
    })
    print(f"状态: {result.get('retrieval_state')}")
    print(f"来源: {result.get('retrieval_source')}")
