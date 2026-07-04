"""
推荐系统主工作流图（已优化：话题判断+意图分析合并为一次 LLM 调用）。

工作流节点顺序：

  user_input
      │
      ▼
  [combined_analysis] ─── No ──→ END（拒绝非鞋类问题）
      │ Yes
      ├── need_products ──→ [hybrid_retrieve] ──┬── need_knowledge ──→ [knowledge_retrieve] ──┐
      │                                          │                                              │
      │                                          └── !need_knowledge ──────────────────────────┤
      │                                                                                         │
      ├── !need_products && need_knowledge ────────────────────────────→ [knowledge_retrieve] ──┤
      │                                                                                         │
      └── !need_products && !need_knowledge ───────────────────────────────────────────────────┤
                                                                                                │
                                                                                                ▼
                                                                                        [rag_recommender]
                                                                                                │
                                                                                                ▼
                                                                                              END
"""

import concurrent.futures
import logging
import sys
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.recommender.combined_analysis_node import combined_analysis_node
from src.recommender.knowledge_retrieve_node import knowledge_retrieve_node
from src.recommender.rag_node import rag_recommender
from src.recommender.self_query_node import self_query_retrieve
from src.recommender.state import RecState

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ────────────────────────────── 路由函数 ──────────────────────────────


def route_after_combined_analysis(state: RecState) -> str:
    """合并分析后的路由：按需分发到并行检索或直接生成。"""
    if state.get("on_topic") != "Yes":
        logger.info("[路由] 话题检查不通过 → 终止流程")
        return END

    need_products = state.get("need_products", False)
    need_knowledge = state.get("need_knowledge", False)

    if need_products or need_knowledge:
        logger.info("[路由] 需要检索 → 进入并行检索节点")
        return "parallel_retrieve"

    logger.info("[路由] 无需商品也无知识 → 直接进入生成")
    return "rag_recommender"


def route_after_hybrid_retrieve(state: RecState) -> str:
    """商品检索后的路由：如果还需要知识资料，继续知识检索。"""
    need_knowledge = state.get("need_knowledge", False)

    if need_knowledge:
        logger.info("[路由] 商品检索完成，还需知识 → 进入知识库检索")
        return "knowledge_retrieve"

    logger.info("[路由] 商品检索完成，无需知识 → 进入 RAG 生成")
    return "rag_recommender"


# ────────────────────────────── 并行检索节点 ──────────────────────────────


def parallel_retrieve_node(state: RecState) -> RecState:
    """
    并行检索节点：当需要商品和知识时，同时运行两种检索。

    商品检索（self_query_retrieve）和知识检索（knowledge_retrieve_node）
    互相独立，写入 state 的不同字段，因此可以安全地并行执行。

    - 仅需商品 → 直接调用 self_query_retrieve
    - 仅需知识 → 直接调用 knowledge_retrieve_node
    - 两者都需要 → 并行执行，合并结果
    - 都不需要 → 直接跳过
    """
    need_products = state.get("need_products", False)
    need_knowledge = state.get("need_knowledge", False)

    # 都不需要 → 跳过
    if not need_products and not need_knowledge:
        logger.info("[并行检索] 无需检索，跳过")
        state["retrieval_state"] = "skipped"
        state["retrieval_source"] = "none"
        state["products"] = ""
        state["documents"] = []
        state["knowledge_docs"] = ""
        state["knowledge_retrieval_state"] = "skipped"
        return state

    # 只需要一种 → 顺序执行（避免不必要的线程开销）
    if need_products and not need_knowledge:
        logger.info("[并行检索] 仅需商品检索")
        return self_query_retrieve(state)

    if need_knowledge and not need_products:
        logger.info("[并行检索] 仅需知识检索")
        return knowledge_retrieve_node(state)

    # 两种都需要 → 并行执行
    logger.info("[并行检索] 开始并行检索（商品 + 知识）")

    # 使用深拷贝避免共享嵌套对象
    import copy
    product_state = copy.deepcopy(dict(state))
    knowledge_state = copy.deepcopy(dict(state))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        product_future = executor.submit(self_query_retrieve, product_state)
        knowledge_future = executor.submit(knowledge_retrieve_node, knowledge_state)

        product_result = product_future.result()
        knowledge_result = knowledge_future.result()

    # 智能合并结果：只更新各自负责的字段，避免覆盖
    # 商品检索负责的字段
    product_fields = ["products", "documents", "retrieval_state", "retrieval_source", "recommended_product_ids"]
    for field in product_fields:
        if field in product_result:
            state[field] = product_result[field]

    # 知识检索负责的字段
    knowledge_fields = ["knowledge_docs", "knowledge_retrieval_state"]
    for field in knowledge_fields:
        if field in knowledge_result:
            state[field] = knowledge_result[field]

    logger.info("[并行检索] 完成：商品=%s 知识=%s",
                state.get("retrieval_state"),
                state.get("knowledge_retrieval_state"))
    return state


# ────────────────────────────── 图构建 ──────────────────────────────


def create_recommender_graph():
    """
    构建并编译完整的推荐工作流图（已优化）。

    3 节点工作流:
        combined_analysis — 一次 LLM 调用完成主题判断 + 意图分析
        parallel_retrieve — 需要时并行执行商品检索 + 知识库检索
        rag_recommender   — LLM 生成最终推荐
    """
    workflow = StateGraph(RecState)

    # 注册节点
    workflow.add_node("combined_analysis", combined_analysis_node)
    workflow.add_node("parallel_retrieve", parallel_retrieve_node)
    workflow.add_node("rag_recommender", rag_recommender)

    # 入口
    workflow.set_entry_point("combined_analysis")

    # 合并分析 → 并行检索 / 直接生成 / 终止
    workflow.add_conditional_edges(
        "combined_analysis",
        route_after_combined_analysis,
        {
            "parallel_retrieve": "parallel_retrieve",
            "rag_recommender": "rag_recommender",
            END: END,
        },
    )

    # 检索 → 生成
    workflow.add_edge("parallel_retrieve", "rag_recommender")

    # 生成 → 结束
    workflow.add_edge("rag_recommender", END)

    memory = MemorySaver()
    compiled = workflow.compile(checkpointer=memory)

    logger.info("[工作流] 图编译完成，节点数: %s", len(workflow.nodes))
    return compiled


# ────────────────────────────── 自测 ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    test_cases = [
        "推荐一双适合秋冬通勤的皮鞋，300元以下",

    ]

    graph = create_recommender_graph()

    for query in test_cases:
        print("\n" + "=" * 70)
        print(f"测试问题: {query}")
        print("=" * 70)

        result = graph.invoke(
            {"query": query},
            config={"configurable": {"thread_id": f"test-{hash(query)}"}},
        )

        print(f"\n意图分析: {result.get('intent_analysis', '-')}")
        print(f"需要商品: {result.get('need_products')}")
        print(f"需要知识: {result.get('need_knowledge')}")
        print(f"商品检索: {result.get('retrieval_state')}")
        print(f"知识检索: {result.get('knowledge_retrieval_state')}")
        print(f"\n最终回答:\n{result.get('recommendation', '-')}")
