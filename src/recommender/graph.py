"""
推荐系统主工作流图。

工作流节点顺序：

  user_input
      │
      ▼
  [check_topic] ─── No ──→ END（拒绝非服装问题）
      │ Yes
      ▼
  [analyze_intent] ───────── 分析：需要商品？需要知识？
      │
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

import logging
import sys
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.recommender.analyze_intent_node import analyze_intent_node
from src.recommender.check_topic_node import check_topic_node
from src.recommender.rag_node import rag_recommender
from src.recommender.self_query_node import self_query_retrieve
from src.recommender.state import RecState

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ────────────────────────────── 路由函数 ──────────────────────────────


def route_after_topic_check(state: RecState) -> str:
    """话题检查后的路由：通过则进入意图分析，否则终止。"""
    on_topic = state.get("on_topic")

    if on_topic == "Yes":
        logger.info("[路由] 话题检查通过 → 进入意图分析")
        return "analyze_intent"

    logger.info("[路由] 话题检查不通过 → 终止流程")
    return END


def route_after_intent(state: RecState) -> str:
    """意图分析后的路由：按需分发到商品检索或知识检索。"""
    need_products = state.get("need_products", True)
    need_knowledge = state.get("need_knowledge", False)

    if need_products:
        logger.info("[路由] 需要商品 → 进入商品检索")
        return "hybrid_retrieve"

    if need_knowledge:
        logger.info("[路由] 仅需知识 → 进入知识库检索")
        return "knowledge_retrieve"

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


# ────────────────────────────── 知识检索节点 ──────────────────────────────


def knowledge_retrieve_node(state: RecState) -> RecState:
    """
    LangGraph 知识库检索节点。

    从用户上传的知识库文件中检索与 query 相关的文档片段。
    如果知识库为空或检索失败，不影响后续流程。
    """
    query = state["query"]

    logger.info("-" * 50)
    logger.info("[知识检索] 开始检索知识库")
    logger.info("[知识检索] 查询语句: %s", query)

    try:
        from src.knowledge_base.knowledge_retriever import retrieve_knowledge

        knowledge_docs = retrieve_knowledge(state["query"])

        if knowledge_docs:
            state["knowledge_docs"] = knowledge_docs
            state["knowledge_retrieval_state"] = "success"
            logger.info("[知识检索] 状态: 成功，找到相关文档片段")
            logger.info("[知识检索] 内容预览: %s...", knowledge_docs[:200])
        else:
            state["knowledge_docs"] = ""
            state["knowledge_retrieval_state"] = "empty"
            logger.info("[知识检索] 状态: 知识库为空或无匹配内容")

    except ImportError:
        logger.warning("[知识检索] 模块不可用，跳过")
        state["knowledge_docs"] = ""
        state["knowledge_retrieval_state"] = "skipped"
    except Exception:
        logger.exception("[知识检索] 检索异常，跳过")
        state["knowledge_docs"] = ""
        state["knowledge_retrieval_state"] = "skipped"

    logger.info("-" * 50)
    return state


# ────────────────────────────── 图构建 ──────────────────────────────


def create_recommender_graph():
    """
    构建并编译完整的推荐工作流图。

    节点:
        check_topic       — 判断是否服装相关
        analyze_intent    — 拆解用户需求（商品 / 知识 / 混合）
        hybrid_retrieve   — 商品混合检索（Chroma + BM25 + Reranker）
        knowledge_retrieve — 知识库向量检索
        rag_recommender   — LLM 生成最终推荐
    """
    workflow = StateGraph(RecState)

    # 注册所有节点
    workflow.add_node("check_topic", check_topic_node)
    workflow.add_node("analyze_intent", analyze_intent_node)
    workflow.add_node("hybrid_retrieve", self_query_retrieve)
    workflow.add_node("knowledge_retrieve", knowledge_retrieve_node)
    workflow.add_node("rag_recommender", rag_recommender)

    # 入口
    workflow.set_entry_point("check_topic")

    # 话题检查 → 意图分析 或 终止
    workflow.add_conditional_edges(
        "check_topic",
        route_after_topic_check,
        {
            "analyze_intent": "analyze_intent",
            END: END,
        },
    )

    # 意图分析 → 商品检索 / 知识检索 / 直接生成
    workflow.add_conditional_edges(
        "analyze_intent",
        route_after_intent,
        {
            "hybrid_retrieve": "hybrid_retrieve",
            "knowledge_retrieve": "knowledge_retrieve",
            "rag_recommender": "rag_recommender",
        },
    )

    # 商品检索 → 知识检索 或 生成
    workflow.add_conditional_edges(
        "hybrid_retrieve",
        route_after_hybrid_retrieve,
        {
            "knowledge_retrieve": "knowledge_retrieve",
            "rag_recommender": "rag_recommender",
        },
    )

    # 知识检索 → 生成
    workflow.add_edge("knowledge_retrieve", "rag_recommender")

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
        "推荐一件适合秋冬通勤的外套",
        "羊毛衫怎么洗不会缩水",
        "推荐一件睡衣，顺便告诉我怎么洗护",
        "羽绒服和棉服哪个更保暖",
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
