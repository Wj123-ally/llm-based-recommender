import logging
import sys
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.recommender.check_topic_node import check_topic_node
from src.recommender.rag_node import rag_recommender
from src.recommender.self_query_node import self_query_retrieve
from src.recommender.state import RecState

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def route_after_topic_check(state: RecState) -> str:
    if state.get("on_topic") == "Yes":
        return "hybrid_retrieve"

    return END


def knowledge_retrieve_node(state: RecState) -> RecState:
    """
    LangGraph 知识库检索节点。

    从用户上传的知识库文件中检索与 query 相关的文档片段。
    如果知识库为空或检索失败，不影响后续流程。
    """
    try:
        from src.knowledge_base.knowledge_retriever import retrieve_knowledge

        knowledge_docs = retrieve_knowledge(state["query"])
        state["knowledge_docs"] = knowledge_docs
        return state
    except ImportError:
        logger.warning("知识检索模块不可用，跳过")
        state["knowledge_docs"] = ""
        return state
    except Exception:
        logger.exception("知识库检索失败，跳过")
        state["knowledge_docs"] = ""
        return state


def create_recommender_graph():
    workflow = StateGraph(RecState)

    workflow.add_node("check_topic", check_topic_node)
    workflow.add_node("hybrid_retrieve", self_query_retrieve)
    workflow.add_node("knowledge_retrieve", knowledge_retrieve_node)
    workflow.add_node("rag_recommender", rag_recommender)

    workflow.set_entry_point("check_topic")

    workflow.add_conditional_edges(
        "check_topic",
        route_after_topic_check,
        {
            "hybrid_retrieve": "hybrid_retrieve",
            END: END,
        },
    )
    workflow.add_edge("hybrid_retrieve", "knowledge_retrieve")
    workflow.add_edge("knowledge_retrieve", "rag_recommender")
    workflow.add_edge("rag_recommender", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    graph = create_recommender_graph()

    result = graph.invoke(
        {"query": "推荐一件适合秋冬通勤的外套"},
        config={"configurable": {"thread_id": "test-thread"}},
    )

    print("用户问题:", result.get("query"))
    print("主题判断:", result.get("on_topic"))
    print("检索状态:", result.get("retrieval_state"))
    print("检索来源:", result.get("retrieval_source"))
    print("最终回答:")
    print(result.get("recommendation"))
