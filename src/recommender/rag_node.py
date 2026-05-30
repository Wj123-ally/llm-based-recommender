import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser

from src.recommender.state import RecState
from src.recommender.utils import create_rag_template
from src.shared import create_chat_llm

logger = logging.getLogger(__name__)


def build_rag_chain() -> Any:
    """
    构建 RAG 生成链。
    """
    prompt = create_rag_template()
    llm = create_chat_llm(temperature=0.3, max_tokens=800)
    output_parser = StrOutputParser()

    return prompt | llm | output_parser


def rag_recommender(state: RecState) -> RecState:
    """
    LangGraph 推荐生成节点。

    基于 query、products 和 knowledge_docs 生成最终推荐回答。
    """
    query = state["query"]
    products = state.get("products", "")
    knowledge_docs = state.get("knowledge_docs", "")

    if not products:
        state["recommendation"] = (
            "抱歉，我暂时没有找到合适的商品。你可以换个描述再试试。"
        )
        return state

    try:
        rag_chain = build_rag_chain()
        recommendation = rag_chain.invoke(
            {
                "query": query,
                "docs": products,
                "knowledge": knowledge_docs if knowledge_docs else "暂无相关知识资料。",
            }
        )

        state["recommendation"] = recommendation
        return state
    except Exception as exc:
        logger.exception("生成推荐回答失败")
        state["error"] = str(exc)
        state["recommendation"] = "抱歉，生成推荐结果时出现问题，请稍后再试。"
        return state
