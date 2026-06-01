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
    支持三种模式：
    - 纯商品推荐（有 products，无 knowledge）
    - 纯知识回答（无 products，有 knowledge）
    - 混合推荐（两者都有）
    """
    query = state["query"]
    products = state.get("products", "")
    knowledge_docs = state.get("knowledge_docs", "")

    # 只有两者都为空时才返回抱歉
    if not products and not knowledge_docs:
        state["recommendation"] = (
            "抱歉，我暂时没有找到合适的商品或相关知识。你可以换个描述再试试。"
        )
        state["previous_query"] = query
        return state

    try:
        rag_chain = build_rag_chain()

        # 构建对话上下文
        previous_query = state.get("previous_query", "")
        if previous_query:
            context_text = (
                f"这是用户对上一轮的追问。\n"
                f"上一轮用户问题：{previous_query}\n"
                f"当前追问：{query}\n"
                f"请结合上一轮的推荐主题，理解当前的细化需求。"
            )
        else:
            context_text = "这是用户的第一轮提问，没有历史上下文。"

        # 根据实际可用资源，传入真实内容或明确占位
        chain_input = {
            "query": query,
            "docs": products if products else "暂无匹配的商品资料。",
            "knowledge": knowledge_docs if knowledge_docs else "暂无相关知识资料。",
            "context": context_text,
        }

        recommendation = rag_chain.invoke(chain_input)
        state["recommendation"] = recommendation
        # 保存当前 query，下一轮作为 previous_query 用于理解追问
        state["previous_query"] = query
        return state
    except Exception as exc:
        logger.exception("生成推荐回答失败")
        state["error"] = str(exc)
        state["recommendation"] = "抱歉，生成推荐结果时出现问题，请稍后再试。"
        return state
