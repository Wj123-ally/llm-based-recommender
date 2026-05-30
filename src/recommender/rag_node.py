import logging
import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser

from src.recommender.state import RecState
from src.recommender.utils import create_rag_template

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as settings  # noqa: E402


logger = logging.getLogger(__name__)


def create_llm() -> Any:
    """
    创建 DashScope / 通义千问聊天模型。

    后续如果需要替换模型，只需要修改这个函数。
    """
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise EnvironmentError("请先设置环境变量 DASHSCOPE_API_KEY")

    try:
        from langchain_community.chat_models import ChatTongyi
    except ImportError:
        from langchain_community.chat_models.tongyi import ChatTongyi

    temperature = getattr(settings, "LLM_TEMPERATURE", 0.3)
    max_tokens = getattr(settings, "LLM_MAX_TOKENS", 800)

    return ChatTongyi(
        model=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_rag_chain() -> Any:
    """
    构建 RAG 生成链。
    """
    prompt = create_rag_template()
    llm = create_llm()
    output_parser = StrOutputParser()

    return prompt | llm | output_parser


def rag_recommender(state: RecState) -> RecState:
    """
    LangGraph 推荐生成节点。

    只负责基于 query 和 products 生成最终推荐回答。
    """
    query = state["query"]
    products = state.get("products", "")

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
            }
        )

        state["recommendation"] = recommendation
        return state
    except Exception as exc:
        logger.exception("生成推荐回答失败")
        state["error"] = str(exc)
        state["recommendation"] = "抱歉，生成推荐结果时出现问题，请稍后再试。"
        return state
