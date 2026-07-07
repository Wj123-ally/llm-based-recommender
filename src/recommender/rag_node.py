import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser

from src.recommender.state import RecState
from src.recommender.utils import create_rag_template
from src.shared import create_chat_llm

logger = logging.getLogger(__name__)

MAX_RECENT_RECOMMENDATION_TURNS = 5
MAX_PRODUCTS_PER_MEMORY_TURN = 6


def _metadata_value(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _summarize_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, document in enumerate(documents, start=1):
        metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
        if not isinstance(metadata, dict):
            continue
        product_id = _metadata_value(metadata, "id", "product_id")
        image_url = _metadata_value(metadata, "image_url", "商品图片")
        key = product_id or image_url
        if not key or key in seen:
            continue
        seen.add(key)
        summaries.append(
            {
                "rank": index,
                "source_rank": _metadata_value(metadata, "_source_rank"),
                "id": product_id,
                "title": _metadata_value(metadata, "title", "商品标题"),
                "color": _metadata_value(metadata, "color", "text_color", "image_color"),
                "shoe_type": _metadata_value(metadata, "shoe_type"),
                "gender": _metadata_value(metadata, "gender", "target_user"),
                "season": _metadata_value(metadata, "season"),
                "image_url": image_url,
            }
        )
        if len(summaries) >= MAX_PRODUCTS_PER_MEMORY_TURN:
            break
    return summaries


def _remember_recommendations(state: RecState) -> None:
    products = _summarize_documents(state.get("documents", []))
    if not products:
        return

    recent = state.get("recent_recommendations", [])
    if not isinstance(recent, list):
        recent = []
    recent = [
        {
            "query": state.get("query", ""),
            "products": products,
        }
    ] + recent
    state["recent_recommendations"] = recent[:MAX_RECENT_RECOMMENDATION_TURNS]
    state["last_recommended_products"] = products


def _format_recent_recommendations(recent: Any) -> str:
    if not isinstance(recent, list) or not recent:
        return ""

    lines = ["Recent products actually shown to the user:"]
    for turn_index, item in enumerate(recent[:MAX_RECENT_RECOMMENDATION_TURNS], start=1):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "")
        if query:
            lines.append(f"Turn {turn_index} query: {query}")
        products = item.get("products", [])
        if not isinstance(products, list):
            continue
        for product in products[:MAX_PRODUCTS_PER_MEMORY_TURN]:
            if not isinstance(product, dict):
                continue
            lines.append(
                "- "
                f"previous_display_rank={product.get('rank', '')}; "
                f"title={product.get('title', '')}; "
                f"color={product.get('color', '')}; "
                f"shoe_type={product.get('shoe_type', '')}; "
                f"gender={product.get('gender', '')}; "
                f"season={product.get('season', '')}"
            )
    return "\n".join(lines)


def build_rag_chain(streaming: bool = False) -> Any:
    """
    构建 RAG 生成链。

    Args:
        streaming: 是否启用流式输出
    """
    prompt = create_rag_template()
    llm = create_chat_llm(temperature=0.3, max_tokens=500, streaming=streaming)
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

    if state.get("need_products", False) and not products:
        state["recommendation"] = (
            "抱歉，我暂时没有找到完全符合当前条件的商品。"
            "如果你愿意，我可以帮你放宽部分条件，例如颜色、鞋型或使用场景后再重新筛选。"
        )
        state["documents"] = []
        state["previous_query"] = query
        return state

    # 只有两者都为空时才返回抱歉
    if not products and not knowledge_docs:
        state["recommendation"] = (
            "抱歉，我暂时没有找到合适的商品或相关知识。你可以换个描述再试试。"
        )
        state["documents"] = []
        state["previous_query"] = query
        return state

    try:
        rag_chain = build_rag_chain()

        # 构建对话上下文
        previous_query = state.get("previous_query", "")
        context_mode = state.get("context_mode", "new_request")
        recent_context = ""
        if context_mode == "follow_up":
            recent_context = _format_recent_recommendations(
                state.get("recent_recommendations", [])
            )
        if previous_query:
            if context_mode == "new_request":
                context_text = (
                    f"上一轮用户问题：{previous_query}\n"
                    f"当前用户问题：{query}\n"
                    f"context_mode：{context_mode}\n"
                    "当前轮是新的商品需求。不要继承上一轮商品、上一轮商品编号或上一轮约束。"
                )
            elif context_mode == "clarification":
                context_text = (
                    f"上一轮用户问题：{previous_query}\n"
                    f"当前用户补充条件：{query}\n"
                    f"context_mode：{context_mode}\n"
                    "当前轮只继承上一轮需求条件，不使用上一轮商品列表作为候选池。"
                )
            else:
                context_text = (
                    f"上一轮用户问题：{previous_query}\n"
                    f"当前用户追问：{query}\n"
                    f"context_mode：{context_mode}\n"
                    "当前轮显式指代上一轮展示商品，仅用上一轮商品记忆解析指代。"
                )
        else:
            context_text = "这是用户的第一轮提问，没有历史上下文。"

        if context_mode == "new_request":
            context_text += "\nCurrent turn is a new request. Do not inherit old product constraints unless the user explicitly asks to."
        elif recent_context:
            context_text += (
                "\nUse the following list only to resolve references like previous, "
                "the white one, second item, or that product. Do not treat it as a hard filter.\n"
                "Never use previous_display_rank as 商品N in the current answer. "
                "In the current answer, 商品N must refer only to the current 商品资料 list.\n"
                f"{recent_context}"
            )

        # 根据实际可用资源，传入真实内容或明确占位
        chain_input = {
            "query": query,
            "docs": products if products else "暂无匹配的商品资料。",
            "knowledge": knowledge_docs if knowledge_docs else "暂无相关知识资料。",
            "context": context_text,
        }

        recommendation = rag_chain.invoke(chain_input)
        state["recommendation"] = recommendation
        _remember_recommendations(state)
        # 保存当前 query，下一轮作为 previous_query 用于理解追问
        state["previous_query"] = query
        return state
    except Exception as exc:
        logger.exception("生成推荐回答失败")
        state["error"] = str(exc)
        state["recommendation"] = "抱歉，生成推荐结果时出现问题，请稍后再试。"
        state["documents"] = []
        return state
