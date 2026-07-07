"""
意图分析节点。

在话题检查通过后，用 LLM 将用户问题拆解为两类需求：
- need_products：是否需要从商品数据库推荐具体鞋类商品
- need_knowledge：是否需要从知识库获取鞋码、脚型、材质、清洁保养、搭配或选购知识

两种需求可以同时存在（混合型问题），也可以只存在一种。
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.recommender.state import RecState
from src.shared import create_chat_llm

logger = logging.getLogger(__name__)


class UserIntent(BaseModel):
    """用户意图分析结果。"""

    need_products: bool = Field(
        description="用户是否需要推荐具体鞋类商品。询问'推荐一双'、'有什么好的'、'帮我选一双'都属于需要商品推荐。"
    )
    need_knowledge: bool = Field(
        description=(
            "用户是否需要鞋码选择、脚型适配、鞋类材质、清洁保养、场景搭配、选购指南等知识。"
            "询问'鞋码怎么选'、'宽脚适合什么鞋'、'真皮鞋怎么保养'、'跑鞋怎么选'都属于需要知识。"
            "如果只是纯粹推荐商品而没有询问任何知识类问题，则为 false。"
        )
    )
    analysis: str = Field(
        description="用一句话解释你的分析判断，说明用户意图属于哪种类型，以及为什么需要/不需要商品推荐和知识库资料。"
    )


def analyze_intent_node(state: RecState) -> RecState:
    """
    LangGraph 意图分析节点。

    分析用户问题，判断需要调用哪些后端资源：
    - need_products -> 调用商品检索
    - need_knowledge -> 调用知识库检索
    """
    query = state["query"]
    previous_query = state.get("previous_query", "")

    logger.info("=" * 60)
    logger.info("[意图分析] 开始分析用户问题")
    logger.info("[意图分析] 用户问题: %s", query)
    logger.info("[意图分析] 上一轮 query: %s", previous_query or "(无)")

    system_prompt = """
你是一个鞋类商品推荐系统的意图分析器。

用户的问题可能包含一种或两种需求：

1. 商品推荐需求：
   - 用户明确要求推荐鞋类商品（如“推荐一双白色运动鞋”）
   - 用户描述使用场景并希望获得购买建议（如“适合通勤的男士皮鞋”“夏天穿的凉鞋”）
   - 用户想要对比或筛选鞋类商品
   - 用户对上一轮推荐结果进行追问或追加条件（如“我要女性的”“有没有黑色的”“便宜点”）

2. 知识资料需求：
   - 用户询问鞋码怎么选、脚型怎么适配
   - 用户询问鞋面/鞋底材质、透气、防滑、防水、保暖等功能差异
   - 用户询问鞋类清洁、保养、收纳方法
   - 用户询问鞋类搭配、场景选择、选购指南

对于每个用户问题，你需要判断它是否需要商品推荐、是否需要知识资料。
两者可以同时存在（如“推荐一双适合跑步的鞋，顺便说下跑鞋怎么选”）。

重要：如果用户的问题是简短追问（如“我要女性的”“男款有吗”），
请结合整个对话历史判断意图，而不仅仅是字面意思。
这类追问通常意味着用户对上一轮推荐结果不满意或想要细化条件。

注意：你只做意图判断，不要回答用户问题。
"""

    if previous_query:
        user_message = (
            f"上一轮用户问题：{previous_query}\n"
            f"当前用户追问：{query}\n\n"
            f"请结合上下文判断当前追问的意图。"
        )
    else:
        user_message = f"分析以下用户问题的意图：{query}"

    try:
        llm = create_chat_llm(temperature=0)
        structured_llm = llm.with_structured_output(UserIntent)
        result = structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
        )

        state["need_products"] = result.need_products
        state["need_knowledge"] = result.need_knowledge
        state["intent_analysis"] = result.analysis

        logger.info("[意图分析] 需要商品推荐: %s", result.need_products)
        logger.info("[意图分析] 需要知识资料: %s", result.need_knowledge)
        logger.info("[意图分析] 分析说明: %s", result.analysis)

        if result.need_products and result.need_knowledge:
            logger.info("[意图分析] 意图类型: 混合需求（商品推荐 + 知识资料）")
        elif result.need_products:
            logger.info("[意图分析] 意图类型: 纯商品推荐")
        elif result.need_knowledge:
            logger.info("[意图分析] 意图类型: 纯知识查询")
        else:
            logger.warning("[意图分析] 意图类型: 无法判断，默认仅推荐商品")
            state["need_products"] = True

        logger.info("=" * 60)
        return state

    except Exception as exc:
        logger.exception("[意图分析] 失败，回退到默认行为（仅商品推荐）")
        state["need_products"] = True
        state["need_knowledge"] = False
        state["intent_analysis"] = f"意图分析失败({exc})，默认仅推荐商品"
        logger.info("=" * 60)
        return state


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    test_cases = [
        "推荐一双适合秋冬通勤的黑色皮鞋",
        "真皮鞋怎么保养",
        "推荐一双跑鞋，顺便说下跑鞋怎么选",
        "宽脚适合什么鞋型",
        "推荐一双白色运动鞋",
    ]

    for query in test_cases:
        state: RecState = {"query": query}
        result = analyze_intent_node(state)
        print(f"\n问题: {query}")
        print(f"  need_products: {result.get('need_products')}")
        print(f"  need_knowledge: {result.get('need_knowledge')}")
        print(f"  analysis: {result.get('intent_analysis')}")
