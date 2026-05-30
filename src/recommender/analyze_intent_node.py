"""
意图分析节点。

在话题检查通过后，用 LLM 将用户问题拆解为两类需求：
- need_products：是否需要从商品数据库推荐具体商品
- need_knowledge：是否需要从知识库获取洗护、保养、面料等专业知识

两种需求可以同时存在（混合型问题），也可以只存在一种。
"""

import logging
import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.recommender.state import RecState

logger = logging.getLogger(__name__)


class UserIntent(BaseModel):
    """用户意图分析结果。"""

    need_products: bool = Field(
        description="用户是否需要推荐具体的商品（如衣服、鞋子、包包）。"
        "询问'推荐一件'、'有什么好的'、'帮我选一个'都属于需要商品推荐。"
    )
    need_knowledge: bool = Field(
        description="用户是否需要洗护保养、面料材质、穿搭技巧、选购指南等专业知识。"
        "询问'怎么洗'、'如何保养'、'什么面料好'、'怎么搭配'都属于需要知识。"
        "注意：如果只是纯粹推荐商品而没有询问任何知识类问题，则为 false。"
    )
    analysis: str = Field(
        description="用一句话解释你的分析判断，说明用户意图属于哪种类型，"
        "以及为什么需要/不需要商品推荐和知识库资料。"
    )


def create_llm():
    """创建用于意图分析的 LLM。"""
    try:
        from langchain_community.chat_models import ChatTongyi
    except ImportError:
        from langchain_community.chat_models.tongyi import ChatTongyi

    return ChatTongyi(
        model=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
        temperature=0,
    )


def analyze_intent_node(state: RecState) -> RecState:
    """
    LangGraph 意图分析节点。

    分析用户问题，判断需要调用哪些后端资源：
    - need_products → 调用商品检索
    - need_knowledge → 调用知识库检索

    日志记录完整的分析过程和结果。
    """
    query = state["query"]

    logger.info("=" * 60)
    logger.info("[意图分析] 开始分析用户问题")
    logger.info("[意图分析] 用户问题: %s", query)

    system_prompt = """
你是一个服装推荐系统的意图分析器。

用户的问题可能包含一种或两种需求：

1. 商品推荐需求：
   - 用户明确要求推荐商品（如"推荐一件羽绒服"）
   - 用户描述需求场景希望获得购买建议（如"冬天通勤穿什么"）
   - 用户想要对比或选择商品

2. 知识资料需求：
   - 用户询问洗护方法（如"怎么洗"、"如何保养"）
   - 用户询问面料材质知识（如"纯棉和亚麻哪个透气"）
   - 用户询问穿搭技巧（如"小个子怎么穿显高"）
   - 用户询问选购指南（如"买羽绒服要注意什么"）

对于每个用户问题，你需要判断它是否需要商品推荐、是否需要知识资料。
两者可以同时存在（如"推荐一件睡衣，顺便告诉我怎么洗护"）。

注意：你只做意图判断，不要回答用户问题。
"""

    try:
        llm = create_llm()
        structured_llm = llm.with_structured_output(UserIntent)
        result = structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"分析以下用户问题的意图：{query}"),
            ]
        )

        state["need_products"] = result.need_products
        state["need_knowledge"] = result.need_knowledge
        state["intent_analysis"] = result.analysis

        logger.info("[意图分析] 需要商品推荐: %s", result.need_products)
        logger.info("[意图分析] 需要知识资料: %s", result.need_knowledge)
        logger.info("[意图分析] 分析说明: %s", result.analysis)

        # 分类记录
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
        "推荐一件适合秋冬通勤的小香风外套",
        "羊毛衫怎么洗才不会缩水",
        "推荐一件睡衣，顺便告诉我怎么洗护",
        "羽绒服和棉服哪个更保暖",
        "推荐一双白色运动鞋",
    ]

    for query in test_cases:
        state: RecState = {"query": query}
        result = analyze_intent_node(state)
        print(f"\n问题: {query}")
        print(f"  need_products: {result.get('need_products')}")
        print(f"  need_knowledge: {result.get('need_knowledge')}")
        print(f"  analysis: {result.get('intent_analysis')}")
