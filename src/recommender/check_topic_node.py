import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.recommender.state import RecState


class TopicGrade(BaseModel):
    """
    主题相关性判断结果。
    """

    # 只能返回 Yes 或 No
    score: Literal["Yes", "No"] = Field(
        description="用户问题是否和服装、鞋包、穿搭、配饰或商品推荐相关"
    )


def create_llm():
    """
    创建 DashScope / 通义千问兼容的 LangChain 聊天模型。

    后续如果需要更换模型，只需要修改这个函数。
    """
    try:
        from langchain_community.chat_models import ChatTongyi
    except ImportError:
        from langchain_community.chat_models.tongyi import ChatTongyi

    return ChatTongyi(
        model=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
        temperature=0,
    )


def check_topic_node(state: RecState) -> RecState:
    """
    判断用户问题是否属于服装推荐系统的可回答范围。
    """
    query = state["query"]

    system_prompt = """
你是一个中文服装推荐系统的主题判断器。

你的任务只判断用户问题是否和以下主题相关：
- 服装
- 鞋包
- 穿搭
- 配饰
- 商品推荐

如果相关，返回 Yes。
如果不相关，返回 No。

注意：
你只做主题判断，不要回答用户问题，不要推荐商品，不要解释原因。
"""

    llm = create_llm()
    structured_llm = llm.with_structured_output(TopicGrade)

    result = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户问题：{query}"),
        ]
    )

    state["on_topic"] = result.score

    if result.score == "No":
        state["recommendation"] = (
            "抱歉，我只能回答服装、鞋包、穿搭或商品推荐相关的问题。"
        )

    return state


if __name__ == "__main__":
    test_states: list[RecState] = [
        {"query": "推荐一件适合秋冬通勤的小香风外套"},
        {"query": "今天北京天气怎么样"},
    ]

    for state in test_states:
        result = check_topic_node(state)
        print("用户问题:", result["query"])
        print("主题判断:", result["on_topic"])
        print("推荐回答:", result.get("recommendation", ""))
        print("-" * 50)
