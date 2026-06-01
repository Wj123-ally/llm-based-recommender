import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.recommender.state import RecState
from src.shared import create_chat_llm


class TopicGrade(BaseModel):
    """
    主题相关性判断结果。
    """

    # 只能返回 Yes 或 No
    score: Literal["Yes", "No"] = Field(
        description="用户问题是否和服装、鞋包、穿搭、配饰、商品推荐、洗护保养、面料材质、尺码选购相关"
    )


def check_topic_node(state: RecState) -> RecState:
    """
    判断用户问题是否属于服装推荐系统的可回答范围。
    """
    query = state["query"]
    previous_query = state.get("previous_query", "")

    system_prompt = """
你是一个中文服装推荐系统的主题判断器。

你的任务是判断用户问题是否和以下主题相关：
- 服装、鞋包、配饰
- 穿搭、造型、搭配建议
- 洗护、保养、收纳
- 面料、材质、工艺
- 尺码、版型、选购
- 商品推荐、商品对比

只要用户问题涉及穿衣、买衣、护衣中的任何一个环节，都属于相关。
仅有当问题明确和服装完全无关（如天气、美食、编程、汽车等）时，才返回 No。

重要：如果提供上一轮对话的上下文且当前问题是简短追问（如"我要女性的"），
应视为与服装相关（因为是对上一轮服装推荐的延续），返回 Yes。

注意：
你只做主题判断，不要回答用户问题，不要推荐商品，不要解释原因。
"""

    llm = create_chat_llm(temperature=0)
    structured_llm = llm.with_structured_output(TopicGrade)

    if previous_query:
        user_message = (
            f"上一轮用户问题：{previous_query}\n"
            f"当前用户追问：{query}\n\n"
            f"请结合上下文判断当前追问是否与服装相关。"
        )
    else:
        user_message = f"用户问题：{query}"

    result = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
    )

    state["on_topic"] = result.score

    if result.score == "No":
        state["recommendation"] = (
            "抱歉，我只能回答服装、鞋包、穿搭、洗护保养、商品推荐等相关的问题。"
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
