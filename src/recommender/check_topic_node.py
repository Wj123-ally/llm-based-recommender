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
        description="用户问题是否和鞋类商品、鞋型、鞋码、颜色、材质、季节、使用场景、功能、搭配或选购建议相关"
    )


def check_topic_node(state: RecState) -> RecState:
    """
    判断用户问题是否属于鞋类商品推荐系统的可回答范围。
    """
    query = state["query"]
    previous_query = state.get("previous_query", "")

    system_prompt = """
你是一个鞋类商品推荐系统的主题判断器。
你的任务是判断用户问题是否和以下鞋类主题相关：
- 鞋类商品推荐、商品对比、选购建议
- 鞋型：运动鞋、跑鞋、板鞋、皮鞋、凉鞋、拖鞋、靴子、马丁靴、高跟鞋等
- 鞋码、尺码、脚型、舒适度、跟高、闭合方式
- 鞋面/鞋底材质、颜色、季节、风格
- 使用场景：通勤、运动、跑步、户外、居家、婚礼、正式场合等
- 功能：透气、防滑、保暖、轻便、防水、增高等
- 鞋类搭配建议、鞋类保养与清洁

只有当问题明确和鞋类商品完全无关（如天气、美食、编程、汽车等）时，才返回 No。

重要：如果提供上一轮对话上下文且当前问题是简短追问（如“我要女性的”“有没有黑色的”“便宜点”），
应视为对上一轮鞋类推荐的延续，返回 Yes。

注意：
你只做主题判断，不要回答用户问题，不要推荐商品，不要解释原因。
"""

    llm = create_chat_llm(temperature=0)
    structured_llm = llm.with_structured_output(TopicGrade)

    if previous_query:
        user_message = (
            f"上一轮用户问题：{previous_query}\n"
            f"当前用户追问：{query}\n\n"
            f"请结合上下文判断当前追问是否与鞋类商品相关。"
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
            "抱歉，我只能回答鞋类商品推荐、鞋型/鞋码/材质/功能、鞋类搭配和选购等相关问题。"
        )

    return state


if __name__ == "__main__":
    test_states: list[RecState] = [
        {"query": "推荐一双适合秋冬通勤的黑色皮鞋"},
        {"query": "今天北京天气怎么样"},
    ]

    for state in test_states:
        result = check_topic_node(state)
        print("用户问题:", result["query"])
        print("主题判断:", result["on_topic"])
        print("推荐回答:", result.get("recommendation", ""))
        print("-" * 50)
