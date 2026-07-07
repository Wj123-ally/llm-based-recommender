"""
合并分析节点：主题判断 + 意图分析。

将原来的 check_topic_node 和 analyze_intent_node 合并为一次 LLM 调用，
减少一次 LLM 往返延迟。
"""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.recommender.state import RecState
from src.shared import create_chat_llm

logger = logging.getLogger(__name__)


class CombinedAnalysis(BaseModel):
    """
    合并分析结果：一次调用同时输出主题判断和意图分析。
    """

    on_topic: Literal["Yes", "No"] = Field(
        description="用户问题是否和鞋类商品、鞋型、鞋码、颜色、材质、季节、使用场景、功能、搭配或选购建议相关"
    )
    need_products: bool = Field(
        description="用户是否需要推荐具体鞋类商品，如运动鞋、跑鞋、板鞋、皮鞋、凉鞋、拖鞋、靴子或高跟鞋"
    )
    need_knowledge: bool = Field(
        description="用户是否需要鞋码选择、脚型适配、鞋材质、鞋类清洁保养、场景搭配、选购指南等知识"
    )
    context_mode: Literal["follow_up", "clarification", "new_request"] = Field(
        default="new_request",
        description=(
            "Classify whether this turn should inherit prior shopping context. "
            "follow_up: references prior results; clarification: adds constraints "
            "to the same task; new_request: starts a new shopping task."
        ),
    )
    gender: str | None = Field(
        default=None,
        description="商品硬过滤性别，如 女、男；没有明确要求时为 null",
    )
    brand: str | None = Field(
        default=None,
        description="商品硬过滤品牌；没有明确品牌要求时为 null",
    )
    material: str | None = Field(
        default=None,
        description="商品硬过滤材质，如 真皮、牛皮、网面、帆布；没有明确要求时为 null",
    )
    season: str | None = Field(
        default=None,
        description="商品硬过滤季节，如 春、夏、秋、冬、春夏、秋冬；没有明确要求时为 null",
    )
    include_colors: list[str] = Field(
        default_factory=list,
        description="用户明确想要的颜色；软描述或不确定时留空",
    )
    exclude_colors: list[str] = Field(
        default_factory=list,
        description="用户明确不要或负面评价的颜色",
    )
    include_shoe_types: list[str] = Field(
        default_factory=list,
        description="用户明确想要的鞋型，如 运动鞋、跑步鞋、板鞋、皮鞋、凉鞋、拖鞋、单鞋",
    )
    exclude_shoe_types: list[str] = Field(
        default_factory=list,
        description="用户明确不要或负面评价的鞋型，例如 皮鞋太闷 应输出 皮鞋",
    )
    analysis: str = Field(description="用一句话解释你的分析判断")


def combined_analysis_node(state: RecState) -> RecState:
    """
    合并节点：一次 LLM 调用完成主题判断和意图分析。
    """
    query = state["query"]
    previous_query = state.get("previous_query", "")

    logger.info("=" * 60)
    logger.info("[合并分析] 开始分析用户问题（主题判断 + 意图分析）")
    logger.info("[合并分析] 用户问题: %s", query)
    logger.info("[合并分析] 上一轮 query: %s", previous_query or "(无)")

    system_prompt = """
你是一个鞋类商品推荐系统的分析器。你需要同时完成两个任务。

任务一：主题判断 (on_topic)
判断用户问题是否和以下鞋类主题相关：
- 鞋类商品推荐、商品对比、选购建议
- 鞋型：运动鞋、跑鞋、板鞋、皮鞋、凉鞋、拖鞋、靴子、马丁靴、高跟鞋等
- 鞋码、尺码、脚型、舒适度、跟高、闭合方式
- 鞋面/鞋底材质、颜色、季节、风格
- 使用场景：通勤、运动、跑步、户外、居家、婚礼、正式场合等
- 功能：透气、防滑、保暖、轻便、防水、增高等
- 鞋类搭配建议、鞋类保养与清洁

只有当问题明确和鞋类商品完全无关（如天气、美食、编程、汽车等）时，on_topic 才返回 No。
如果提供上一轮对话上下文且当前问题是简短追问（如“我要女性的”“有没有黑色的”“便宜点”），
应视为对上一轮鞋类推荐的延续，返回 Yes。

任务二：意图分析 (need_products / need_knowledge)
如果 on_topic 为 Yes，进一步分析用户需要哪些服务：

1. 商品推荐需求 (need_products)：
   - 明确要求推荐鞋类商品（如“推荐一双白色运动鞋”）
   - 描述使用场景并希望获得购买建议（如“适合通勤的男士皮鞋”“夏天穿的凉鞋”）
   - 想要对比或筛选鞋类商品
   - 对上一轮推荐结果进行追问或追加条件（如“我要女性的”“有没有黑色的”“便宜点”）

2. 知识资料需求 (need_knowledge)：
   - 询问鞋码怎么选、脚型怎么适配
   - 询问鞋面/鞋底材质、透气、防滑、防水、保暖等功能差异
   - 询问鞋类清洁、保养、收纳方法
   - 询问鞋类搭配、场景选择、选购指南

两者可以同时为 true（如“推荐一双适合跑步的鞋，顺便说下跑鞋怎么选”）。
如果 on_topic 为 No，need_products 和 need_knowledge 都设为 false。
注意：你只做分析判断，不要回答用户问题，不要推荐商品。
"""
    system_prompt += """

任务三：商品检索结构化约束
当 need_products 为 true 时，还要提取商品检索条件：
- gender / brand / material / season：只有用户明确要求时才填写。
- include_colors / include_shoe_types：用户明确想要的颜色或鞋型。
- exclude_colors / exclude_shoe_types：用户明确不要、否定、抱怨或负面评价的颜色或鞋型。
- “皮鞋太闷”“皮鞋不适合”“不想要皮鞋”“不要黑色”这类表达必须进入 exclude，不要进入 include。
- 如果同一个值同时可能进入 include 和 exclude，优先放入 exclude。
- 只提取商品资料中可用于检索的客观条件，不要把“舒适、轻便、得体、适合妈妈”这类软偏好硬塞进 hard filter。
"""

    system_prompt += """

Context inheritance rule:
- Output context_mode="follow_up" only when the user explicitly refers to previously shown products, such as "上一轮", "刚才", "上面这些", "这几个", "那个", "这个", "第2个", "白色那双", "刚才那款", "这几款里".
- Output context_mode="clarification" only when the user adds a constraint but does not name a new shoe type or new product category, such as "女款", "不要黑色", "便宜点", "可爱一点", "尺码大一点".
- Output context_mode="new_request" when the user names a concrete shoe type/product category, scene, or starts over, such as "有洞洞鞋吗", "我喜欢洞洞鞋", "推荐运动鞋", "想看板鞋", "换成靴子", "推荐通勤皮鞋".
- A previous product list is memory for resolving references only. Do not search within previous products unless the current query explicitly says to use the previous/currently shown items.
- If uncertain, prefer "new_request" to avoid leaking old constraints into a new shopping task.
"""

    if previous_query:
        user_message = (
            f"上一轮用户问题：{previous_query}\n"
            f"当前用户追问：{query}\n\n"
            f"请结合上下文分析当前追问。"
        )
    else:
        user_message = f"分析以下用户问题：{query}"

    try:
        llm = create_chat_llm(temperature=0)
        structured_llm = llm.with_structured_output(CombinedAnalysis)
        result = structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
        )

        state["on_topic"] = result.on_topic

        if result.on_topic == "Yes":
            state["need_products"] = result.need_products
            state["need_knowledge"] = result.need_knowledge or result.need_products
        else:
            state["need_products"] = False
            state["need_knowledge"] = False
            state["products"] = ""
            state["documents"] = []
            state["retrieval_state"] = "skipped"
            state["retrieval_source"] = "none"
            state["knowledge_docs"] = ""
            state["knowledge_retrieval_state"] = "skipped"
            state["recommendation"] = (
                "抱歉，我只能回答鞋类商品推荐、鞋型/鞋码/材质/功能、鞋类搭配和选购等相关问题。"
            )

        state["intent_analysis"] = result.analysis
        state["product_filters"] = _extract_product_filters(result)
        state["context_mode"] = _resolve_context_mode(
            query,
            previous_query,
            getattr(result, "context_mode", "new_request"),
        )

        logger.info("[合并分析] 主题判断: %s", result.on_topic)
        logger.info("[合并分析] 需要商品推荐: %s", state["need_products"])
        logger.info("[合并分析] 需要知识资料: %s", state["need_knowledge"])
        logger.info("[合并分析] context_mode: %s", state["context_mode"])
        logger.info("[合并分析] 商品过滤条件: %s", state["product_filters"])
        logger.info("[合并分析] 分析说明: %s", result.analysis)
        logger.info("=" * 60)

        return state

    except Exception as exc:
        logger.exception("[合并分析] 失败，回退到默认行为（商品推荐）")
        state["on_topic"] = "Yes"
        state["need_products"] = True
        state["need_knowledge"] = True
        state["intent_analysis"] = f"分析失败({exc})，默认仅推荐商品"
        state["product_filters"] = {}
        state["context_mode"] = _infer_context_mode(query, previous_query)
        logger.info("=" * 60)
        return state


def _infer_context_mode(query: str, previous_query: str = "") -> str:
    if not previous_query:
        return "new_request"

    text = (query or "").strip()
    if _has_explicit_previous_product_reference(text):
        return "follow_up"
    if _has_new_product_request(text):
        return "new_request"

    clarification_terms = [
        "不要",
        "不想要",
        "女款",
        "男款",
        "便宜",
        "贵",
        "可爱",
        "颜色",
        "尺码",
    ]
    if len(text) < 15 and any(term in text for term in clarification_terms):
        return "clarification"
    return "new_request"


def _resolve_context_mode(query: str, previous_query: str, llm_mode: str) -> str:
    if not previous_query:
        return "new_request"

    text = (query or "").strip()
    if _has_explicit_previous_product_reference(text):
        return "follow_up"
    if _has_new_product_request(text):
        return "new_request"
    if llm_mode in {"follow_up", "clarification", "new_request"}:
        return llm_mode
    return _infer_context_mode(query, previous_query)


def _has_explicit_previous_product_reference(text: str) -> bool:
    previous_reference_terms = [
        "上一轮",
        "上轮",
        "刚才",
        "之前",
        "上面",
        "这些",
        "这几个",
        "那几个",
        "里面",
        "这双",
        "那双",
        "这个",
        "那个",
        "这款",
        "那款",
        "第一个",
        "第二个",
        "第三个",
        "第1个",
        "第2个",
        "第3个",
        "商品1",
        "商品2",
        "商品3",
        "白色那",
        "黑色那",
        "刚才那",
    ]
    return any(term in text for term in previous_reference_terms)


def _has_new_product_request(text: str) -> bool:
    shoe_type_terms = [
        "洞洞鞋",
        "运动鞋",
        "跑鞋",
        "跑步鞋",
        "板鞋",
        "皮鞋",
        "凉鞋",
        "拖鞋",
        "棉拖",
        "靴子",
        "马丁靴",
        "高跟鞋",
        "单鞋",
        "老爹鞋",
        "乐福鞋",
        "玛丽珍",
        "雪地靴",
    ]
    request_terms = [
        "有",
        "有没有",
        "还有",
        "推荐",
        "想看",
        "想要",
        "喜欢",
        "换成",
        "换个",
        "来",
    ]
    if not any(term in text for term in shoe_type_terms):
        return False
    if any(term in text for term in request_terms):
        return True
    return len(text) <= 12


def _extract_product_filters(result: CombinedAnalysis) -> dict[str, object]:
    filters: dict[str, object] = {}
    for key in ["gender", "brand", "material", "season"]:
        value = getattr(result, key, None)
        if value:
            filters[key] = value

    for key in [
        "include_colors",
        "exclude_colors",
        "include_shoe_types",
        "exclude_shoe_types",
    ]:
        values = getattr(result, key, None)
        if values:
            filters[key] = list(values)

    return filters
