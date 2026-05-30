from typing import Any, Literal, TypedDict


class RecState(TypedDict, total=False):
    """
    推荐流程中的状态类型。

    使用 total=False 是因为流程刚开始时，state 里可能只有 query。
    后续节点会逐步补充主题判断、检索结果、推荐回答等字段。
    """

    # 用户输入的原始问题
    query: str

    # 主题判断结果：Yes 表示和商品推荐相关，No 表示不相关
    on_topic: Literal["Yes", "No"]

    # 最终返回给用户的推荐回答
    recommendation: str

    # 后续检索到的商品文本
    products: str

    # 后续检索到的原始文档信息，用于展示标题、图片、类目等 metadata
    documents: list[dict[str, Any]]

    # 后续检索状态：success 表示检索到商品，empty 表示没有检索到商品
    retrieval_state: Literal["success", "empty"]

    # 后续检索来源：chroma 表示向量库检索，hybrid 表示混合检索，none 表示没有检索结果
    retrieval_source: Literal["chroma", "hybrid", "none"]

    # 流程中产生的错误信息
    error: str
