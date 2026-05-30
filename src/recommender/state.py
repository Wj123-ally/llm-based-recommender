from typing import Any, Literal, TypedDict


class RecState(TypedDict, total=False):
    """
    推荐流程中的状态类型。

    使用 total=False 是因为流程刚开始时，state 里可能只有 query。
    后续节点会逐步补充主题判断、检索结果、推荐回答等字段。
    """

    # 用户输入的原始问题
    query: str

    # 主题判断结果：Yes 表示和服装相关，No 表示不相关
    on_topic: Literal["Yes", "No"]

    # 意图分析结果：是否需要商品推荐
    need_products: bool

    # 意图分析结果：是否需要知识库资料（洗护、保养、面料等）
    need_knowledge: bool

    # 意图分析的原始说明（LLM 输出的解释文本，用于日志和调试）
    intent_analysis: str

    # 最终返回给用户的推荐回答
    recommendation: str

    # 检索到的商品文本
    products: str

    # 从知识库检索到的相关文档片段
    knowledge_docs: str

    # 检索到的原始文档信息，用于展示标题、图片、类目等 metadata
    documents: list[dict[str, Any]]

    # 商品检索状态：success 表示检索到商品，empty 表示没有，skipped 表示无需检索
    retrieval_state: Literal["success", "empty", "skipped"]

    # 商品检索来源：chroma 表示向量库检索，hybrid 表示混合检索，none 表示未检索或无结果
    retrieval_source: Literal["chroma", "hybrid", "none"]

    # 知识库检索状态：success / empty / skipped
    knowledge_retrieval_state: Literal["success", "empty", "skipped"]

    # 流程中产生的错误信息
    error: str
