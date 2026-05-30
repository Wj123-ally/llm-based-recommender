from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.structured_query import Comparator, Comparison

try:
    from langchain.prompts import PromptTemplate
except ImportError:
    from langchain_core.prompts import PromptTemplate

try:
    from langchain_classic.chains.query_constructor.schema import AttributeInfo
except ImportError:
    from langchain.chains.query_constructor.schema import AttributeInfo


class CustomChromaTranslator(ChromaTranslator):
    """
    自定义 Chroma 查询转换器。

    在保留 ChromaTranslator 默认能力的基础上，额外支持 LIKE。
    """

    allowed_comparators = [*ChromaTranslator.allowed_comparators, Comparator.LIKE]

    def visit_comparison(self, comparison: Comparison) -> dict:
        """
        把 LIKE 转换成 Chroma 可执行的过滤条件。
        """
        if comparison.comparator == Comparator.LIKE:
            return {comparison.attribute: {"$in": [comparison.value]}}

        return super().visit_comparison(comparison)


ATTRIBUTE_INFO = [
    AttributeInfo(
        name="title",
        description="商品标题",
        type="string",
    ),
    AttributeInfo(
        name="industry",
        description="行业，例如 服饰时尚",
        type="string",
    ),
    AttributeInfo(
        name="category1",
        description="一级类目，例如 女装、男装、鞋靴、箱包",
        type="string",
    ),
    AttributeInfo(
        name="category2",
        description="二级类目，例如 半身裙、针织衫、运动鞋",
        type="string",
    ),
    AttributeInfo(
        name="category3",
        description="三级类目",
        type="string",
    ),
    AttributeInfo(
        name="category4",
        description="四级类目",
        type="string",
    ),
    AttributeInfo(
        name="attributes",
        description="商品属性文本，可能包含品牌、季节、风格、材质、人群、功能功效等",
        type="string",
    ),
]

DOC_CONTENT = "中文电商服装商品资料，包含商品标题、类目、风格、季节、材质、人群和属性描述。"


def get_metadata_info():
    """
    返回 Self-Query 需要的 metadata 字段说明和文档内容说明。
    """
    return ATTRIBUTE_INFO, DOC_CONTENT


def create_rag_template() -> PromptTemplate:
    """
    创建中文服装推荐 RAG prompt 模板。
    """
    template = """
你是一个中文服装导购助手。

你只能基于给定商品资料回答用户问题，不要编造商品。
最多推荐 3 款商品，每款说明推荐理由。
如果资料不足，请说明无法找到完全匹配商品。
请严格按编号格式输出推荐商品，编号必须使用 "1. "、"2. "、"3. "。
每个编号只介绍一个商品，介绍完该商品就结束该编号段落，不要把多个商品写在同一段里。
推荐顺序必须和商品资料中的商品顺序一致，方便前端在每个商品段落后展示对应图片。

用户需求：
{query}

商品资料：
{docs}

请用自然、简洁的中文给出推荐回答。
"""

    return PromptTemplate(
        template=template,
        input_variables=["docs", "query"],
    )
