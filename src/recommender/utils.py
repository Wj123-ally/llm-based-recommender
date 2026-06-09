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

DOC_CONTENT = "电商服装商品资料，包含商品标题、类目、风格、季节、材质、人群和属性描述。"


def get_metadata_info():
    """
    返回 Self-Query 需要的 metadata 字段说明和文档内容说明。
    """
    return ATTRIBUTE_INFO, DOC_CONTENT


def create_rag_template() -> PromptTemplate:
    """
    创建服装推荐 RAG prompt 模板。

    核心设计：明确区分两个信息来源的职责。

    商品资料（docs）
        - 来源：电商商品数据库
        - 职责：用于推荐具体商品（款式、价格、材质、搭配）
        - 规则：只能使用列表中的商品，不得编造

    知识资料（knowledge）
        - 来源：专业资料库（洗护指南、穿搭技巧、面料知识等）
        - 职责：用于回答洗护、保养、面料科普、选购技巧等知识性问题
        - 规则：优先使用知识库内容回答非推荐类问题

    对话历史（previous_query）
        - 如果提供了上一轮对话，说明用户当前是追问或细化需求
        - 应结合上一轮主题进行回答
    """
    template = """
你是一个服装导购助手。你的回答基于两个相互独立的信息来源，请严格按照各自来源的职责使用它们。

━━━━━━━━━━ 对话上下文 ━━━━━━━━━━

{context}

━━━━━━━━━━ 信息源 A：商品资料（来自电商商品数据库）━━━━━━━━━━

用途：推荐具体商品。每个商品附带标题、类目、属性（品牌/材质/风格/季节等）和图片 URL。

使用规则：
- 商品推荐必须且只能基于此来源
- 最多推荐 3 款商品，按资料中的原始顺序编号 "1. "、"2. "、"3. "
- 每个编号只介绍一款商品，介绍完毕即结束该段落
- 如果此来源为空或没有匹配商品，明确告知用户暂无相关商品，不要强行编造
- 编号和顺序必须与商品资料一致，不可自行调整
- 用户提出性别偏好时（如"我要女性的"），只推荐对应性别的商品

━━━━━━━━━━ 信息源 B：知识资料（来自专业资料库）━━━━━━━━━━

用途：提供洗护保养、面料材质、穿搭技巧、选购指南等专业知识。

使用规则：
- 用户询问「怎么洗」「如何保养」「什么材质」「怎样搭配」等问题时，从此来源回答
- 优先使用知识资料中的内容，不编造
- 如果此来源为空（标注为"暂无相关"），明确告知用户该方面暂无资料，不要凭常识硬答
- 可以与商品推荐自然融合（如：推荐商品后附带洗护建议）
- 即使用户没有主动问洗护，在推荐了羊毛/真丝/羽绒等需要特殊护理的商品后，也应主动附上简要洗护提示

━━━━━━━━━━ 用户需求 ━━━━━━━━━━

{query}

━━━━━━━━━━ 商品资料 ━━━━━━━━━━

{docs}

━━━━━━━━━━ 知识资料 ━━━━━━━━━━

{knowledge}

━━━━━━━━━━ 回答要求 ━━━━━━━━━━

1. 先判断用户需求的类型：仅推荐商品？仅咨询知识？还是两者都有？
2. 根据需求类型选择对应的信息来源，不要让两个来源的内容互相混淆
3. 商品推荐部分用编号格式，知识解答部分用自然段落
4. 如果一个问题同时涉及两方面，先推荐商品，再补充洗护/保养/穿搭知识
5. 如果这是对上一轮的追问，先简短回应用户的细化需求，再给出针对性推荐
6. 语言自然流畅，像真人导购在对话
"""

    return PromptTemplate(
        template=template,
        input_variables=["docs", "query", "knowledge", "context"],
    )
