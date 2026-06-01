import logging
import pickle
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as settings  # noqa: E402
from src.retriever.product_documents import (  # noqa: E402
    build_product_content,
    chinese_tokenize,
    get_document_key,
)
from src.shared import create_chroma_collection  # noqa: E402


logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    document: Document
    sources: set[str] = field(default_factory=set)
    chroma_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@lru_cache(maxsize=1)
def load_chroma_index():
    """
    从磁盘加载 Chroma 向量索引。

    使用 DashScope embedding 模型初始化，与索引构建时保持一致。
    Chroma 索引在 embedding_pipeline 中构建并保存到 CHROMA_INDEX_PATH。
    """
    return create_chroma_collection(settings.CHROMA_COLLECTION_NAME)


@lru_cache(maxsize=1)
def load_bm25_index() -> Any:
    if not settings.BM25_INDEX_PATH.exists():
        raise FileNotFoundError(f"BM25 index not found at {settings.BM25_INDEX_PATH}")

    with open(settings.BM25_INDEX_PATH, "rb") as file:
        payload = pickle.load(file)

    if isinstance(payload, dict) and "tokenized_corpus" in payload:
        from rank_bm25 import BM25Okapi

        return {
            "documents": payload["documents"],
            "bm25": BM25Okapi(payload["tokenized_corpus"]),
            "version": payload.get("version", 2),
        }

    logger.warning("检测到旧版 BM25 索引，建议重新运行 embedding_pipeline 生成分词索引")
    return payload


@lru_cache(maxsize=1)
def load_cross_encoder():
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    return HuggingFaceCrossEncoder(model_name=settings.CROSS_ENCODER_MODEL_NAME)


def retrieve_from_chroma(query: str) -> list[RetrievalCandidate]:
    """
    使用 Chroma 进行稠密向量（语义）检索。

    Chroma 负责捕捉语义相似度 —— 例如用户查"秋冬通勤外套"，
    能匹配到"加厚西装领大衣"这种关键词不完全一致但语义相近的商品。

    选择 Chroma（而非 FAISS）的原因是：
    - 未来计划换数据库、更新知识库，元数据字段会增加
    - Chroma 原生支持 metadata filtering，扩展成本更低
    - 知识库模块已使用 Chroma，统一存储降低维护成本
    """
    vectorstore = load_chroma_index()
    top_k = getattr(settings, "CHROMA_RETRIEVER_TOP_K", 20)
    threshold = getattr(settings, "CHROMA_SIMILARITY_THRESHOLD", 0.2)
    results: list[tuple[Document, float | None]] = []

    if hasattr(vectorstore, "similarity_search_with_relevance_scores"):
        scored_docs = vectorstore.similarity_search_with_relevance_scores(
            query,
            k=top_k,
        )
        results = [
            (document, float(score))
            for document, score in scored_docs
            if score is not None and float(score) >= threshold
        ]
    else:
        documents = vectorstore.similarity_search(query, k=top_k)
        results = [(document, None) for document in documents]

    candidates: list[RetrievalCandidate] = []
    for document, score in results:
        candidates.append(
            RetrievalCandidate(
                document=document,
                sources={"chroma"},
                chroma_score=score,
            )
        )

    return candidates


def retrieve_from_bm25(query: str) -> list[RetrievalCandidate]:
    index = load_bm25_index()
    top_k = getattr(settings, "BM25_RETRIEVER_TOP_K", 20)
    min_score = getattr(settings, "BM25_MIN_SCORE", 0)

    if isinstance(index, dict) and "bm25" in index:
        query_tokens = chinese_tokenize(query)
        if not query_tokens:
            return []

        scores = index["bm25"].get_scores(query_tokens)
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda item: scores[item],
            reverse=True,
        )

        candidates: list[RetrievalCandidate] = []
        for item in ranked_indexes[:top_k]:
            score = float(scores[item])
            if score <= min_score:
                continue

            candidates.append(
                RetrievalCandidate(
                    document=index["documents"][item],
                    sources={"bm25"},
                    bm25_score=score,
                )
            )

        return candidates

    if hasattr(index, "invoke"):
        index.k = top_k
        return [
            RetrievalCandidate(document=document, sources={"bm25"})
            for document in index.invoke(query)
        ]

    raise TypeError(f"Unsupported BM25 index type: {type(index)!r}")


def rrf_fusion(
    *candidate_groups: list[RetrievalCandidate],
    k: int | None = None,
) -> list[RetrievalCandidate]:
    """
    使用 RRF (Reciprocal Rank Fusion) 算法融合多个检索源的候选结果。

    RRF 公式：RRF_score(d) = Σ_{r ∈ retrievers} 1 / (k + rank_r(d))

    其中 k 是平滑常数（默认 60），rank_r(d) 是文档 d 在检索源 r
    中的排名（1-indexed）。RRF 不依赖绝对分数，只依赖相对排名，
    因此可以公平地融合 Chroma（余弦相似度）和 BM25（词频分数）两种
    不同量纲的检索结果。

    同文档在多个来源中出现时：
    - RRF 分数累加（出现在多个来源且排名靠前的文档得分最高）
    - 各路原始分数保留各自的最大值
    - sources 取并集
    """
    if k is None:
        k = getattr(settings, "RRF_K", 60)

    merged: dict[str, RetrievalCandidate] = {}

    for group in candidate_groups:
        for rank, candidate in enumerate(group, start=1):
            key = get_document_key(candidate.document)
            rrf_contribution = 1.0 / (k + rank)
            existing = merged.get(key)

            if existing is None:
                candidate.rrf_score = rrf_contribution
                merged[key] = candidate
            else:
                existing.rrf_score = (existing.rrf_score or 0) + rrf_contribution
                existing.chroma_score = _max_optional(
                    existing.chroma_score,
                    candidate.chroma_score,
                )
                existing.bm25_score = _max_optional(
                    existing.bm25_score,
                    candidate.bm25_score,
                )
                existing.sources.update(candidate.sources)

    return sorted(
        merged.values(),
        key=lambda c: c.rrf_score if c.rrf_score is not None else float("-inf"),
        reverse=True,
    )


def rerank_candidates(
    query: str,
    candidates: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    """
    使用 Cross-Encoder 对融合后的候选做细粒度语义重排。

    Cross-Encoder 将 (query, document) 成对输入 Transformer，
    输出相关性分数。相比双塔模型（Chroma embedding），Cross-Encoder
    能捕捉 query-document 之间的细粒度交互，但计算成本更高，
    因此只对 RRF 融合后的候选集（而非全量）进行重排。
    """
    if not candidates:
        return []

    top_n = getattr(settings, "RERANK_TOP_N", 5)
    cross_encoder = load_cross_encoder()
    pairs = [
        (
            query,
            build_product_content(candidate.document.metadata or {})
            or candidate.document.page_content,
        )
        for candidate in candidates
    ]
    scores = cross_encoder.score(pairs)

    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = float(score)

    return sorted(
        candidates,
        key=lambda item: item.rerank_score
        if item.rerank_score is not None
        else float("-inf"),
        reverse=True,
    )[:top_n]


def retrieve_products(
    query: str,
    filters: dict[str, object] | None = None,
) -> list[Document]:
    """
    混合检索主入口：Chroma + BM25 → RRF 融合 → Cross-Encoder 重排。

    流程：
    1. Chroma 稠密检索 — 语义匹配（Top-K）
    2. BM25 稀疏检索 — 关键词匹配（Top-K）
    3. RRF 融合去重 — 基于排名融合，不依赖绝对分数
    4. 元数据过滤 — SQL 叠加过滤（价格/品牌/材质/季节）
    5. Cross-Encoder 重排 — 细粒度语义打分（Top-N）
    6. 如果 Cross-Encoder 失败，回退到 RRF 分数排序
    """
    chroma_candidates = retrieve_from_chroma(query)
    bm25_candidates = retrieve_from_bm25(query)
    candidates = rrf_fusion(chroma_candidates, bm25_candidates)

    if not candidates:
        return []

    try:
        ranked_candidates = rerank_candidates(query, candidates)
    except Exception:
        logger.exception("Cross-Encoder 重排失败，回退到 RRF 分数排序")
        ranked_candidates = sorted(
            candidates,
            key=lambda c: c.rrf_score if c.rrf_score is not None else float("-inf"),
            reverse=True,
        )[: getattr(settings, "RERANK_TOP_N", 5)]

    docs = [_with_retrieval_metadata(candidate) for candidate in ranked_candidates]

    # 元数据过滤
    if filters:
        docs = apply_metadata_filters(docs, filters)

    return docs


def parse_query_filters(query: str) -> dict[str, object]:
    """
    从用户查询中提取结构化过滤条件。

    使用规则 + 正则，不依赖 LLM。
    支持的过滤维度：价格区间、品牌、材质、季节、性别。

    Examples:
        "300元以下的冬季外套" → {"price_max": 300, "season": "冬"}
        "海澜之家衬衫500以内" → {"brand": "海澜之家", "price_max": 500}
        "夏天穿的棉质连衣裙" → {"season": "夏", "material": "棉"}
    """
    import re

    filters: dict[str, object] = {}

    # 价格：XXX元以下 / XXX以内 / XXX-XXX元 / XXX以下
    price_below = re.search(r"(\d+)\s*元?\s*(?:以下|以内|内)", query)
    if price_below:
        filters["price_max"] = float(price_below.group(1))

    price_above = re.search(r"(\d+)\s*元?\s*(?:以上|以上)", query)
    if price_above:
        filters["price_min"] = float(price_above.group(1))

    price_range = re.search(r"(\d+)\s*[-~至到]\s*(\d+)\s*元?", query)
    if price_range:
        filters["price_min"] = float(price_range.group(1))
        filters["price_max"] = float(price_range.group(2))

    # 季节
    for keyword, season in [
        ("春", "春"), ("夏", "夏"), ("秋", "秋"),
        ("冬", "冬"), ("春夏", "春夏"), ("秋冬", "秋冬"),
        ("春秋", "春秋"), ("夏天", "夏"), ("冬天", "冬"),
        ("春季", "春"), ("夏季", "夏"), ("秋季", "秋"), ("冬季", "冬"),
    ]:
        if keyword in query and "season" not in filters:
            filters["season"] = season
            break

    # 品牌（从常见品牌库匹配）
    COMMON_BRANDS = [
        "伊芙丽", "太平鸟", "欧时力", "优衣库", "ZARA", "H&M",
        "海澜之家", "七匹狼", "劲霸", "柒牌", "雅戈尔",
        "百丽", "达芙妮", "红蜻蜓", "奥康", "NIKE", "Adidas",
        "安踏", "李宁", "特步", "爱慕", "曼妮芬", "古今",
        "巴拉巴拉", "江南布衣", "MO&Co", "地素", "ONLY",
    ]
    for brand in COMMON_BRANDS:
        if brand.lower() in query.lower() and "brand" not in filters:
            filters["brand"] = brand
            break

    # 材质
    MATERIALS = [
        "棉", "麻", "丝", "羊毛", "羊绒", "真丝", "雪纺",
        "牛仔", "皮革", "皮", "羽绒", "羊皮", "牛皮",
        "聚酯纤维", "莫代尔", "亚麻", "灯芯绒", "针织",
    ]
    for mat in sorted(MATERIALS, key=len, reverse=True):
        if mat in query and "material" not in filters:
            filters["material"] = mat
            break

    # 性别
    if any(w in query for w in ["女", "女式", "女士", "女款"]):
        filters["gender"] = "女"
    elif any(w in query for w in ["男", "男式", "男士", "男款"]):
        filters["gender"] = "男"

    return filters


def apply_metadata_filters(
    documents: list[Document],
    filters: dict[str, object],
) -> list[Document]:
    """
    在检索结果上叠加 SQL 元数据过滤。

    通过 ProductRepo.filter_by_ids 利用 SQLite 的索引做高效过滤，
    保留向量检索的语义相关性排序，只排除不符合条件的商品。

    Args:
        documents: 向量检索返回的 Document 列表（已排序）
        filters: parse_query_filters 的输出

    Returns:
        过滤后的 Document 列表（保持原顺序）
    """
    if not filters or not documents:
        return documents

    doc_ids = [doc.metadata.get("id", "") for doc in documents]
    doc_ids = [did for did in doc_ids if did]

    if not doc_ids:
        return documents

    try:
        from src.database.product_repo import ProductRepo

        repo = ProductRepo()

        filtered_rows = repo.filter_by_ids(
            product_ids=doc_ids,
            price_min=float(filters["price_min"]) if "price_min" in filters else None,
            price_max=float(filters["price_max"]) if "price_max" in filters else None,
            brand=str(filters["brand"]) if "brand" in filters else None,
            material=str(filters["material"]) if "material" in filters else None,
            season=str(filters["season"]) if "season" in filters else None,
            gender=str(filters["gender"]) if "gender" in filters else None,
        )

        kept_ids = {row["id"] for row in filtered_rows}
        return [doc for doc in documents if doc.metadata.get("id") in kept_ids]

    except Exception:
        logger.exception("元数据过滤失败，返回未过滤结果")
        return documents


def create_bm25_payload(documents: list[Document]) -> dict[str, Any]:
    return {
        "version": 2,
        "documents": documents,
        "tokenized_corpus": [
            chinese_tokenize(document.page_content) for document in documents
        ],
        "tokenizer": "jieba_or_cjk_fallback",
    }


def _with_retrieval_metadata(candidate: RetrievalCandidate) -> Document:
    metadata = dict(candidate.document.metadata or {})
    metadata["_retrieval"] = {
        "sources": sorted(candidate.sources),
        "chroma_score": candidate.chroma_score,
        "bm25_score": candidate.bm25_score,
        "rrf_score": candidate.rrf_score,
        "rerank_score": candidate.rerank_score,
    }

    content = build_product_content(metadata) or candidate.document.page_content
    return Document(
        page_content=content,
        metadata=metadata,
        id=getattr(candidate.document, "id", None),
    )


def _fallback_rank(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    return sorted(candidates, key=_fallback_score, reverse=True)


def _fallback_score(candidate: RetrievalCandidate) -> float:
    score = 0.0

    if candidate.chroma_score is not None:
        score += candidate.chroma_score
    if candidate.bm25_score is not None:
        score += candidate.bm25_score

    return score


def _max_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left

    return max(left, right)


if __name__ == "__main__":
    docs = retrieve_products("推荐一件适合秋冬通勤的外套")
    print("Retrieved documents:", len(docs))
    for doc in docs:
        print(doc.metadata.get("_retrieval"), doc.page_content[:120])
