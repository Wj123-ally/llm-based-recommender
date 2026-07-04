from __future__ import annotations

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
from src.shared import create_embedding_model  # noqa: E402

logger = logging.getLogger(__name__)

KNOWN_SHOE_TYPES = {
    "网鞋",
    "皮鞋",
    "单鞋",
    "乐福鞋",
    "运动鞋",
    "跑步鞋",
    "凉鞋",
    "拖鞋",
    "板鞋",
    "帆布鞋",
    "靴子",
    "马丁靴",
    "高跟鞋",
}


@dataclass
class RetrievalCandidate:
    document: Document
    sources: set[str] = field(default_factory=set)
    dense_score: float | None = None
    bm25_score: float | None = None
    multimodal_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@lru_cache(maxsize=1)
def load_dense_embedding_model() -> Any:
    return create_embedding_model()


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

    logger.warning("Detected legacy BM25 index; rebuild with embedding.py")
    return payload


@lru_cache(maxsize=1)
def load_cross_encoder() -> Any:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    return HuggingFaceCrossEncoder(model_name=settings.CROSS_ENCODER_MODEL_NAME)


def retrieve_from_dense_vectors(
    query: str,
    filters: dict[str, object] | None = None,
) -> list[RetrievalCandidate]:
    top_k = getattr(settings, "DENSE_RETRIEVER_TOP_K", 20)
    search_top_k = _expanded_filter_top_k(top_k, filters)
    threshold = getattr(settings, "DENSE_SIMILARITY_THRESHOLD", 0.2)
    query_embedding = load_dense_embedding_model().embed_query(query)

    from src.retriever.milvus_store import search_documents

    candidates: list[RetrievalCandidate] = []
    for document, score in search_documents(
        settings.MILVUS_TEXT_COLLECTION_NAME,
        query_embedding,
        search_top_k,
    ):
        if score is not None and score < threshold:
            continue
        candidates.append(
            RetrievalCandidate(
                document=document,
                sources={"milvus_text"},
                dense_score=score,
            )
        )
    if filters:
        candidates = apply_candidate_filters(candidates, filters)
    return candidates[:top_k]


def retrieve_from_bm25(
    query: str,
    filters: dict[str, object] | None = None,
) -> list[RetrievalCandidate]:
    index = load_bm25_index()
    top_k = getattr(settings, "BM25_RETRIEVER_TOP_K", 20)
    min_score = getattr(settings, "BM25_MIN_SCORE", 0)

    if isinstance(index, dict) and "bm25" in index:
        query_tokens = chinese_tokenize(query)
        if not query_tokens:
            return []

        scores = index["bm25"].get_scores(query_tokens)
        documents = index["documents"]
        eligible_indexes = range(len(scores))
        if filters:
            eligible_indexes = [
                item
                for item in eligible_indexes
                if metadata_matches_filters(documents[item].metadata or {}, filters)
            ]
        ranked_indexes = sorted(
            eligible_indexes,
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
                    document=documents[item],
                    sources={"bm25"},
                    bm25_score=score,
                )
            )
        return candidates

    if hasattr(index, "invoke"):
        index.k = top_k
        candidates = [
            RetrievalCandidate(document=document, sources={"bm25"})
            for document in index.invoke(query)
        ]
        if filters:
            candidates = apply_candidate_filters(candidates, filters)
        return candidates

    raise TypeError(f"Unsupported BM25 index type: {type(index)!r}")


def retrieve_from_multimodal_images(
    query: str,
    filters: dict[str, object] | None = None,
) -> list[RetrievalCandidate]:
    if not getattr(settings, "ENABLE_MULTIMODAL_RETRIEVER", False):
        return []

    try:
        from src.retriever.multimodal_retriever import retrieve_from_product_images

        top_k = getattr(settings, "MULTIMODAL_RETRIEVER_TOP_K", 20)
        search_top_k = _expanded_filter_top_k(top_k, filters)
        candidates = [
            RetrievalCandidate(
                document=document,
                sources={"multimodal_image"},
                multimodal_score=score,
            )
            for document, score in retrieve_from_product_images(query, top_k=search_top_k)
        ]
        if filters:
            candidates = apply_candidate_filters(candidates, filters)
        return candidates[:top_k]
    except Exception as exc:
        logger.warning("Multimodal image retriever unavailable; skipped: %s", exc)
        return []


def rrf_fusion(
    *candidate_groups: list[RetrievalCandidate],
    k: int | None = None,
) -> list[RetrievalCandidate]:
    if k is None:
        k = getattr(settings, "RRF_K", 60)

    merged: dict[str, RetrievalCandidate] = {}
    for group in candidate_groups:
        for rank, candidate in enumerate(group, start=1):
            key = get_document_key(candidate.document)
            contribution = 1.0 / (k + rank)
            existing = merged.get(key)
            if existing is None:
                candidate.rrf_score = contribution
                merged[key] = candidate
                continue

            existing.rrf_score = (existing.rrf_score or 0) + contribution
            existing.dense_score = _max_optional(
                existing.dense_score,
                candidate.dense_score,
            )
            existing.bm25_score = _max_optional(
                existing.bm25_score,
                candidate.bm25_score,
            )
            existing.multimodal_score = _max_optional(
                existing.multimodal_score,
                candidate.multimodal_score,
            )
            existing.sources.update(candidate.sources)

    return sorted(
        merged.values(),
        key=lambda candidate: candidate.rrf_score
        if candidate.rrf_score is not None
        else float("-inf"),
        reverse=True,
    )


def rerank_candidates(
    query: str,
    candidates: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
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
    exclude_ids: list[str] | None = None,
) -> list[Document]:
    """
    执行混合检索（向量 + BM25 + 多模态），并返回排序后的商品文档。

    Args:
        query: 查询字符串
        filters: 结构化过滤条件（season, material, brand等）
        exclude_ids: 需要排除的商品ID列表（用于避免重复推荐）

    Returns:
        排序后的商品文档列表
    """
    dense_candidates = retrieve_from_dense_vectors(query, filters)
    bm25_candidates = retrieve_from_bm25(query, filters)
    multimodal_candidates = retrieve_from_multimodal_images(query, filters)
    candidates = rrf_fusion(dense_candidates, bm25_candidates, multimodal_candidates)
    if filters:
        candidates = apply_candidate_filters(candidates, filters)

    # 排除已推荐的商品
    if exclude_ids:
        initial_count = len(candidates)
        candidates = [
            c for c in candidates
            if c.document.metadata.get("id") not in exclude_ids
        ]
        if initial_count > len(candidates):
            logger.info(
                "[商品检索] 排除已推荐商品: 排除前%d个, 排除后%d个",
                initial_count,
                len(candidates)
            )

    if not candidates:
        return []

    try:
        ranked_candidates = rerank_candidates(query, candidates)
    except Exception:
        logger.exception("Cross-Encoder rerank failed; falling back to RRF")
        ranked_candidates = sorted(
            candidates,
            key=lambda item: item.rrf_score
            if item.rrf_score is not None
            else float("-inf"),
            reverse=True,
        )[: getattr(settings, "RERANK_TOP_N", 5)]

    docs = [_with_retrieval_metadata(candidate) for candidate in ranked_candidates]
    if filters and _can_use_sql_filter(filters):
        docs = apply_metadata_filters(docs, filters)
    return docs


def _can_use_sql_filter(filters: dict[str, object]) -> bool:
    sql_filter_keys = {"brand", "material", "season"}
    return bool(filters) and set(filters).issubset(sql_filter_keys)


def _expanded_filter_top_k(
    top_k: int,
    filters: dict[str, object] | None,
) -> int:
    if not filters:
        return top_k

    multiplier = max(1, int(getattr(settings, "FILTERED_RECALL_MULTIPLIER", 5)))
    min_k = max(top_k, int(getattr(settings, "FILTERED_RECALL_MIN_K", 100)))
    max_k = max(min_k, int(getattr(settings, "FILTERED_RECALL_MAX_K", 200)))
    return min(max(top_k * multiplier, min_k), max_k)


def parse_query_filters(query: str) -> dict[str, object]:
    filters: dict[str, object] = {}

    for keyword, season in [
        ("春夏", "春夏"),
        ("秋冬", "秋冬"),
        ("春秋", "春秋"),
        ("夏天", "夏"),
        ("冬天", "冬"),
        ("春季", "春"),
        ("夏季", "夏"),
        ("秋季", "秋"),
        ("冬季", "冬"),
        ("春", "春"),
        ("夏", "夏"),
        ("秋", "秋"),
        ("冬", "冬"),
    ]:
        if keyword in query and "season" not in filters:
            filters["season"] = season
            break

    for mat in sorted(
        ["真皮", "牛皮", "羊皮", "皮革", "网面", "帆布", "绒面", "橡胶"],
        key=len,
        reverse=True,
    ):
        if mat in query and "material" not in filters:
            filters["material"] = mat
            break

    if any(word in query for word in ["女", "女式", "女士", "女款"]):
        filters["gender"] = "女"
    elif any(word in query for word in ["男", "男式", "男士", "男款"]):
        filters["gender"] = "男"

    include_colors = _parse_include_colors(query)
    exclude_colors = _parse_exclude_colors(query)
    if include_colors:
        filters["include_colors"] = include_colors
    if exclude_colors:
        filters["exclude_colors"] = exclude_colors

    include_shoe_types = _parse_include_shoe_types(query)
    exclude_shoe_types = _parse_exclude_shoe_types(query)
    if include_shoe_types:
        filters["include_shoe_types"] = include_shoe_types
    if exclude_shoe_types:
        filters["exclude_shoe_types"] = exclude_shoe_types

    return filters


def _parse_include_colors(query: str) -> list[str]:
    color_groups: list[str] = []
    if any(word in query for word in ["浅颜色", "浅色", "浅色系", "亮色", "清爽颜色"]):
        color_groups.extend(
            [
                "白色",
                "白",
                "米色",
                "米白",
                "杏色",
                "粉色",
                "浅粉",
                "浅蓝",
                "浅灰",
                "银色",
                "香槟",
                "卡其",
            ]
        )

    for color in [
        "白色",
        "米色",
        "米白",
        "杏色",
        "粉色",
        "浅粉",
        "浅蓝",
        "浅灰",
        "银色",
        "卡其",
        "灰色",
        "蓝色",
        "棕色",
        "红色",
    ]:
        if color in query:
            color_groups.append(color)

    return _dedupe(color_groups)


def _parse_exclude_colors(query: str) -> list[str]:
    colors: list[str] = []
    if any(
        phrase in query
        for phrase in [
            "不要黑色",
            "不要黑",
            "不想要黑色",
            "不想要黑",
            "不是黑色",
            "非黑色",
            "排除黑色",
            "不要黑色的",
        ]
    ):
        colors.extend(["黑色", "黑"])

    if any(word in query for word in ["浅颜色", "浅色", "浅色系"]):
        colors.extend(["黑色", "黑", "深色", "深灰", "藏青", "深棕"])

    return _dedupe(colors)


def _parse_include_shoe_types(query: str) -> list[str]:
    shoe_types = []
    for shoe_type in sorted(KNOWN_SHOE_TYPES, key=len, reverse=True):
        if shoe_type in query and not _is_negated(query, shoe_type):
            shoe_types.append(shoe_type)
    return _dedupe(shoe_types)


def _parse_exclude_shoe_types(query: str) -> list[str]:
    shoe_types = []
    for shoe_type in sorted(KNOWN_SHOE_TYPES, key=len, reverse=True):
        if _is_negated(query, shoe_type):
            shoe_types.append(shoe_type)
    return _dedupe(shoe_types)


def _is_negated(query: str, value: str) -> bool:
    if any(prefix + value in query for prefix in ["不要", "不想要", "不是", "非", "排除"]):
        return True

    start = query.find(value)
    if start < 0:
        return False

    nearby_after = query[start + len(value) : start + len(value) + 5]
    return any(
        word in nearby_after
        for word in ["太闷", "闷", "不适合", "不舒服", "不透气", "不太适合"]
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def apply_candidate_filters(
    candidates: list[RetrievalCandidate],
    filters: dict[str, object],
) -> list[RetrievalCandidate]:
    if not filters or not candidates:
        return candidates

    initial_count = len(candidates)
    filtered: list[RetrievalCandidate] = []
    for candidate in candidates:
        metadata = candidate.document.metadata or {}
        if not metadata_matches_filters(metadata, filters):
            continue
        filtered.append(candidate)

    # 诊断日志：当过滤导致空结果时输出警告
    if len(filtered) == 0 and initial_count > 0:
        logger.warning(
            "[过滤诊断] 所有%d个候选被过滤掉! 过滤条件: %s",
            initial_count,
            filters
        )
        # 采样输出前3个商品的相关字段帮助调试
        for i, candidate in enumerate(candidates[:3], 1):
            meta = candidate.document.metadata or {}
            logger.warning(
                "  候选商品%d: brand='%s', season='%s', material='%s', title='%s'",
                i,
                meta.get('brand', ''),
                meta.get('season', ''),
                meta.get('material', ''),
                (meta.get('title', '') or '')[:40]
            )
    elif len(filtered) < initial_count:
        logger.info(
            "[过滤诊断] 过滤前: %d个候选, 过滤后: %d个候选, 过滤条件: %s",
            initial_count,
            len(filtered),
            filters
        )

    return filtered


def merge_product_filters(
    llm_filters: dict[str, object] | None,
    rule_filters: dict[str, object] | None,
) -> dict[str, object]:
    merged: dict[str, object] = {}

    for source in [llm_filters or {}, rule_filters or {}]:
        for key in ["gender", "brand", "material", "season"]:
            value = source.get(key)
            if value:
                merged[key] = str(value)

        for key in [
            "include_colors",
            "exclude_colors",
            "include_shoe_types",
            "exclude_shoe_types",
        ]:
            values = _coerce_filter_list(source.get(key))
            if values:
                merged[key] = _dedupe(_coerce_filter_list(merged.get(key)) + values)

    _prefer_exclusions(merged, "include_colors", "exclude_colors")
    _prefer_exclusions(merged, "include_shoe_types", "exclude_shoe_types")
    _keep_known_shoe_types(merged, "include_shoe_types")
    _keep_known_shoe_types(merged, "exclude_shoe_types")
    return merged


def _coerce_filter_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def _prefer_exclusions(
    filters: dict[str, object],
    include_key: str,
    exclude_key: str,
) -> None:
    excludes = set(_coerce_filter_list(filters.get(exclude_key)))
    if not excludes:
        return

    includes = [
        value
        for value in _coerce_filter_list(filters.get(include_key))
        if value not in excludes
    ]
    if includes:
        filters[include_key] = includes
    else:
        filters.pop(include_key, None)


def _keep_known_shoe_types(filters: dict[str, object], key: str) -> None:
    values = [
        value
        for value in _coerce_filter_list(filters.get(key))
        if value in KNOWN_SHOE_TYPES
    ]
    if values:
        filters[key] = values
    else:
        filters.pop(key, None)


def metadata_matches_filters(
    metadata: dict[str, Any],
    filters: dict[str, object],
) -> bool:
    gender = filters.get("gender")
    if gender and not _metadata_contains_any(
        metadata,
        ["gender", "target_user", "title", "content_text"],
        [str(gender)],
    ):
        return False

    for key in ["brand", "material", "season"]:
        value = filters.get(key)
        if value and not _metadata_contains_any(metadata, [key], [str(value)]):
            return False

    include_colors = filters.get("include_colors")
    if isinstance(include_colors, list) and include_colors:
        if not _metadata_contains_any(
            metadata,
            ["color", "text_color", "image_color", "title", "content_text", "tags"],
            [str(color) for color in include_colors],
        ):
            return False

    exclude_colors = filters.get("exclude_colors")
    if isinstance(exclude_colors, list) and exclude_colors:
        if _metadata_contains_any(
            metadata,
            ["color", "text_color", "image_color", "title", "content_text", "tags"],
            [str(color) for color in exclude_colors],
        ):
            return False

    include_shoe_types = filters.get("include_shoe_types")
    if isinstance(include_shoe_types, list) and include_shoe_types:
        if not _metadata_contains_any(
            metadata,
            ["shoe_type", "category_l2", "category_l3", "category_l4", "title", "content_text", "tags"],
            [str(shoe_type) for shoe_type in include_shoe_types],
        ):
            return False

    exclude_shoe_types = filters.get("exclude_shoe_types")
    if isinstance(exclude_shoe_types, list) and exclude_shoe_types:
        if _metadata_contains_any(
            metadata,
            ["shoe_type", "category_l2", "category_l3", "category_l4", "title", "content_text", "tags"],
            [str(shoe_type) for shoe_type in exclude_shoe_types],
        ):
            return False

    return True


def _metadata_contains_any(
    metadata: dict[str, Any],
    fields: list[str],
    needles: list[str],
) -> bool:
    values = [str(metadata.get(field) or "") for field in fields]
    haystack = " ".join(values).lower()
    return any(needle and needle.lower() in haystack for needle in needles)


def apply_metadata_filters(
    documents: list[Document],
    filters: dict[str, object],
) -> list[Document]:
    if not filters or not documents:
        return documents

    doc_ids = [str(doc.metadata.get("id") or "") for doc in documents]
    doc_ids = [doc_id for doc_id in doc_ids if doc_id]
    if not doc_ids:
        return documents

    try:
        from src.database.product_repo import ProductRepo

        repo = ProductRepo()
        filtered_rows = repo.filter_by_ids(
            product_ids=doc_ids,
            brand=str(filters["brand"]) if "brand" in filters else None,
            material=str(filters["material"]) if "material" in filters else None,
            season=str(filters["season"]) if "season" in filters else None,
            gender=str(filters["gender"]) if "gender" in filters else None,
        )
        kept_ids = {row["id"] for row in filtered_rows}
        return [doc for doc in documents if doc.metadata.get("id") in kept_ids]

    except Exception:
        logger.exception("Metadata filtering failed; returning unfiltered results")
        return documents


def _with_retrieval_metadata(candidate: RetrievalCandidate) -> Document:
    metadata = dict(candidate.document.metadata or {})
    metadata["_retrieval"] = {
        "sources": sorted(candidate.sources),
        "dense_score": candidate.dense_score,
        "bm25_score": candidate.bm25_score,
        "multimodal_score": candidate.multimodal_score,
        "rrf_score": candidate.rrf_score,
        "rerank_score": candidate.rerank_score,
    }

    return Document(
        page_content=build_product_content(metadata) or candidate.document.page_content,
        metadata=metadata,
        id=getattr(candidate.document, "id", None),
    )


def _max_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


if __name__ == "__main__":
    docs = retrieve_products("推荐一双黑色通勤皮鞋")
    print("Retrieved documents:", len(docs))
    for doc in docs:
        print(doc.metadata.get("_retrieval"), doc.page_content[:120])
