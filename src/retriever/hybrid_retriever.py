import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

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


logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    document: Document
    sources: set[str] = field(default_factory=set)
    chroma_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


@lru_cache(maxsize=1)
def initialize_embeddings_model():
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise EnvironmentError("请先设置环境变量 DASHSCOPE_API_KEY")

    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            try:
                from langchain_community.embeddings import DashScopeEmbeddings
            except ImportError:
                from langchain_dashscope import DashScopeEmbeddings

            try:
                return DashScopeEmbeddings(model=settings.DASHSCOPE_EMBEDDING_MODEL)
            except TypeError:
                return DashScopeEmbeddings(
                    model_name=settings.DASHSCOPE_EMBEDDING_MODEL
                )
        except Exception as exc:
            last_error = exc
            logger.warning("初始化 DashScope embedding 失败，第 %s 次重试", attempt)
            if attempt < 3:
                time.sleep(2)

    raise RuntimeError("初始化 DashScope embedding 模型失败") from last_error


@lru_cache(maxsize=1)
def load_chroma_index():
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    vectorstore = Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=initialize_embeddings_model(),
        persist_directory=str(settings.CHROMA_INDEX_PATH),
    )

    try:
        logger.info("Chroma 向量库文档数量：%s", vectorstore._collection.count())
    except Exception:
        logger.exception("读取 Chroma 文档数量失败")

    return vectorstore


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


def merge_candidates(
    *candidate_groups: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    merged: dict[str, RetrievalCandidate] = {}

    for group in candidate_groups:
        for candidate in group:
            key = get_document_key(candidate.document)
            existing = merged.get(key)

            if existing is None:
                merged[key] = candidate
                continue

            existing.sources.update(candidate.sources)
            existing.chroma_score = _max_optional(
                existing.chroma_score,
                candidate.chroma_score,
            )
            existing.bm25_score = _max_optional(
                existing.bm25_score,
                candidate.bm25_score,
            )

    return list(merged.values())


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


def retrieve_products(query: str) -> list[Document]:
    chroma_candidates = retrieve_from_chroma(query)
    bm25_candidates = retrieve_from_bm25(query)
    candidates = merge_candidates(chroma_candidates, bm25_candidates)

    if not candidates:
        return []

    try:
        ranked_candidates = rerank_candidates(query, candidates)
    except Exception:
        logger.exception("Cross-Encoder 重排失败，回退到召回分数排序")
        ranked_candidates = _fallback_rank(candidates)[
            : getattr(settings, "RERANK_TOP_N", 5)
        ]

    return [_with_retrieval_metadata(candidate) for candidate in ranked_candidates]


def create_bm25_payload(documents: list[Document]) -> dict[str, Any]:
    return {
        "version": 2,
        "documents": documents,
        "tokenized_corpus": [
            chinese_tokenize(document.page_content) for document in documents
        ],
        "tokenizer": "jieba_or_cjk_fallback",
    }


def retriever_flow(
    weights: Optional[list[float]] = None,
    k: int = 20,
    top_n: int = 5,
    save_path: Optional[Path] = None,
):
    logger.info(
        "Runtime hybrid retriever uses Chroma + BM25 + Cross-Encoder; "
        "weights/save_path are kept only for compatibility."
    )
    return retrieve_products


def _with_retrieval_metadata(candidate: RetrievalCandidate) -> Document:
    metadata = dict(candidate.document.metadata or {})
    metadata["_retrieval"] = {
        "sources": sorted(candidate.sources),
        "chroma_score": candidate.chroma_score,
        "bm25_score": candidate.bm25_score,
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
