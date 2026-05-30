import logging
from typing import Any

from langchain_core.documents import Document

from src.recommender.state import RecState
from src.retriever.hybrid_retriever import retrieve_products
from src.retriever.product_documents import build_product_content


logger = logging.getLogger(__name__)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"商品 {index}\n{build_product_content(doc.metadata) or doc.page_content}"
        for index, doc in enumerate(docs, start=1)
    )


def serialize_docs(docs: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "page_content": build_product_content(doc.metadata) or doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


def self_query_retrieve(state: RecState) -> RecState:
    query = state["query"]

    try:
        results = retrieve_products(query)

        if not results:
            state["retrieval_state"] = "empty"
            state["retrieval_source"] = "none"
            return state

        state["retrieval_state"] = "success"
        state["retrieval_source"] = "hybrid"
        state["products"] = format_docs(results)
        state["documents"] = serialize_docs(results)

        return state
    except Exception as exc:
        logger.exception("混合检索失败")
        state["retrieval_state"] = "empty"
        state["retrieval_source"] = "none"
        state["error"] = str(exc)
        return state


if __name__ == "__main__":
    test_state: RecState = {
        "query": "推荐一件适合秋冬通勤的小香风外套",
    }

    result = self_query_retrieve(test_state)
    print("用户问题:", result["query"])
    print("检索状态:", result.get("retrieval_state"))
    print("检索来源:", result.get("retrieval_source"))
    print("错误信息:", result.get("error", ""))
    print("商品文本预览:", result.get("products", "")[:500])
