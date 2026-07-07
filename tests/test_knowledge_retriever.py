import pytest

langchain_available = False
try:
    from langchain_core.documents import Document

    import src.knowledge_base.knowledge_retriever as knowledge_retriever
    import src.retriever.milvus_store as milvus_store

    langchain_available = True
except ImportError:
    pass

if not langchain_available:
    pytest.skip("langchain_core not installed", allow_module_level=True)


class FakeEmbeddingModel:
    def embed_query(self, query: str) -> list[float]:
        assert query
        return [0.1, 0.2, 0.3]


class TestRetrieveKnowledge:
    def test_returns_matching_snippets_with_source(self, monkeypatch):
        monkeypatch.setattr(milvus_store, "count_collection", lambda collection: 2)
        monkeypatch.setattr(
            knowledge_retriever,
            "create_embedding_model",
            lambda: FakeEmbeddingModel(),
        )

        captured = {}

        def fake_search_documents(collection_name, query_embedding, top_k):
            captured["collection_name"] = collection_name
            captured["query_embedding"] = query_embedding
            captured["top_k"] = top_k
            return [
                (
                    Document(
                        page_content="宽脚用户应优先选择宽楦或鞋面延展性好的鞋。",
                        metadata={"source_filename": "03_size_fit_and_foot_type.md"},
                        id="03_size_fit_and_foot_type_0",
                    ),
                    0.82,
                ),
                (
                    Document(
                        page_content="通勤皮鞋应兼顾支撑、透气和防滑。",
                        metadata={"source_filename": "08_leather_formal_commute_guide.md"},
                        id="08_leather_formal_commute_guide_0",
                    ),
                    0.65,
                ),
            ]

        monkeypatch.setattr(milvus_store, "search_documents", fake_search_documents)

        result = knowledge_retriever.retrieve_knowledge(
            "宽脚怎么选鞋",
            top_k=2,
            similarity_threshold=0.3,
        )

        assert captured == {
            "collection_name": "knowledge_base_collection",
            "query_embedding": [0.1, 0.2, 0.3],
            "top_k": 2,
        }
        assert "03_size_fit_and_foot_type.md" in result
        assert "宽脚用户应优先选择宽楦" in result
        assert "08_leather_formal_commute_guide.md" in result
        assert "通勤皮鞋应兼顾支撑" in result

    def test_filters_snippets_below_similarity_threshold(self, monkeypatch):
        monkeypatch.setattr(milvus_store, "count_collection", lambda collection: 1)
        monkeypatch.setattr(
            knowledge_retriever,
            "create_embedding_model",
            lambda: FakeEmbeddingModel(),
        )
        monkeypatch.setattr(
            milvus_store,
            "search_documents",
            lambda collection_name, query_embedding, top_k: [
                (
                    Document(
                        page_content="这条分数太低，不应进入知识上下文。",
                        metadata={"source_filename": "low_score.md"},
                    ),
                    0.29,
                )
            ],
        )

        result = knowledge_retriever.retrieve_knowledge(
            "宽脚怎么选鞋",
            similarity_threshold=0.3,
        )

        assert result == ""

    def test_empty_collection_skips_embedding_and_search(self, monkeypatch):
        monkeypatch.setattr(milvus_store, "count_collection", lambda collection: 0)

        def fail_create_embedding_model():
            raise AssertionError("embedding model should not be initialized")

        def fail_search_documents(collection_name, query_embedding, top_k):
            raise AssertionError("search should not be called")

        monkeypatch.setattr(
            knowledge_retriever,
            "create_embedding_model",
            fail_create_embedding_model,
        )
        monkeypatch.setattr(milvus_store, "search_documents", fail_search_documents)

        assert knowledge_retriever.retrieve_knowledge("任意问题") == ""
