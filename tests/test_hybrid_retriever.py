"""
测试 hybrid_retriever.py 的纯逻辑函数。

测试 rrf_fusion 等不依赖外部 API 的函数。
"""

import pytest

langchain_available = False
try:
    from langchain_core.documents import Document

    import src.retriever.hybrid_retriever as hybrid_retriever
    from src.retriever.hybrid_retriever import (
        RetrievalCandidate,
        _max_optional,
        merge_product_filters,
        parse_query_filters,
        retrieve_from_bm25,
        rrf_fusion,
    )

    langchain_available = True
except ImportError:
    pass

# 模块级标记：如果 langchain_core 未安装则跳过所有测试
if not langchain_available:
    pytest.skip("langchain_core not installed", allow_module_level=True)


class TestRrfFusion:
    """rrf_fusion RRF 融合去重测试。"""

    def _make_candidate(
        self,
        doc_id: str,
        content: str,
        sources: set[str],
        dense_score: float | None = None,
        bm25_score: float | None = None,
    ):
        return RetrievalCandidate(
            document=Document(
                page_content=content,
                metadata={"id": doc_id},
                id=doc_id,
            ),
            sources=sources,
            dense_score=dense_score,
            bm25_score=bm25_score,
        )

    def test_fusion_from_two_sources(self):
        """同一文档从两个来源被检索到，应合并为一条且 RRF 分数累加。"""
        dense_candidates = [
            self._make_candidate("1", "商品A", {"milvus_text"}, dense_score=0.8),
        ]
        bm25_candidates = [
            self._make_candidate("1", "商品A", {"bm25"}, bm25_score=0.6),
        ]

        merged = rrf_fusion(dense_candidates, bm25_candidates)
        assert len(merged) == 1
        assert merged[0].sources == {"milvus_text", "bm25"}
        assert merged[0].dense_score == 0.8
        assert merged[0].bm25_score == 0.6
        # 文档在两路中都是 rank=1，RRF = 1/(60+1) + 1/(60+1) = 2/61
        assert merged[0].rrf_score == pytest.approx(2 / 61)

    def test_fusion_disjoint_sets(self):
        """无重叠文档时保持各自独立，按 RRF 分数降序排列。"""
        dense = [
            self._make_candidate("1", "商品A", {"milvus_text"}, dense_score=0.8),
        ]
        bm25 = [
            self._make_candidate("2", "商品B", {"bm25"}, bm25_score=0.7),
        ]

        merged = rrf_fusion(dense, bm25)
        assert len(merged) == 2
        # 两个文档 RRF 分数相同（都是单路 rank=1），排序应稳定
        for c in merged:
            assert c.rrf_score is not None

    def test_fusion_three_sets_with_overlap(self):
        """三个来源的部分重叠应正确处理 RRF 累加。"""
        set1 = [self._make_candidate("1", "A", {"milvus_text"}, dense_score=0.9)]
        set2 = [
            self._make_candidate("1", "A", {"bm25"}, bm25_score=0.5),
            self._make_candidate("2", "B", {"bm25"}, bm25_score=0.8),
        ]
        set3 = [self._make_candidate("2", "B", {"milvus_text"}, dense_score=0.6)]

        merged = rrf_fusion(set1, set2, set3)
        assert len(merged) == 2

        # 找 id=1：出现在 set1 (rank=1) 和 set2 (rank=1)
        doc1 = next(c for c in merged if c.document.id == "1")
        assert doc1.sources == {"milvus_text", "bm25"}
        assert doc1.dense_score == 0.9
        assert doc1.bm25_score == 0.5
        # RRF: set1 rank=1 → 1/61, set2 rank=1 → 1/61, sum = 2/61
        assert doc1.rrf_score == pytest.approx(2 / 61)

        # 找 id=2：出现在 set2 (rank=2) 和 set3 (rank=1)
        doc2 = next(c for c in merged if c.document.id == "2")
        assert doc2.sources == {"bm25", "milvus_text"}
        assert doc2.bm25_score == 0.8
        assert doc2.dense_score == 0.6
        # RRF: set2 rank=2 → 1/62, set3 rank=1 → 1/61, sum = 1/61 + 1/62
        expected_rrf = 1 / 61 + 1 / 62
        assert doc2.rrf_score == pytest.approx(expected_rrf)

    def test_empty_groups(self):
        """空组不应导致错误。"""
        merged = rrf_fusion([], [])
        assert merged == []

    def test_fusion_keeps_higher_score(self):
        """融合时保留两个来源中更高的原始分数。"""
        dense = [
            self._make_candidate("1", "A", {"milvus_text"}, dense_score=0.9),
        ]
        bm25 = [
            self._make_candidate("1", "A", {"bm25"}, dense_score=0.3),
        ]

        merged = rrf_fusion(dense, bm25)
        assert len(merged) == 1
        assert merged[0].dense_score == 0.9  # 保留较高值

    def test_fusion_single_group(self):
        """单路检索也应正常返回 RRF 分数。"""
        candidates = [
            self._make_candidate("1", "A", {"milvus_text"}, dense_score=0.9),
            self._make_candidate("2", "B", {"milvus_text"}, dense_score=0.7),
            self._make_candidate("3", "C", {"milvus_text"}, dense_score=0.5),
        ]

        merged = rrf_fusion(candidates)
        assert len(merged) == 3
        # rank 1: 1/61, rank 2: 1/62, rank 3: 1/63
        assert merged[0].rrf_score == pytest.approx(1 / 61)  # 最高 RRF
        assert merged[1].rrf_score == pytest.approx(1 / 62)
        assert merged[2].rrf_score == pytest.approx(1 / 63)  # 最低 RRF

    def test_fusion_rrf_sorts_descending(self):
        """RRF 融合结果应按 RRF 分数降序排列。"""
        # doc1 出现在 Milvus rank=1, doc2 在 Milvus rank=2 且 BM25 rank=1
        # doc2 RRF 更高，应排在 doc1 前面
        dense = [
            self._make_candidate("1", "Top1", {"milvus_text"}, dense_score=0.9),
            self._make_candidate("2", "Top2", {"milvus_text"}, dense_score=0.8),
        ]
        bm25 = [
            self._make_candidate("2", "Top2", {"bm25"}, bm25_score=0.9),
        ]

        merged = rrf_fusion(dense, bm25)
        # doc2: RRF = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.0325
        # doc1: RRF = 1/(60+1) = 1/61 ≈ 0.0164
        # doc2 has higher RRF, should be first
        assert merged[0].document.id == "2"
        assert merged[1].document.id == "1"


class TestMaxOptional:
    """_max_optional 辅助函数测试。"""

    def test_both_defined(self):
        assert _max_optional(3.0, 5.0) == 5.0

    def test_left_none(self):
        assert _max_optional(None, 5.0) == 5.0

    def test_right_none(self):
        assert _max_optional(3.0, None) == 3.0

    def test_both_none(self):
        assert _max_optional(None, None) is None


class TestFilteredRetrieval:
    def test_merge_product_filters_prefers_rule_exclusions(self):
        merged = merge_product_filters(
            {"include_shoe_types": ["皮鞋", "运动鞋"]},
            {"exclude_shoe_types": ["皮鞋"]},
        )

        assert merged["include_shoe_types"] == ["运动鞋"]
        assert merged["exclude_shoe_types"] == ["皮鞋"]

    def test_merge_product_filters_drops_scene_as_shoe_type(self):
        merged = merge_product_filters(
            {"include_colors": ["黑色"], "include_shoe_types": ["通勤鞋"]},
            {},
        )

        assert merged == {"include_colors": ["黑色"]}

    def test_parse_excludes_shoe_type_with_negative_reason_after_value(self):
        filters = parse_query_filters("我觉得运动鞋比较好皮鞋太闷了并不适合上班穿")

        assert filters.get("include_shoe_types") == ["运动鞋"]
        assert filters.get("exclude_shoe_types") == ["皮鞋"]
        assert "material" not in filters

    def test_bm25_filters_before_scoring_selection(self, monkeypatch):
        documents = [
            Document(
                page_content="formal leather shoe",
                metadata={"id": "1", "shoe_type": "leather"},
                id="1",
            ),
            Document(
                page_content="comfortable sneaker",
                metadata={"id": "2", "shoe_type": "sneaker"},
                id="2",
            ),
        ]

        class FakeBm25:
            def get_scores(self, query_tokens):
                return [100.0, 1.0]

        monkeypatch.setattr(
            hybrid_retriever,
            "load_bm25_index",
            lambda: {"documents": documents, "bm25": FakeBm25(), "version": 2},
        )
        monkeypatch.setattr(
            hybrid_retriever,
            "chinese_tokenize",
            lambda query: ["shoe"],
        )
        monkeypatch.setattr(
            hybrid_retriever.settings,
            "BM25_RETRIEVER_TOP_K",
            10,
            raising=False,
        )

        candidates = retrieve_from_bm25(
            "shoe",
            filters={"include_shoe_types": ["sneaker"]},
        )

        assert [candidate.document.metadata["id"] for candidate in candidates] == ["2"]
