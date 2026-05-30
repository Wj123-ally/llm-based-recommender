"""
测试 hybrid_retriever.py 的纯逻辑函数。

测试 merge_candidates、fallback 排序等不依赖外部 API 的函数。
"""

import pytest

langchain_available = False
try:
    from langchain_core.documents import Document

    from src.retriever.hybrid_retriever import (
        RetrievalCandidate,
        _fallback_rank,
        _fallback_score,
        _max_optional,
        merge_candidates,
    )

    langchain_available = True
except ImportError:
    pass

# 模块级标记：如果 langchain_core 未安装则跳过所有测试
if not langchain_available:
    pytest.skip("langchain_core not installed", allow_module_level=True)


class TestMergeCandidates:
    """merge_candidates 去重合并测试。"""

    def _make_candidate(
        self,
        doc_id: str,
        content: str,
        sources: set[str],
        chroma_score: float | None = None,
        bm25_score: float | None = None,
    ):
        return RetrievalCandidate(
            document=Document(
                page_content=content,
                metadata={"id": doc_id},
                id=doc_id,
            ),
            sources=sources,
            chroma_score=chroma_score,
            bm25_score=bm25_score,
        )

    def test_merge_from_two_sources(self):
        """同一文档从两个来源被检索到，应合并为一条。"""
        chroma_candidates = [
            self._make_candidate("1", "商品A", {"chroma"}, chroma_score=0.8),
        ]
        bm25_candidates = [
            self._make_candidate("1", "商品A", {"bm25"}, bm25_score=0.6),
        ]

        merged = merge_candidates(chroma_candidates, bm25_candidates)
        assert len(merged) == 1
        assert merged[0].sources == {"chroma", "bm25"}
        assert merged[0].chroma_score == 0.8
        assert merged[0].bm25_score == 0.6

    def test_merge_disjoint_sets(self):
        """无重叠文档时保持各自独立。"""
        chroma = [
            self._make_candidate("1", "商品A", {"chroma"}, chroma_score=0.8),
        ]
        bm25 = [
            self._make_candidate("2", "商品B", {"bm25"}, bm25_score=0.7),
        ]

        merged = merge_candidates(chroma, bm25)
        assert len(merged) == 2

    def test_merge_three_sets_with_overlap(self):
        """三个来源的部分重叠应正确处理。"""
        set1 = [self._make_candidate("1", "A", {"a"}, chroma_score=0.9)]
        set2 = [
            self._make_candidate("1", "A", {"b"}, bm25_score=0.5),
            self._make_candidate("2", "B", {"b"}, bm25_score=0.8),
        ]
        set3 = [self._make_candidate("2", "B", {"c"}, chroma_score=0.6)]

        merged = merge_candidates(set1, set2, set3)
        assert len(merged) == 2

        # 找 id=1
        doc1 = next(c for c in merged if c.document.id == "1")
        assert doc1.sources == {"a", "b"}
        assert doc1.chroma_score == 0.9
        assert doc1.bm25_score == 0.5

        # 找 id=2
        doc2 = next(c for c in merged if c.document.id == "2")
        assert doc2.sources == {"b", "c"}
        assert doc2.bm25_score == 0.8
        assert doc2.chroma_score == 0.6

    def test_empty_groups(self):
        """空组不应导致错误。"""
        merged = merge_candidates([], [])
        assert merged == []

    def test_merge_keeps_higher_score(self):
        """合并时保留两个来源中更高的分数。"""
        chroma = [
            self._make_candidate("1", "A", {"chroma"}, chroma_score=0.9),
        ]
        bm25 = [
            self._make_candidate("1", "A", {"bm25"}, chroma_score=0.3),
        ]

        merged = merge_candidates(chroma, bm25)
        assert len(merged) == 1
        assert merged[0].chroma_score == 0.9  # 保留较高值


class TestFallbackRank:
    """_fallback_rank 回退排序测试。"""

    def test_rank_by_score_sum(self):
        """回退排序应按分数之和降序排列。"""
        c1 = RetrievalCandidate(
            document=Document(page_content="C", id="1"),
            sources={"chroma"},
            chroma_score=0.9,
        )
        c2 = RetrievalCandidate(
            document=Document(page_content="A", id="2"),
            sources={"bm25"},
            bm25_score=0.3,
        )
        c3 = RetrievalCandidate(
            document=Document(page_content="B", id="3"),
            sources={"chroma", "bm25"},
            chroma_score=0.5,
            bm25_score=0.4,
        )

        ranked = _fallback_rank([c2, c3, c1])
        # c1=0.9, c3=0.9, c2=0.3
        assert ranked[0].document.id in ("1", "3")
        assert ranked[-1].document.id == "2"

    def test_empty_list(self):
        """空列表排序不应报错。"""
        result = _fallback_rank([])
        assert result == []


class TestFallbackScore:
    """_fallback_score 函数测试。"""

    def test_both_scores_present(self):
        c = RetrievalCandidate(
            document=Document(page_content="x", id="1"),
            sources=set(),
            chroma_score=0.8,
            bm25_score=0.2,
        )
        assert _fallback_score(c) == 1.0

    def test_only_chroma(self):
        c = RetrievalCandidate(
            document=Document(page_content="x", id="1"),
            sources=set(),
            chroma_score=0.7,
        )
        assert _fallback_score(c) == 0.7

    def test_only_bm25(self):
        c = RetrievalCandidate(
            document=Document(page_content="x", id="1"),
            sources=set(),
            bm25_score=0.5,
        )
        assert _fallback_score(c) == 0.5

    def test_no_scores(self):
        c = RetrievalCandidate(
            document=Document(page_content="x", id="1"),
            sources=set(),
        )
        assert _fallback_score(c) == 0.0


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
