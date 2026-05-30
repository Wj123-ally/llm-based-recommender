"""
测试 RecState 类型定义。

纯逻辑测试，无外部依赖。
"""

import pytest

from src.recommender.state import RecState


class TestRecState:
    """RecState TypedDict 行为测试。"""

    def test_minimal_state_with_query_only(self):
        """最简状态 — 只有 query 字段即可创建。"""
        state: RecState = {"query": "推荐一件羽绒服"}
        assert state["query"] == "推荐一件羽绒服"
        # total=False 意味着未设置的字段不应该出现在 .keys() 里
        assert "on_topic" not in state

    def test_full_state_all_fields_accessible(self):
        """完整状态 — 所有字段均可读。"""
        state: RecState = {
            "query": "测试问题",
            "on_topic": "Yes",
            "need_products": True,
            "need_knowledge": False,
            "intent_analysis": "纯推荐",
            "recommendation": "推荐商品A",
            "products": "商品1\n商品2",
            "knowledge_docs": "",
            "documents": [{"metadata": {"title": "test"}}],
            "retrieval_state": "success",
            "retrieval_source": "hybrid",
            "knowledge_retrieval_state": "skipped",
            "error": "",
        }
        assert state["query"] == "测试问题"
        assert state["on_topic"] == "Yes"
        assert state["need_products"] is True
        assert state["need_knowledge"] is False
        assert state["knowledge_retrieval_state"] == "skipped"

    def test_retrieval_state_literals_accepted(self):
        """retrieval_state 应接受 'success'、'empty'、'skipped'。"""
        for value in ("success", "empty", "skipped"):
            state: RecState = {
                "query": "test",
                "retrieval_state": value,
            }
            assert state["retrieval_state"] == value

    def test_knowledge_retrieval_state_literals_accepted(self):
        """knowledge_retrieval_state 应接受 'success'、'empty'、'skipped'。"""
        for value in ("success", "empty", "skipped"):
            state: RecState = {
                "query": "test",
                "knowledge_retrieval_state": value,
            }
            assert state["knowledge_retrieval_state"] == value

    def test_documents_defaults_to_list(self):
        """documents 字段应为 dict 列表。"""
        state: RecState = {
            "query": "test",
            "documents": [],
        }
        assert state["documents"] == []

    def test_get_method_works_on_optional_fields(self):
        """未设置字段 .get() 应返回 None。"""
        state: RecState = {"query": "test"}
        assert state.get("on_topic") is None
        assert state.get("need_products") is None
        assert state.get("products", "") == ""
