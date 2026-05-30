"""
测试推荐图节点的纯逻辑部分。

使用 mock 模拟 LLM 调用，测试节点的条件分支和状态更新逻辑。
"""

from unittest import mock

import pytest

from src.recommender.state import RecState


class TestRouteAfterTopicCheck:
    """route_after_topic_check 路由函数测试。"""

    def test_yes_routes_to_analyze_intent(self):
        from src.recommender.graph import route_after_topic_check

        state: RecState = {"query": "test", "on_topic": "Yes"}
        result = route_after_topic_check(state)
        assert result == "analyze_intent"

    def test_no_routes_to_end(self):
        from src.recommender.graph import route_after_topic_check

        state: RecState = {"query": "test", "on_topic": "No"}
        result = route_after_topic_check(state)
        assert result == "__end__"

    def test_missing_on_topic_routes_to_end(self):
        from src.recommender.graph import route_after_topic_check

        state: RecState = {"query": "test"}
        result = route_after_topic_check(state)
        assert result == "__end__"


class TestRouteAfterIntent:
    """route_after_intent 路由函数测试。"""

    def test_both_needs_products_first(self):
        """两者都需要时先走商品检索。"""
        from src.recommender.graph import route_after_intent

        state: RecState = {
            "query": "test",
            "need_products": True,
            "need_knowledge": True,
        }
        result = route_after_intent(state)
        assert result == "hybrid_retrieve"

    def test_only_products(self):
        from src.recommender.graph import route_after_intent

        state: RecState = {
            "query": "test",
            "need_products": True,
            "need_knowledge": False,
        }
        result = route_after_intent(state)
        assert result == "hybrid_retrieve"

    def test_only_knowledge(self):
        from src.recommender.graph import route_after_intent

        state: RecState = {
            "query": "test",
            "need_products": False,
            "need_knowledge": True,
        }
        result = route_after_intent(state)
        assert result == "knowledge_retrieve"

    def test_neither_goes_to_rag(self):
        from src.recommender.graph import route_after_intent

        state: RecState = {
            "query": "test",
            "need_products": False,
            "need_knowledge": False,
        }
        result = route_after_intent(state)
        assert result == "rag_recommender"

    def test_default_behavior_when_fields_missing(self):
        """字段缺失时默认需要商品推荐。"""
        from src.recommender.graph import route_after_intent

        state: RecState = {"query": "test"}
        result = route_after_intent(state)
        assert result == "hybrid_retrieve"


class TestRouteAfterHybridRetrieve:
    """route_after_hybrid_retrieve 路由函数测试。"""

    def test_need_knowledge_after_products(self):
        from src.recommender.graph import route_after_hybrid_retrieve

        state: RecState = {
            "query": "test",
            "need_knowledge": True,
        }
        result = route_after_hybrid_retrieve(state)
        assert result == "knowledge_retrieve"

    def test_no_knowledge_after_products(self):
        from src.recommender.graph import route_after_hybrid_retrieve

        state: RecState = {
            "query": "test",
            "need_knowledge": False,
        }
        result = route_after_hybrid_retrieve(state)
        assert result == "rag_recommender"


class TestCheckTopicNode:
    """check_topic_node Pydantic 模型测试。"""

    def test_topic_grade_only_accepts_yes_no(self):
        """TopicGrade 只接受 Yes/No。"""
        from src.recommender.check_topic_node import TopicGrade
        from pydantic import ValidationError

        # Yes 和 No 都可以
        TopicGrade(score="Yes")
        TopicGrade(score="No")

        # 其他值应失败
        with pytest.raises(ValidationError):
            TopicGrade(score="Maybe")


class TestAnalyzeIntentNode:
    """analyze_intent_node 测试（mock LLM）。"""

    def test_user_intent_model_both_true(self):
        """UserIntent 模型 both=true 的序列化。"""
        from src.recommender.analyze_intent_node import UserIntent

        intent = UserIntent(
            need_products=True,
            need_knowledge=True,
            analysis="混合需求：用户既要推荐也要洗护知识",
        )
        assert intent.need_products is True
        assert intent.need_knowledge is True
        assert "洗护" in intent.analysis

    def test_user_intent_model_knowledge_only(self):
        """UserIntent 模型仅知识查询。"""
        from src.recommender.analyze_intent_node import UserIntent

        intent = UserIntent(
            need_products=False,
            need_knowledge=True,
            analysis="纯知识查询：用户只问怎么洗护",
        )
        assert intent.need_products is False
        assert intent.need_knowledge is True

    def test_user_intent_node_with_failure_defaults(self):
        """意图分析失败时应默认 need_products=True。"""
        from src.recommender.analyze_intent_node import analyze_intent_node

        state: RecState = {"query": "test"}

        # Mock create_chat_llm 抛出异常模拟失败
        with mock.patch(
            "src.recommender.analyze_intent_node.create_chat_llm",
            side_effect=RuntimeError("API unavailable"),
        ):
            result = analyze_intent_node(state)

        # 回退：默认仅商品推荐
        assert result["need_products"] is True
        assert result["need_knowledge"] is False
        assert "意图分析失败" in result["intent_analysis"]


class TestSelfQueryRetrieve:
    """self_query_retrieve 条件跳过测试。"""

    def test_skip_when_no_products_needed(self):
        """need_products=False 时应跳过检索。"""
        from src.recommender.self_query_node import self_query_retrieve

        state: RecState = {
            "query": "怎么洗羊毛衫",
            "need_products": False,
        }
        result = self_query_retrieve(state)

        assert result["retrieval_state"] == "skipped"
        assert result["retrieval_source"] == "none"
        assert result["products"] == ""
        assert result["documents"] == []

    def test_skip_preserves_other_state(self):
        """跳过检索时不应影响其他字段。"""
        from src.recommender.self_query_node import self_query_retrieve

        state: RecState = {
            "query": "怎么洗羊毛衫",
            "need_products": False,
            "need_knowledge": True,
        }
        result = self_query_retrieve(state)

        assert result["retrieval_state"] == "skipped"
        # need_knowledge 原样保留，由后续节点处理
        assert result["need_knowledge"] is True


class TestRagNode:
    """rag_recommender 节点逻辑测试。"""

    def test_empty_products_returns_placeholder(self):
        """无商品时返回友好提示。"""
        from src.recommender.rag_node import rag_recommender

        state: RecState = {
            "query": "test",
            "products": "",
        }
        result = rag_recommender(state)

        assert "抱歉" in result["recommendation"] or "没有" in result["recommendation"]

    def test_knowledge_default_when_missing(self):
        """缺失 knowledge_docs 时不报错。"""
        from src.recommender.rag_node import rag_recommender

        state: RecState = {
            "query": "test",
            "products": "",
        }
        # 不应抛出异常
        result = rag_recommender(state)
        assert "recommendation" in result


class TestKnowledgeRetrieveNode:
    """knowledge_retrieve_node 测试。"""

    def test_import_error_graceful(self):
        """模块不可用时应优雅跳过。"""
        from src.recommender.graph import knowledge_retrieve_node

        state: RecState = {"query": "test"}

        with mock.patch.dict(
            "sys.modules",
            {"src.knowledge_base.knowledge_retriever": None},
        ):
            result = knowledge_retrieve_node(state)
            assert result["knowledge_docs"] == ""

    def test_exception_returns_empty(self):
        """知识检索异常时应返回空字符串。"""
        from src.recommender.graph import knowledge_retrieve_node

        state: RecState = {"query": "test"}

        with mock.patch(
            "src.knowledge_base.knowledge_retriever.retrieve_knowledge",
            side_effect=RuntimeError("search failed"),
        ):
            result = knowledge_retrieve_node(state)
            assert result["knowledge_docs"] == ""
