from types import SimpleNamespace

import src.recommender.combined_analysis_node as combined_analysis


class FakeStructuredLlm:
    def __init__(self, result):
        self.result = result

    def invoke(self, messages):
        return self.result


class FakeLlm:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema):
        return FakeStructuredLlm(self.result)


class TestCombinedAnalysisNode:
    def test_product_recommendation_always_enables_knowledge(self, monkeypatch):
        result = SimpleNamespace(
            on_topic="Yes",
            need_products=True,
            need_knowledge=False,
            gender=None,
            brand=None,
            material=None,
            season=None,
            include_colors=[],
            exclude_colors=[],
            include_shoe_types=[],
            exclude_shoe_types=[],
            analysis="用户需要商品推荐。",
        )
        monkeypatch.setattr(
            combined_analysis,
            "create_chat_llm",
            lambda temperature=0: FakeLlm(result),
        )

        state = combined_analysis.combined_analysis_node({"query": "推荐一双黑色女鞋"})

        assert state["need_products"] is True
        assert state["need_knowledge"] is True

    def test_pure_knowledge_query_still_enables_knowledge(self, monkeypatch):
        result = SimpleNamespace(
            on_topic="Yes",
            need_products=False,
            need_knowledge=True,
            gender=None,
            brand=None,
            material=None,
            season=None,
            include_colors=[],
            exclude_colors=[],
            include_shoe_types=[],
            exclude_shoe_types=[],
            analysis="用户需要知识解答。",
        )
        monkeypatch.setattr(
            combined_analysis,
            "create_chat_llm",
            lambda temperature=0: FakeLlm(result),
        )

        state = combined_analysis.combined_analysis_node({"query": "宽脚怎么选鞋"})

        assert state["need_products"] is False
        assert state["need_knowledge"] is True
