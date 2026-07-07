"""
测试多轮对话状态管理bug修复。

验证场景：
1. 第一轮搜索运动鞋，成功推荐3双
2. 第二轮搜索拖鞋（假设库存为空），应该返回空列表，而不是上一轮的运动鞋
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender.graph import create_recommender_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def test_multi_turn_empty_results():
    """测试第二轮检索为空时，不应返回第一轮的商品。"""
    graph = create_recommender_graph()
    thread_id = "test-multi-turn-bug"

    # 第一轮：搜索运动鞋
    print("\n" + "=" * 70)
    print("第一轮：推荐运动鞋")
    print("=" * 70)

    result1 = graph.invoke(
        {"query": "推荐运动鞋"},
        config={"configurable": {"thread_id": thread_id}},
    )

    documents1 = result1.get("documents", [])
    print(f"\n第一轮返回商品数: {len(documents1)}")
    if documents1:
        for i, doc in enumerate(documents1[:3], 1):
            meta = doc.get("metadata", {})
            print(f"  商品{i}: {meta.get('title', '-')}")

    # 第二轮：搜索拖鞋（假设库存为空或很少）
    print("\n" + "=" * 70)
    print("第二轮：推荐拖鞋")
    print("=" * 70)

    result2 = graph.invoke(
        {"query": "推荐拖鞋"},
        config={"configurable": {"thread_id": thread_id}},
    )

    documents2 = result2.get("documents", [])
    recommendation2 = result2.get("recommendation", "")

    print(f"\n第二轮返回商品数: {len(documents2)}")
    print(f"第二轮推荐文本: {recommendation2[:100]}...")

    if documents2:
        print("\n第二轮返回的商品:")
        for i, doc in enumerate(documents2[:3], 1):
            meta = doc.get("metadata", {})
            print(f"  商品{i}: {meta.get('title', '-')} | 鞋型: {meta.get('shoe_type', '-')}")

    # 验证：如果第二轮说没有找到商品，那么documents应该为空
    if "没有找到" in recommendation2 or "暂时没有" in recommendation2:
        print("\n" + "=" * 70)
        print("验证结果:")
        if len(documents2) == 0:
            print("✅ 通过：第二轮未找到商品时，documents为空列表")
        else:
            print(f"❌ 失败：第二轮说没找到商品，但documents返回了{len(documents2)}个商品")
            print("   这些商品可能是第一轮的残留数据！")
        print("=" * 70)
    else:
        print("\n第二轮成功找到商品，跳过bug验证")


if __name__ == "__main__":
    test_multi_turn_empty_results()
