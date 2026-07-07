"""
三轮对话上下文关联测试。

测试场景：
1. 第一轮：推荐运动鞋（建立初始上下文）
2. 第二轮：我要女款的（补充条件，应该继承第一轮的"运动鞋"需求）
3. 第三轮：换成白色（继续补充条件，应该继承"女款运动鞋"）

预期行为：
- 第二轮应该推荐"女款运动鞋"，而不是重新搜索所有女鞋
- 第三轮应该推荐"白色女款运动鞋"，而不是所有白鞋
- context_mode应该正确识别为 new_request → clarification → clarification
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


def print_separator(title: str):
    """打印分隔符"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_result(turn: int, result: dict):
    """打印单轮结果"""
    print(f"\n【第{turn}轮结果】")
    print(f"  context_mode: {result.get('context_mode', '-')}")
    print(f"  需要商品: {result.get('need_products', '-')}")
    print(f"  需要知识: {result.get('need_knowledge', '-')}")
    print(f"  检索状态: {result.get('retrieval_state', '-')}")
    print(f"  商品过滤条件: {result.get('product_filters', {})}")

    documents = result.get("documents", [])
    print(f"  返回商品数: {len(documents)}")

    if documents:
        print(f"\n  商品列表:")
        for i, doc in enumerate(documents[:5], 1):
            meta = doc.get("metadata", {})
            title = meta.get("title", "-")
            shoe_type = meta.get("shoe_type", "-")
            gender = meta.get("gender", "-")
            color = meta.get("color", "-")
            print(f"    商品{i}: {title[:40]}")
            print(f"           鞋型={shoe_type} | 性别={gender} | 颜色={color}")

    recommendation = result.get("recommendation", "")
    print(f"\n  推荐回答: {recommendation[:200]}")
    if len(recommendation) > 200:
        print(f"           ...（共{len(recommendation)}字）")


def test_three_turn_context():
    """测试三轮对话的上下文关联"""
    graph = create_recommender_graph()
    thread_id = "test-three-turn-context"

    print_separator("三轮对话上下文关联测试")

    # ============================================================
    # 第一轮：推荐运动鞋（建立初始上下文）
    # ============================================================
    print_separator("第一轮：推荐运动鞋")
    query1 = "推荐运动鞋"
    print(f"用户输入: {query1}")

    result1 = graph.invoke(
        {"query": query1},
        config={"configurable": {"thread_id": thread_id}},
    )
    print_result(1, result1)

    # 验证第一轮
    assert result1.get("context_mode") == "new_request", \
        f"第一轮应该是new_request，实际是{result1.get('context_mode')}"

    # ============================================================
    # 第二轮：我要女款的（补充条件）
    # ============================================================
    print_separator("第二轮：我要女款的")
    query2 = "我要女款的"
    print(f"用户输入: {query2}")
    print(f"上一轮query: {query1}")

    result2 = graph.invoke(
        {"query": query2},
        config={"configurable": {"thread_id": thread_id}},
    )
    print_result(2, result2)

    # 验证第二轮
    context_mode2 = result2.get("context_mode")
    print(f"\n【验证点1】第二轮context_mode: {context_mode2}")
    if context_mode2 == "clarification":
        print("  ✅ 正确：识别为补充条件（clarification）")
    elif context_mode2 == "new_request":
        print("  ⚠️  警告：识别为新请求（new_request），可能丢失上下文")
    else:
        print(f"  ❌ 错误：识别为{context_mode2}")

    # 检查是否正确继承了"运动鞋"这个上下文
    filters2 = result2.get("product_filters", {})
    print(f"\n【验证点2】第二轮过滤条件: {filters2}")
    has_shoe_type = "include_shoe_types" in filters2 and "运动鞋" in filters2.get("include_shoe_types", [])
    has_gender = filters2.get("gender") == "女"

    if has_shoe_type and has_gender:
        print("  ✅ 正确：同时包含'运动鞋'和'女'条件")
    elif has_gender and not has_shoe_type:
        print("  ⚠️  警告：只有'女'条件，缺少'运动鞋'上下文")
    else:
        print(f"  ❌ 错误：过滤条件不符合预期")

    # ============================================================
    # 第三轮：换成白色（继续补充条件）
    # ============================================================
    print_separator("第三轮：换成白色")
    query3 = "换成白色"
    print(f"用户输入: {query3}")
    print(f"上一轮query: {query2}")

    result3 = graph.invoke(
        {"query": query3},
        config={"configurable": {"thread_id": thread_id}},
    )
    print_result(3, result3)

    # 验证第三轮
    context_mode3 = result3.get("context_mode")
    print(f"\n【验证点3】第三轮context_mode: {context_mode3}")
    if context_mode3 == "clarification":
        print("  ✅ 正确：识别为补充条件（clarification）")
    elif context_mode3 == "new_request":
        print("  ⚠️  警告：识别为新请求（new_request），可能丢失上下文")
    else:
        print(f"  ❌ 错误：识别为{context_mode3}")

    # 检查是否正确继承了"女款运动鞋"并添加了"白色"
    filters3 = result3.get("product_filters", {})
    print(f"\n【验证点4】第三轮过滤条件: {filters3}")
    has_shoe_type3 = "include_shoe_types" in filters3 and "运动鞋" in filters3.get("include_shoe_types", [])
    has_gender3 = filters3.get("gender") == "女"
    has_color3 = "白" in str(filters3.get("include_colors", []))

    if has_shoe_type3 and has_gender3 and has_color3:
        print("  ✅ 正确：同时包含'运动鞋'+'女'+'白色'条件")
    elif has_color3 and not (has_shoe_type3 and has_gender3):
        print("  ⚠️  警告：只有'白色'条件，缺少前两轮的上下文")
    else:
        print(f"  ❌ 错误：过滤条件不符合预期")

    # ============================================================
    # 第四轮：换成拖鞋（新请求，应该清空之前的上下文）
    # ============================================================
    print_separator("第四轮：换成拖鞋（新请求验证）")
    query4 = "换成拖鞋"
    print(f"用户输入: {query4}")
    print(f"上一轮query: {query3}")

    result4 = graph.invoke(
        {"query": query4},
        config={"configurable": {"thread_id": thread_id}},
    )
    print_result(4, result4)

    # 验证第四轮：应该是new_request，因为"换成拖鞋"明确了新的鞋型
    context_mode4 = result4.get("context_mode")
    print(f"\n【验证点5】第四轮context_mode: {context_mode4}")
    if context_mode4 == "new_request":
        print("  ✅ 正确：识别为新请求（new_request）")
    else:
        print(f"  ⚠️  警告：应该是new_request，实际是{context_mode4}")

    filters4 = result4.get("product_filters", {})
    print(f"\n【验证点6】第四轮过滤条件: {filters4}")
    has_slippers = "拖鞋" in str(filters4.get("include_shoe_types", []))
    has_old_context = "运动鞋" in str(filters4) or filters4.get("gender") == "女"

    if has_slippers and not has_old_context:
        print("  ✅ 正确：只有'拖鞋'条件，没有继承旧上下文")
    elif has_slippers and has_old_context:
        print("  ❌ 错误：继承了不应该继承的旧上下文")
    else:
        print(f"  ⚠️  警告：过滤条件不符合预期")

    # 验证documents是否被正确清空或更新
    documents4 = result4.get("documents", [])
    print(f"\n【验证点7】第四轮documents数量: {len(documents4)}")
    if documents4:
        # 检查是否真的是拖鞋
        shoe_types4 = [doc.get("metadata", {}).get("shoe_type", "") for doc in documents4]
        print(f"  返回的鞋型: {shoe_types4}")
        if all("拖鞋" in st for st in shoe_types4 if st):
            print("  ✅ 正确：返回的都是拖鞋")
        else:
            print("  ❌ 错误：返回了非拖鞋商品")
    else:
        print("  ℹ️  没有找到拖鞋商品")

    # ============================================================
    # 总结报告
    # ============================================================
    print_separator("测试总结")

    test_results = [
        ("第一轮context_mode", result1.get("context_mode") == "new_request"),
        ("第二轮context_mode", context_mode2 == "clarification"),
        ("第二轮继承运动鞋", has_shoe_type and has_gender),
        ("第三轮context_mode", context_mode3 == "clarification"),
        ("第三轮继承完整上下文", has_shoe_type3 and has_gender3 and has_color3),
        ("第四轮context_mode", context_mode4 == "new_request"),
        ("第四轮清空旧上下文", has_slippers and not has_old_context),
    ]

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    print(f"\n通过率: {passed}/{total} ({passed*100//total}%)")
    print("\n详细结果:")
    for name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {name}")

    if passed == total:
        print("\n🎉 所有测试通过！上下文关联功能正常。")
    else:
        print(f"\n⚠️  有 {total - passed} 项测试失败，需要修复。")

    return passed, total


if __name__ == "__main__":
    try:
        passed, total = test_three_turn_context()
        sys.exit(0 if passed == total else 1)
    except Exception as e:
        logger.exception("测试执行失败")
        sys.exit(1)
