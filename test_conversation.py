#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三轮对话测试脚本"""
import requests
import json
import time

API_URL = "http://localhost:8000/recommend/"

def print_result(round_num, question, response_data):
    """打印测试结果"""
    print(f"\n{'='*80}")
    print(f"第 {round_num} 轮对话")
    print(f"{'='*80}")
    print(f"问题: {question}")
    print(f"\nthread_id: {response_data.get('thread_id', 'N/A')}")
    print(f"\n回答:\n{response_data.get('answer', 'N/A')}")

    documents = response_data.get('documents', [])
    print(f"\n推荐商品数: {len(documents)}")

    if documents:
        print("\n推荐的商品列表:")
        for i, doc in enumerate(documents[:5], 1):
            metadata = doc.get('metadata', {})
            print(f"\n  商品 {i}:")
            print(f"    名称: {metadata.get('name', 'Unknown')}")
            print(f"    品牌: {metadata.get('brand', 'Unknown')}")
            print(f"    价格: ¥{metadata.get('price', 'Unknown')}")
            print(f"    类型: {metadata.get('type', 'Unknown')}")
            print(f"    季节: {metadata.get('season', 'Unknown')}")
            print(f"    材质: {metadata.get('material', 'Unknown')}")
    print(f"\n{'='*80}\n")

def test_three_rounds():
    """进行三轮上下文相关的对话测试"""
    session = requests.Session()

    # 第一轮：询问夏季运动鞋
    print("开始测试...")
    question1 = "我想买一双适合夏天穿的运动鞋，要透气舒适的"
    print(f"\n发送第1轮请求: {question1}")

    try:
        resp1 = session.post(
            API_URL,
            json={"question": question1},
            timeout=60
        )
        resp1.raise_for_status()
        data1 = resp1.json()
        print_result(1, question1, data1)

        # 等待一下再发送第二轮
        time.sleep(2)

        # 第二轮：基于第一轮的上下文，询问价格范围
        question2 = "有没有价格在300-500元之间的？"
        print(f"\n发送第2轮请求: {question2}")

        resp2 = session.post(
            API_URL,
            json={"question": question2},
            timeout=60
        )
        resp2.raise_for_status()
        data2 = resp2.json()
        print_result(2, question2, data2)

        # 等待一下再发送第三轮
        time.sleep(2)

        # 第三轮：进一步缩小选择，询问品牌
        question3 = "这些鞋子中有耐克或者阿迪达斯的吗？推荐一款最适合跑步的"
        print(f"\n发送第3轮请求: {question3}")

        resp3 = session.post(
            API_URL,
            json={"question": question3},
            timeout=60
        )
        resp3.raise_for_status()
        data3 = resp3.json()
        print_result(3, question3, data3)

        # 分析测试结果
        print("\n" + "="*80)
        print("测试结果分析")
        print("="*80)

        # 检查上下文是否保持
        thread_ids = [
            data1.get('thread_id'),
            data2.get('thread_id'),
            data3.get('thread_id')
        ]
        print(f"\n1. 上下文连续性检查:")
        print(f"   Thread IDs: {thread_ids}")
        if len(set(thread_ids)) == 1 and thread_ids[0]:
            print("   ✓ 三轮对话使用相同的thread_id，上下文已保持")
        else:
            print("   ✗ Thread ID不一致，上下文可能未正确保持")

        # 检查推荐商品的相关性
        print(f"\n2. 推荐相关性检查:")
        docs1 = data1.get('documents', [])
        docs2 = data2.get('documents', [])
        docs3 = data3.get('documents', [])

        if docs1:
            print(f"   第1轮: 推荐了{len(docs1)}个商品")
            seasons = [d.get('metadata', {}).get('season', '') for d in docs1]
            print(f"   - 季节分布: {set(seasons)}")

        if docs2:
            print(f"   第2轮: 推荐了{len(docs2)}个商品")
            prices = [d.get('metadata', {}).get('price', 0) for d in docs2]
            in_range = sum(1 for p in prices if 300 <= p <= 500)
            print(f"   - 价格在300-500元之间的商品数: {in_range}/{len(docs2)}")

        if docs3:
            print(f"   第3轮: 推荐了{len(docs3)}个商品")
            brands = [d.get('metadata', {}).get('brand', '') for d in docs3]
            print(f"   - 品牌分布: {set(brands)}")

        print(f"\n3. 话术一致性检查:")
        print(f"   - 第1轮回答长度: {len(data1.get('answer', ''))}")
        print(f"   - 第2轮回答长度: {len(data2.get('answer', ''))}")
        print(f"   - 第3轮回答长度: {len(data3.get('answer', ''))}")

        if all([data1.get('answer'), data2.get('answer'), data3.get('answer')]):
            print("   ✓ 所有轮次都生成了回答")
        else:
            print("   ✗ 部分轮次未生成回答")

        print("\n测试完成!")

    except requests.exceptions.RequestException as e:
        print(f"\n✗ 请求失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    test_three_rounds()
