#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证修复效果的三轮对话测试"""
import requests
import json

API_URL = "http://localhost:8000/recommend/"

def test_fixed_version():
    """测试修复后的版本"""
    session = requests.Session()

    print("="*80)
    print("修复后的三轮对话测试")
    print("="*80)

    # 第一轮
    print("\n【第一轮】夏季透气运动鞋")
    resp1 = session.post(API_URL, json={"question": "我想买一双适合夏天穿的运动鞋，要透气舒适的"}, timeout=60)
    data1 = resp1.json()
    docs1 = data1.get('documents', [])
    ids1 = [d.get('metadata', {}).get('id') for d in docs1 if d.get('metadata', {}).get('id')]
    print(f"Thread ID: {data1.get('thread_id')}")
    print(f"推荐商品数: {len(docs1)}")
    print(f"商品IDs: {ids1}")
    print(f"回答摘要: {data1.get('answer', '')[:200]}...")

    # 第二轮
    print("\n【第二轮】价格要求")
    resp2 = session.post(API_URL, json={"question": "有没有价格在300-500元之间的？"}, timeout=60)
    data2 = resp2.json()
    docs2 = data2.get('documents', [])
    ids2 = [d.get('metadata', {}).get('id') for d in docs2 if d.get('metadata', {}).get('id')]
    print(f"Thread ID: {data2.get('thread_id')}")
    print(f"推荐商品数: {len(docs2)}")
    print(f"商品IDs: {ids2}")
    print(f"回答摘要: {data2.get('answer', '')[:200]}...")

    # 检查是否有不同的商品
    overlap = set(ids1) & set(ids2)
    print(f"\n与第一轮重复的商品: {len(overlap)}/{len(ids2)}")

    # 第三轮
    print("\n【第三轮】品牌和场景要求")
    resp3 = session.post(API_URL, json={"question": "这些鞋子中有耐克或者阿迪达斯的吗？推荐一款最适合跑步的"}, timeout=90)
    data3 = resp3.json()
    docs3 = data3.get('documents', [])
    ids3 = [d.get('metadata', {}).get('id') for d in docs3 if d.get('metadata', {}).get('id')]
    print(f"Thread ID: {data3.get('thread_id')}")
    print(f"推荐商品数: {len(docs3)}")
    print(f"商品IDs: {ids3}")
    print(f"回答摘要: {data3.get('answer', '')[:300]}...")

    # 检查是否有不同的商品
    overlap_all = set(ids1) & set(ids2) & set(ids3)
    print(f"\n三轮都相同的商品: {len(overlap_all)}/{len(ids1)}")

    # 评估
    print("\n" + "="*80)
    print("修复效果评估")
    print("="*80)

    if len(overlap_all) == 0:
        print("✅ 优秀: 三轮推荐的商品完全不同")
    elif len(overlap_all) < len(ids1):
        print(f"✅ 良好: 三轮推荐有{len(ids1) - len(overlap_all)}个不同的商品")
    else:
        print(f"❌ 仍有问题: 三轮推荐了完全相同的商品")

    # 检查系统是否诚实告知限制
    answer2 = data2.get('answer', '')
    answer3 = data3.get('answer', '')

    if '价格' in answer2 and ('暂无' in answer2 or '咨询' in answer2):
        print("✅ 第二轮诚实告知了价格限制")
    elif '已根据' in answer2 and '价格' in answer2:
        print("⚠️ 第二轮仍声称根据价格筛选（可能误导）")

    if '耐克' in answer3 or '阿迪达斯' in answer3:
        if '暂无' in answer3 or '没有' in answer3:
            print("✅ 第三轮诚实告知了品牌缺失")
        else:
            print("⚠️ 第三轮提到品牌但未说明限制")

if __name__ == "__main__":
    test_fixed_version()
