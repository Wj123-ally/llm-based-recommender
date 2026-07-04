#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断为什么"拖鞋"请求返回"运动鞋"结果"""

import sys
sys.path.insert(0, 'D:\\github_project\\简历项目\\llm-based-recommender')

# 测试检索逻辑
from src.retriever.hybrid_retriever import retrieve_products

print("="*80)
print("测试检索逻辑：为什么'拖鞋'返回'运动鞋'")
print("="*80)

# 测试1：直接检索"拖鞋"
print("\n[测试1] 直接检索'拖鞋'")
results1 = retrieve_products("拖鞋", filters=None)
print(f"  结果数: {len(results1)}")
if results1:
    for i, doc in enumerate(results1[:3], 1):
        title = doc.metadata.get('title', 'N/A')[:50]
        shoe_type = doc.metadata.get('shoe_type', 'N/A')
        print(f"  {i}. {title} (类型: {shoe_type})")
else:
    print("  ❌ 没有结果")

# 测试2：检索"有拖鞋吗，女鞋"
print("\n[测试2] 检索'有拖鞋吗，女鞋'")
results2 = retrieve_products("有拖鞋吗，女鞋", filters=None)
print(f"  结果数: {len(results2)}")
if results2:
    for i, doc in enumerate(results2[:3], 1):
        title = doc.metadata.get('title', 'N/A')[:50]
        shoe_type = doc.metadata.get('shoe_type', 'N/A')
        print(f"  {i}. {title} (类型: {shoe_type})")
else:
    print("  ❌ 没有结果")

# 测试3：带gender过滤的拖鞋
print("\n[测试3] 检索'拖鞋' + gender=女")
results3 = retrieve_products("拖鞋", filters={"gender": "女"})
print(f"  结果数: {len(results3)}")
if results3:
    for i, doc in enumerate(results3[:3], 1):
        title = doc.metadata.get('title', 'N/A')[:50]
        shoe_type = doc.metadata.get('shoe_type', 'N/A')
        gender = doc.metadata.get('gender', 'N/A')
        print(f"  {i}. {title} (类型: {shoe_type}, 性别: {gender})")
else:
    print("  ❌ 没有结果")

# 测试4：数据库中有多少拖鞋
print("\n[测试4] 数据库中拖鞋数量")
from src.database.product_repo import ProductRepo
repo = ProductRepo()
import sqlite3
conn = repo.conn
cursor = conn.execute("SELECT COUNT(*) FROM products WHERE shoe_type LIKE '%拖鞋%' OR title LIKE '%拖鞋%'")
count = cursor.fetchone()[0]
print(f"  数据库中包含'拖鞋'的商品: {count}个")

if count > 0:
    print("\n  示例拖鞋商品:")
    cursor = conn.execute("SELECT title, shoe_type FROM products WHERE shoe_type LIKE '%拖鞋%' OR title LIKE '%拖鞋%' LIMIT 5")
    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"    {i}. {row[0][:50]} (类型: {row[1]})")

print("\n" + "="*80)
