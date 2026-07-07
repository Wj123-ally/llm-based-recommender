#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests

session = requests.Session()
thread_id = 'd294af5d-bd55-49c9-b032-4f466902e48d'

question3 = '这些鞋子中有耐克或者阿迪达斯的吗？推荐一款最适合跑步的'
print(f'发送第3轮请求: {question3}')
print('等待响应...')

try:
    resp3 = session.post(
        'http://localhost:8000/recommend/',
        json={'question': question3},
        cookies={'thread_id': thread_id},
        timeout=90
    )
    resp3.raise_for_status()
    data3 = resp3.json()

    print(f'\nthread_id: {data3.get("thread_id")}')
    print(f'\n回答:\n{data3.get("answer", "N/A")[:800]}...')

    documents = data3.get('documents', [])
    print(f'\n推荐商品数: {len(documents)}')

    if documents:
        print('\n推荐的商品:')
        for i, doc in enumerate(documents[:3], 1):
            metadata = doc.get('metadata', {})
            print(f'  {i}. 品牌:{metadata.get("brand", "Unknown")} - 价格: ¥{metadata.get("price", "Unknown")} - 类型:{metadata.get("type", "Unknown")}')

except requests.exceptions.Timeout:
    print('请求超时')
except Exception as e:
    print(f'错误: {e}')
