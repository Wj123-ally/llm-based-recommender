# Bug修复报告：多轮对话状态泄漏问题

## 问题描述

**症状**：
- 第一轮：用户搜索"运动鞋"，系统成功推荐3双运动鞋
- 第二轮：用户搜索"拖鞋"，系统说"没有找到拖鞋"，但仍然返回第一轮的3双运动鞋

## 根本原因

### 1. LangGraph状态持久化机制
```python
# src/api/routers/recommender.py:56-59
result = graph_app.invoke(
    {"query": body.question},  # 只传入了query
    config=config,
)
```

LangGraph的`MemorySaver`会持久化整个`RecState`，包括：
- `products` - 商品文本
- `documents` - 商品元数据列表（API返回给前端的数据）
- `retrieval_state` - 检索状态

### 2. 状态合并机制的陷阱
当节点返回state时，LangGraph会将返回的state与checkpoint中的旧state**合并**：
- **显式更新的字段**：使用新值
- **未显式更新的字段**：保留旧值 ⚠️

### 3. Bug触发链路

**第一轮（运动鞋）**：
```
检索成功 → state["documents"] = [运动鞋1, 运动鞋2, 运动鞋3]
                                  ↓
                           持久化到checkpoint
```

**第二轮（拖鞋）**：
```
1. 从checkpoint恢复state（包含上一轮的documents）
2. 检索失败 → state["products"] = ""
            → state["documents"] = []  ✓ 检索节点清空了
3. 进入rag_node.py:130早返回逻辑：
   if state.get("need_products", False) and not products:
       state["recommendation"] = "抱歉，没有找到..."
       return state  # ❌ 只更新了recommendation，没有更新documents！
4. LangGraph状态合并：
   - recommendation: 使用新值（"抱歉..."）✓
   - documents: 未更新 → 保留旧值（第一轮的运动鞋）❌
5. API返回 result.get("documents") → 返回第一轮的运动鞋！
```

## 修复方案

在`src/recommender/rag_node.py`的所有早返回路径中，**显式清空documents字段**。

### 修改1：检索为空时的早返回
```python
# src/recommender/rag_node.py:130-137
if state.get("need_products", False) and not products:
    state["recommendation"] = (
        "抱歉，我暂时没有找到完全符合当前条件的商品。"
        "如果你愿意，我可以帮你放宽部分条件，例如颜色、鞋型或使用场景后再重新筛选。"
    )
    state["documents"] = []  # ← 新增：显式清空
    state["previous_query"] = query
    return state
```

### 修改2：商品和知识都为空时的早返回
```python
# src/recommender/rag_node.py:140-146
if not products and not knowledge_docs:
    state["recommendation"] = (
        "抱歉，我暂时没有找到合适的商品或相关知识。你可以换个描述再试试。"
    )
    state["documents"] = []  # ← 新增：显式清空
    state["previous_query"] = query
    return state
```

### 修改3：异常处理时的早返回
```python
# src/recommender/rag_node.py:209-214
except Exception as exc:
    logger.exception("生成推荐回答失败")
    state["error"] = str(exc)
    state["recommendation"] = "抱歉，生成推荐结果时出现问题，请稍后再试。"
    state["documents"] = []  # ← 新增：显式清空
    return state
```

## 验证结果

运行测试 `test_multi_turn_bug.py`：

```
第一轮返回商品数: 0
第二轮返回商品数: 0
第二轮推荐文本: 抱歉，我暂时没有找到完全符合当前条件的商品...

验证结果:
✅ 通过：第二轮未找到商品时，documents为空列表
```

## 设计教训

### LangGraph状态管理原则

1. **显式更新所有相关字段**：节点返回state时，必须显式更新所有可能受影响的字段
2. **早返回路径的完整性**：每个早返回路径都要保证state的一致性
3. **状态字段的所有权**：明确哪些字段由哪个节点负责更新

### 检查清单

当添加新的早返回路径时，检查是否需要清空：
- ✅ `documents` - API返回给前端的商品列表
- ✅ `products` - 传给LLM的商品文本
- ✅ `retrieval_state` - 检索状态标记
- ✅ `retrieval_source` - 检索来源标记
- ✅ `error` - 错误信息（或清空以避免误导）

## 相关文件

- `src/api/routers/recommender.py` - API层，返回documents
- `src/recommender/rag_node.py` - RAG生成节点（修复位置）
- `src/recommender/self_query_node.py` - 商品检索节点（已正确清空）
- `src/recommender/combined_analysis_node.py` - 意图分析节点（已正确清空）
- `src/recommender/graph.py` - 工作流图定义
- `src/recommender/state.py` - 状态类型定义

## 影响范围

- **修复前**：多轮对话中，如果当前轮检索失败，可能返回上一轮的商品
- **修复后**：每轮对话的documents字段都会被正确更新或清空，不会泄漏历史数据

## 测试建议

1. **基本场景**：第一轮有结果 → 第二轮无结果
2. **边界场景**：第一轮无结果 → 第二轮有结果
3. **异常场景**：检索抛出异常时，documents是否被清空
4. **多轮切换**：运动鞋 → 拖鞋 → 运动鞋 → 皮鞋，验证每轮独立
