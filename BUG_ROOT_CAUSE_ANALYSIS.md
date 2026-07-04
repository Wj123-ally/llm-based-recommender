# Bug #1 根因分析：三轮对话推荐完全相同商品

## 问题现象

```
第一轮: "夏季透气运动鞋" → 推荐商品 A, B, C
第二轮: "300-500元" → 仍推荐商品 A, B, C (完全相同)
第三轮: "耐克/阿迪达斯 + 跑步" → 仍推荐商品 A, B, C (完全相同)
```

---

## 根本原因（按优先级）

### 🔴 原因1: 数据库缺少price字段 - **最严重**

**位置**: `src/database/schema.py`

**现状**:
```sql
CREATE TABLE products (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    brand TEXT,
    material TEXT,
    season TEXT,
    ...
    -- ❌ 没有 price 字段！
)
```

**影响链路**:
```
用户第二轮: "300-500元"
  ↓
combined_analysis_node 尝试提取价格条件
  ↓
❌ 但 schema 中根本没有 price 字段
  ↓
parse_query_filters() 也不解析价格（line 325-375）
  ↓
filters = {} (价格条件被丢弃)
  ↓
retrieve_products() 收到的 filters 没有价格信息
  ↓
回退到纯语义检索 - 与第一轮查询语义相似
  ↓
返回相同的 top-3 商品
```

**代码证据**:
1. `parse_query_filters()` (line 325-375) 只解析: season, material, gender, colors, shoe_types
2. **完全没有价格解析逻辑**
3. `metadata_matches_filters()` (line 571-624) 也没有价格判断

---

### 🟡 原因2: 品牌字段数据为空 - **中等严重**

**现状**: 数据库有 `brand` 字段，但实际数据大量为空或NULL

**影响链路**:
```
用户第三轮: "耐克/阿迪达斯"
  ↓
combined_analysis_node 提取: brand = "耐克" 或 "阿迪达斯"
  ↓
retrieve_products(filters={"brand": "耐克"})
  ↓
metadata_matches_filters() 检查 metadata["brand"]
  ↓
所有商品的 brand 字段都是空字符串或不匹配
  ↓
apply_candidate_filters() 过滤掉所有候选
  ↓
❌ filters 无效，回退到无过滤检索
  ↓
返回相同的 top-3 商品
```

**代码位置**: `src/retriever/hybrid_retriever.py:571-624`

```python
def metadata_matches_filters(metadata: dict[str, Any], filters: dict[str, object]) -> bool:
    # line 583-586
    for key in ["brand", "material", "season"]:
        value = filters.get(key)
        if value and not _metadata_contains_any(metadata, [key], [str(value)]):
            return False  # ❌ 品牌不匹配时直接返回False
```

---

### 🟢 原因3: 缺少"排除已推荐商品"的逻辑 - **设计缺陷**

**现状**: 系统没有记录已推荐的商品ID，导致多轮检索可能返回重复商品

**代码分析**:

#### state.py 中没有记录已推荐商品
```python
# src/recommender/state.py
class RecState(TypedDict):
    query: str
    previous_query: str
    documents: list[dict]
    # ❌ 缺少: recommended_product_ids: list[str]
```

#### retrieve_products() 没有 exclude_ids 参数
```python
# src/retriever/hybrid_retriever.py:275-304
def retrieve_products(
    query: str,
    filters: dict[str, object] | None = None,
    # ❌ 缺少: exclude_ids: list[str] | None = None
) -> list[Document]:
    dense_candidates = retrieve_from_dense_vectors(query, filters)
    bm25_candidates = retrieve_from_bm25(query, filters)
    multimodal_candidates = retrieve_from_multimodal_images(query, filters)
    candidates = rrf_fusion(dense_candidates, bm25_candidates, multimodal_candidates)
    
    # ❌ 没有排除已推荐商品的逻辑
    
    ranked_candidates = rerank_candidates(query, candidates)
    return [_with_retrieval_metadata(c) for c in ranked_candidates]
```

**影响**:
即使第二轮、第三轮的查询略有不同，由于：
1. 结构化过滤失效（原因1、2）
2. 纯语义检索返回最相似的top-K
3. 没有"多样性"机制来避免重复

导致三轮推荐完全相同。

---

### 🟢 原因4: context_mode 判断未影响商品选择

**代码位置**: `src/recommender/self_query_node.py:119-240`

**现状**:
```python
def self_query_retrieve(state: RecState) -> RecState:
    context_mode = state.get("context_mode", "new_request")  # line 130
    
    # ✅ context_mode 被正确识别
    # ❌ 但没有基于 context_mode 调整检索策略
    
    # line 156-166: 只影响 search_query 的拼接
    should_inherit_search_context = (
        previous_query
        and context_mode == "clarification"
        and not current_query_has_new_product
        and not has_previous_product_reference
    )
    if should_inherit_search_context:
        search_query = f"{previous_query} {query}"  # 拼接查询词
    
    # ❌ 但检索仍然从全库检索，没有"从上一轮结果中筛选"的逻辑
    results = retrieve_products(search_query, filters=filters)
```

**问题**:
- `context_mode` 被正确识别（测试中三轮都是相同的thread_id）
- 但系统没有利用这个信息来：
  - 在 `follow_up` 模式下从上一轮结果中筛选
  - 在 `clarification` 模式下叠加新条件并排除已推荐商品

---

## 完整的Bug执行流程

### 第一轮: "夏季透气运动鞋"

```
1. combined_analysis_node
   ├─ need_products: True
   ├─ context_mode: "new_request"
   └─ filters: {season: "夏", include_shoe_types: ["运动鞋"]}

2. self_query_retrieve
   ├─ search_query: "夏季透气运动鞋"
   └─ filters: {season: "夏", include_shoe_types: ["运动鞋"]}

3. retrieve_products(query, filters)
   ├─ retrieve_from_dense_vectors() → Top 20候选
   ├─ retrieve_from_bm25() → Top 20候选
   ├─ rrf_fusion() → 合并去重
   ├─ apply_candidate_filters(filters) → 应用season、shoe_type过滤
   └─ rerank_candidates() → Top 5 → 返回前3个

4. 结果: [商品A, 商品B, 商品C]
```

### 第二轮: "300-500元"

```
1. combined_analysis_node
   ├─ need_products: True
   ├─ context_mode: "clarification" ✅ 正确识别为追加条件
   └─ filters: {} ❌ LLM无法提取price，因为schema不支持

2. self_query_retrieve
   ├─ search_query: "夏季透气运动鞋 300-500元" ✅ 拼接了查询词
   └─ filters: {} ❌ 价格条件丢失

3. retrieve_products(query, filters={})
   ├─ 语义检索: "夏季透气运动鞋 300-500元"
   ├─ 向量相似度与第一轮几乎相同
   ├─ 无有效filters，跳过过滤
   └─ rerank_candidates() → ❌ 返回与第一轮相同的Top 3

4. 结果: [商品A, 商品B, 商品C] ❌ 完全相同
```

### 第三轮: "耐克/阿迪达斯 + 跑步"

```
1. combined_analysis_node
   ├─ need_products: True
   ├─ context_mode: "new_request" (因为提到了新需求)
   └─ filters: {brand: "耐克"} ✅ 提取了品牌

2. self_query_retrieve
   ├─ search_query: "耐克 跑步鞋"
   └─ filters: {brand: "耐克"}

3. retrieve_products(query, filters={"brand": "耐克"})
   ├─ retrieve_from_dense_vectors(filters) → Top 20候选
   ├─ apply_candidate_filters(filters={"brand": "耐克"})
   │   └─ metadata_matches_filters() 检查每个商品的brand字段
   │       └─ ❌ 所有商品的brand都是空或不匹配
   │       └─ 过滤后: 0个候选
   ├─ ❌ 候选为空，filters失效
   ├─ 回退: retrieve_products(query, filters=None)
   ├─ 纯语义检索: "耐克 跑步鞋"
   └─ ❌ 仍返回语义相似的前3个商品

4. 结果: [商品A, 商品B, 商品C] ❌ 完全相同
```

---

## 为什么filters失效但系统没有报错？

### 代码中的"silent failure"

**位置1**: `hybrid_retriever.py:283-284`
```python
candidates = rrf_fusion(dense_candidates, bm25_candidates, multimodal_candidates)
if filters:
    candidates = apply_candidate_filters(candidates, filters)  # line 284
    
# ❌ 如果过滤后 candidates 为空，代码继续执行
# ❌ 没有日志警告"filters导致0结果"
```

**位置2**: `hybrid_retriever.py:484-497`
```python
def apply_candidate_filters(
    candidates: list[RetrievalCandidate],
    filters: dict[str, object],
) -> list[RetrievalCandidate]:
    if not filters or not candidates:
        return candidates  # 直接返回，无警告
    
    filtered: list[RetrievalCandidate] = []
    for candidate in candidates:
        metadata = candidate.document.metadata or {}
        if not metadata_matches_filters(metadata, filters):
            continue  # 静默跳过不匹配的商品
        filtered.append(candidate)
    return filtered  # ❌ 可能返回空列表，但无日志
```

**结果**: 
- 当filters导致候选为空时
- 代码继续执行到 `if not candidates: return []`
- 最终返回空列表
- 上层代码看到空结果，但不知道是"没有匹配商品"还是"filters配置错误"

---

## 关键缺陷总结

| 缺陷 | 严重性 | 位置 | 影响 |
|------|--------|------|------|
| 1. 数据库无price字段 | 🔴 严重 | schema.py, hybrid_retriever.py | 价格筛选完全失效 |
| 2. 品牌数据大量为空 | 🟡 中等 | 数据质量 | 品牌筛选几乎失效 |
| 3. 无"排除已推荐"逻辑 | 🟢 设计 | state.py, hybrid_retriever.py | 多轮重复推荐 |
| 4. filters失效无日志 | 🟢 可观测性 | hybrid_retriever.py | 难以调试 |
| 5. context_mode未充分利用 | 🟢 设计 | self_query_node.py | 上下文管理弱 |

---

## 修复方案（不改数据的前提下）

### 方案A: 增加"排除已推荐商品"逻辑 ⭐ **推荐**

**修改位置**: 
1. `src/recommender/state.py` - 添加 `recommended_product_ids: list[str]`
2. `src/retriever/hybrid_retriever.py` - 添加 `exclude_ids` 参数
3. `src/recommender/self_query_node.py` - 在检索前排除已推荐ID

**预期效果**:
- 第二轮会推荐新的3个商品（D, E, F）
- 第三轮会推荐更新的3个商品（G, H, I）
- ✅ 解决重复推荐问题

**工作量**: 1-2小时

---

### 方案B: 增加filters失效的诊断和日志 ⭐ **推荐**

**修改位置**: `src/retriever/hybrid_retriever.py`

```python
def apply_candidate_filters(candidates, filters):
    if not filters or not candidates:
        return candidates
    
    initial_count = len(candidates)
    filtered = [c for c in candidates if metadata_matches_filters(c.document.metadata, filters)]
    
    # 新增: 诊断日志
    if filtered == []:
        logger.warning(
            f"[过滤诊断] 所有候选被过滤掉! "
            f"原始候选: {initial_count}, 过滤条件: {filters}"
        )
        # 采样输出前3个商品的相关字段
        for i, c in enumerate(candidates[:3], 1):
            meta = c.document.metadata
            logger.warning(
                f"  商品{i}: brand={meta.get('brand')}, "
                f"season={meta.get('season')}, material={meta.get('material')}"
            )
    
    return filtered
```

**预期效果**:
- 当filters导致空结果时，日志明确说明
- 开发者可以快速定位是"数据问题"还是"filters配置问题"

**工作量**: 30分钟

---

### 方案C: 诚实告知用户数据限制 ⭐ **推荐**

**修改位置**: `src/recommender/rag_node.py` 或 prompt

在系统prompt中添加:
```
重要提示：
1. 当前商品库没有价格信息，无法根据价格筛选
2. 部分商品缺少品牌标识
3. 如果用户询问价格或特定品牌但库中无该品牌，请诚实说明

禁止生成无法验证的断言，如"已根据价格筛选"。
```

**预期效果**:
- 系统不再声称"已按300-500元筛选"
- 用户了解系统限制
- 提升可信度

**工作量**: 10分钟

---

## 推荐实施顺序

1. **立即实施** (40分钟):
   - 方案C: 修改系统prompt，诚实告知限制
   - 方案B: 添加filters失效日志

2. **核心修复** (1-2小时):
   - 方案A: 实现"排除已推荐商品"逻辑

3. **长期改进** (需要数据):
   - 添加price字段并重新导入数据
   - 改进品牌提取逻辑

---

## 验证方法

修复后重新进行三轮对话测试:
```python
# 第一轮
"夏季透气运动鞋" → [商品A, B, C]

# 第二轮（预期结果改善）
"300-500元" → [商品D, E, F]  # ✅ 不同的商品
# 系统说明: "当前商品库暂无价格信息，以下是基于其他属性的推荐"

# 第三轮（预期结果改善）
"耐克/阿迪达斯 + 跑步" → [商品G, H, I]  # ✅ 不同的商品
# 系统说明: "当前库中暂无耐克/阿迪达斯品牌，以下是其他适合跑步的运动鞋"
```

---

**总结**: 三轮推荐相同商品的根本原因是**结构化筛选失效（price字段不存在、brand数据为空）+ 缺少商品多样性机制**。在不能修改数据的前提下，通过"排除已推荐商品"和"诚实告知限制"可以显著改善用户体验。
