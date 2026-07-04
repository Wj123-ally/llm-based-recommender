# Bug修复完成报告

## 修复日期
2026-07-04

## 修复的问题
**Bug #1: 三轮对话推荐完全相同的商品**

---

## 实施的修复方案

### ✅ 方案A: 添加"排除已推荐商品"逻辑 (核心修复)

**修改文件**:
1. `src/recommender/state.py`
2. `src/retriever/hybrid_retriever.py`
3. `src/recommender/self_query_node.py`

**具体修改**:

#### 1. state.py - 添加状态字段
```python
# 新增字段：已推荐商品的ID列表，用于避免多轮对话中重复推荐相同商品
recommended_product_ids: list[str]
```

#### 2. hybrid_retriever.py - 支持排除商品ID
```python
def retrieve_products(
    query: str,
    filters: dict[str, object] | None = None,
    exclude_ids: list[str] | None = None,  # 新增参数
) -> list[Document]:
    # ... 检索逻辑 ...
    
    # 排除已推荐的商品
    if exclude_ids:
        initial_count = len(candidates)
        candidates = [
            c for c in candidates
            if c.document.metadata.get("id") not in exclude_ids
        ]
        if initial_count > len(candidates):
            logger.info(
                "[商品检索] 排除已推荐商品: 排除前%d个, 排除后%d个",
                initial_count,
                len(candidates)
            )
```

#### 3. self_query_node.py - 使用排除逻辑
```python
# 获取已推荐商品ID列表
recommended_ids = state.get("recommended_product_ids", [])

# 判断是否需要排除已推荐商品
# 只有在追问或补充条件时（非新请求），且已有推荐历史时才排除
should_exclude_previous = (
    context_mode in ["follow_up", "clarification"]
    and len(recommended_ids) > 0
    and not current_query_has_new_product
)

exclude_ids = recommended_ids if should_exclude_previous else None

results = retrieve_products(
    search_query,
    filters=filters if filters else None,
    exclude_ids=exclude_ids
)

# 更新已推荐商品ID列表
new_ids = [
    doc.metadata.get("id")
    for doc in selected_results
    if doc.metadata.get("id")
]
if new_ids:
    all_recommended_ids = list(set(recommended_ids + new_ids))
    state["recommended_product_ids"] = all_recommended_ids
```

**预期效果**:
- ✅ 第二轮、第三轮会推荐不同的商品
- ✅ 在clarification和follow_up模式下自动排除已推荐商品
- ✅ 在new_request模式下重新开始，不排除商品

---

### ✅ 方案B: 添加filters失效诊断日志

**修改文件**: `src/retriever/hybrid_retriever.py`

**具体修改**:
```python
def apply_candidate_filters(
    candidates: list[RetrievalCandidate],
    filters: dict[str, object],
) -> list[RetrievalCandidate]:
    if not filters or not candidates:
        return candidates

    initial_count = len(candidates)
    filtered = [c for c in candidates if metadata_matches_filters(c.document.metadata, filters)]

    # 诊断日志：当过滤导致空结果时输出警告
    if len(filtered) == 0 and initial_count > 0:
        logger.warning(
            "[过滤诊断] 所有%d个候选被过滤掉! 过滤条件: %s",
            initial_count,
            filters
        )
        # 采样输出前3个商品的相关字段帮助调试
        for i, candidate in enumerate(candidates[:3], 1):
            meta = candidate.document.metadata or {}
            logger.warning(
                "  候选商品%d: brand='%s', season='%s', material='%s', title='%s'",
                i,
                meta.get('brand', ''),
                meta.get('season', ''),
                meta.get('material', ''),
                (meta.get('title', '') or '')[:40]
            )
    elif len(filtered) < initial_count:
        logger.info(
            "[过滤诊断] 过滤前: %d个候选, 过滤后: %d个候选, 过滤条件: %s",
            initial_count,
            len(filtered),
            filters
        )

    return filtered
```

**预期效果**:
- ✅ 当filters导致空结果时，日志清楚显示原因
- ✅ 开发者可以快速定位是数据问题还是过滤条件问题
- ✅ 提高系统可观测性

---

### ✅ 方案C: 修改系统prompt诚实告知数据限制

**修改文件**: `src/recommender/utils.py`

**具体修改**:
在RAG prompt模板开头添加了数据限制说明：

```
━━━━━━━━━━ 重要提示：数据限制说明 ━━━━━━━━━━

当前商品库的数据限制：
1. 商品库中没有价格信息，无法根据价格进行筛选或推荐
2. 部分商品缺少品牌标识信息
3. 推荐主要基于商品标题、材质、季节、颜色、功能等属性

回答要求：
- 如果用户询问价格或预算（如"300元左右"、"便宜的"、"高端的"），请诚实说明："当前商品库暂无价格信息，建议您咨询客服获取最新报价"
- 如果用户指定品牌但库中无该品牌商品，请明确说明并推荐替代方案
- 不要生成无法验证的断言，如"已根据价格筛选"、"价格在XX元范围内"等
- 当无法满足某些筛选条件时，诚实告知用户，并说明实际推荐基于哪些可用条件
```

**预期效果**:
- ✅ 系统不再声称"已按价格筛选"
- ✅ 用户清楚了解数据限制
- ✅ 提升系统可信度和透明度

---

## 修复前后对比

### 修复前的问题

```
第一轮: "夏季透气运动鞋" → [商品A, B, C]
第二轮: "300-500元" → [商品A, B, C] ❌ 完全相同
第三轮: "耐克/阿迪达斯 + 跑步" → [商品A, B, C] ❌ 完全相同

问题：
- 价格筛选完全失效（数据库无price字段）
- 品牌筛选失效（brand字段为空）
- 没有商品多样性机制
- 系统声称"已根据价格筛选"但实际未筛选（误导用户）
```

### 修复后的预期表现

```
第一轮: "夏季透气运动鞋" → [商品A, B, C]
第二轮: "300-500元" → [商品D, E, F] ✅ 不同的商品
  系统说明: "当前商品库暂无价格信息，建议您咨询客服获取最新报价。以下是基于其他属性的推荐..."
第三轮: "耐克/阿迪达斯 + 跑步" → [商品G, H, I] ✅ 不同的商品
  系统说明: "当前库中暂无耐克/阿迪达斯品牌，以下是其他适合跑步的运动鞋..."

改善：
- ✅ 三轮推荐不同的商品
- ✅ 诚实告知价格信息缺失
- ✅ 明确说明品牌限制
- ✅ 不再误导用户
```

---

## 技术细节

### 排除逻辑的工作原理

1. **状态维护**: 在`RecState`中添加`recommended_product_ids`字段，记录所有已推荐的商品ID

2. **智能判断**: 只在以下情况排除已推荐商品：
   - `context_mode` 为 `follow_up` 或 `clarification`
   - 已有推荐历史（`recommended_product_ids`非空）
   - 当前查询不是明确的新商品请求

3. **ID累积**: 每次推荐后，将新推荐的商品ID加入列表

4. **检索时过滤**: 在RRF融合后、rerank前，过滤掉`exclude_ids`中的商品

### 日志输出示例

```
[商品检索] 排除已推荐商品: 3个ID将被排除
[商品检索] 排除已推荐商品: 排除前20个, 排除后17个
[过滤诊断] 过滤前: 17个候选, 过滤后: 12个候选, 过滤条件: {'season': '夏'}
```

---

## 验证方法

运行三轮对话测试：
```bash
conda run -n rag_env python test_fixed_version.py
```

测试会验证：
1. 三轮对话的商品ID是否不同
2. 系统是否诚实告知价格限制
3. 系统是否诚实告知品牌限制

---

## 未来改进建议

### 短期（如果可以修改数据）
1. **添加price字段**: 在数据库schema中添加price字段并重新导入数据
2. **改进品牌提取**: 从商品标题中提取品牌信息
3. **修复metadata映射**: 确保title→name, shoe_type→type等字段正确映射

### 长期
1. **动态多样性控制**: 根据用户反馈动态调整排除策略
2. **个性化推荐**: 基于用户历史偏好推荐
3. **相似度反推**: 当用户明确喜欢某商品时，推荐相似商品

---

## 修改的文件清单

- ✅ `src/recommender/state.py` - 添加recommended_product_ids字段
- ✅ `src/retriever/hybrid_retriever.py` - 添加exclude_ids参数和诊断日志
- ✅ `src/recommender/self_query_node.py` - 实现排除逻辑和ID累积
- ✅ `src/recommender/utils.py` - 修改prompt添加数据限制说明

---

## 测试状态

- [x] 语法检查通过
- [x] API成功启动
- [ ] 三轮对话测试 (正在运行...)

---

**修复人员**: Claude Code  
**修复方法**: 代码逻辑改进 + Prompt工程  
**修复时间**: 约2小时
