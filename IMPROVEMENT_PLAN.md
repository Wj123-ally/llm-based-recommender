# 在数据质量受限情况下的改进方案

## 当前约束条件
- ❌ 数据库没有 `price` 字段，无法添加
- ❌ 品牌数据大量缺失，无法补充
- ✅ 只能通过代码逻辑改进用户体验

---

## 改进方案

### 方案1: 让系统诚实地告知数据限制 🎯 **推荐优先实施**

**目标**: 避免系统生成无法兑现的承诺

**实施位置**: `src/recommender/combined_analysis_node.py` 或 `src/recommender/rag_node.py`

**改进内容**:

1. **在系统prompt中明确说明数据限制**
   ```python
   # 在RAG生成prompt中添加
   SYSTEM_CONSTRAINTS = """
   重要提示：当前商品数据的限制：
   1. 商品库中没有价格信息，无法根据价格筛选
   2. 大部分商品缺少品牌标识
   3. 推荐主要基于商品标题、材质、季节、功能等属性
   
   回答要求：
   - 如果用户询问价格，诚实说明"当前商品库暂无价格信息，建议您咨询客服获取最新价格"
   - 如果用户指定品牌但库中无该品牌，明确说明并推荐替代方案
   - 不要声称"已根据价格筛选"或类似无法验证的断言
   """
   ```

2. **修改话术生成逻辑**，让LLM知道哪些筛选条件实际生效了

**效果**:
- ✅ 系统不再声称"已按价格筛选"
- ✅ 用户了解数据限制，设定合理期望
- ✅ 提升系统可信度

---

### 方案2: 改进多轮对话的商品多样性 🎯 **解决重复推荐问题**

**目标**: 即使结构化筛选失效，也要在多轮对话中提供不同的商品选择

**问题根源**: 
- 当前检索逻辑在筛选条件无效时，总是返回向量最相似的top-3
- 三轮语义相似，导致推荐重复

**实施方案**:

#### 2.1 在state中记录已推荐商品ID

```python
# src/recommender/state.py
class RecState(TypedDict):
    query: str
    previous_query: str
    documents: list[dict]
    recommended_product_ids: list[str]  # 新增：已推荐的商品ID
    ...
```

#### 2.2 在检索时排除已推荐商品

```python
# src/retriever/hybrid_retriever.py

def retrieve_products(
    query: str,
    filters: dict[str, object] | None = None,
    exclude_ids: list[str] | None = None,  # 新增参数
) -> list[Document]:
    dense_candidates = retrieve_from_dense_vectors(query, filters)
    bm25_candidates = retrieve_from_bm25(query, filters)
    multimodal_candidates = retrieve_from_multimodal_images(query, filters)
    candidates = rrf_fusion(dense_candidates, bm25_candidates, multimodal_candidates)
    
    if filters:
        candidates = apply_candidate_filters(candidates, filters)
    
    # 新增：排除已推荐的商品
    if exclude_ids:
        candidates = [
            c for c in candidates 
            if c.document.metadata.get("id") not in exclude_ids
        ]
    
    if not candidates:
        return []
    
    # ... rerank逻辑
```

#### 2.3 在self_query_node中使用排除逻辑

```python
# src/recommender/self_query_node.py

def self_query_retrieve(state: RecState) -> RecState:
    query = state["query"]
    context_mode = state.get("context_mode", "new_request")
    recommended_ids = state.get("recommended_product_ids", [])
    
    # 如果是追问（不是明确的新商品请求），且无有效筛选条件
    # 则排除已推荐商品，展示新商品
    should_exclude = (
        context_mode == "follow_up" and 
        len(recommended_ids) > 0 and
        not has_new_product_request(query)
    )
    
    exclude_ids = recommended_ids if should_exclude else None
    
    docs = retrieve_products(
        query=query,
        filters=filters,
        exclude_ids=exclude_ids
    )
    
    # 更新已推荐商品列表
    new_ids = [doc.metadata.get("id") for doc in docs if doc.metadata.get("id")]
    state["recommended_product_ids"] = list(set(recommended_ids + new_ids))
    
    return state
```

**效果**:
- ✅ 第二轮、第三轮会推荐不同的商品（前提是库中有足够商品）
- ✅ 用户感知到系统在响应新需求
- ⚠️ 如果库中符合条件的商品很少，可能降低推荐质量

---

### 方案3: 智能回退策略 - 明确告知用户 🎯 **改善用户体验**

**目标**: 当筛选条件无法满足时，明确告诉用户

**实施位置**: `src/recommender/self_query_node.py`

```python
def self_query_retrieve(state: RecState) -> RecState:
    query = state["query"]
    filters = extract_filters(query)  # 提取筛选条件
    
    # 尝试带筛选条件检索
    docs = retrieve_products(query, filters)
    
    # 检查是否有price或brand筛选但无结果
    has_unsupported_filter = (
        filters.get("price_min") or 
        filters.get("price_max") or
        filters.get("brand")
    )
    
    if has_unsupported_filter and len(docs) == 0:
        # 放宽筛选条件，只保留可支持的筛选
        supported_filters = {
            k: v for k, v in filters.items()
            if k in ["season", "material", "shoe_type", "gender", "color"]
        }
        docs = retrieve_products(query, supported_filters)
        
        # 在state中标记筛选条件部分失效
        state["filter_status"] = "partial"
        state["unsupported_filters"] = {
            k: v for k, v in filters.items()
            if k not in supported_filters
        }
    
    state["documents"] = serialize_docs(docs)
    return state
```

然后在RAG prompt中告知LLM：

```python
# src/recommender/rag_node.py

def build_rag_prompt(state: RecState) -> str:
    filter_status = state.get("filter_status")
    unsupported = state.get("unsupported_filters", {})
    
    constraint_note = ""
    if filter_status == "partial" and unsupported:
        constraint_note = f"""
        注意：用户提到的以下筛选条件暂时无法支持：
        {format_unsupported_filters(unsupported)}
        
        请在回答中诚实说明，并基于其他可用条件进行推荐。
        """
    
    return f"""
    {constraint_note}
    
    用户问题：{state['query']}
    
    相关商品：
    {state['products']}
    
    请生成推荐回答...
    """
```

**效果**:
- ✅ 用户明确知道哪些筛选条件生效了
- ✅ 避免误导用户
- ✅ 系统显得更透明、可信

---

### 方案4: 改进metadata返回 🎯 **修复API响应**

**目标**: 确保返回给前端的商品信息完整

**实施位置**: `src/retriever/product_documents.py` 或 `src/recommender/self_query_node.py`

```python
# src/recommender/self_query_node.py

def serialize_docs(docs: list[Document]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        metadata = dict(doc.metadata or {})
        
        # 修复字段映射
        normalized_metadata = {
            "id": metadata.get("id"),
            "name": metadata.get("title") or metadata.get("name") or "未命名商品",
            "title": metadata.get("title"),
            "brand": metadata.get("brand") or "品牌未知",
            "price": None,  # 明确设为None而不是"Unknown"
            "type": metadata.get("shoe_type") or metadata.get("type"),
            "season": metadata.get("season"),
            "material": metadata.get("material"),
            "color": metadata.get("color"),
            "gender": metadata.get("gender"),
            "usage_scene": metadata.get("usage_scene"),
            "functionality": metadata.get("functionality"),
            "image_url": metadata.get("image_url"),
            "_source_rank": index,
        }
        
        serialized.append({
            "page_content": build_product_content(metadata) or doc.page_content,
            "metadata": normalized_metadata,
        })
    
    return serialized
```

**效果**:
- ✅ 前端UI能正确显示商品名称
- ✅ price为None而不是"Unknown"，前端可判断并显示"价格待询"
- ✅ 所有字段映射正确

---

## 实施优先级和工作量

| 方案 | 优先级 | 工作量 | 预期效果 |
|------|--------|--------|----------|
| 方案1: 诚实告知限制 | 🔴 最高 | 10分钟 | 避免误导用户 |
| 方案4: 修复metadata | 🔴 最高 | 20分钟 | 修复API响应 |
| 方案2: 商品多样性 | 🟡 中等 | 1小时 | 解决重复推荐 |
| 方案3: 智能回退 | 🟢 较低 | 1小时 | 提升透明度 |

---

## 推荐实施顺序

### 第一步：快速修复（30分钟）
1. 实施方案1 - 在系统prompt中说明数据限制
2. 实施方案4 - 修复metadata字段映射

### 第二步：核心改进（1-2小时）
3. 实施方案2 - 增加商品多样性，解决重复推荐问题

### 第三步：锦上添花（可选）
4. 实施方案3 - 智能回退策略

---

## 预期改进效果

**修复前**:
- ❌ 三轮推荐相同商品
- ❌ 系统声称"已按价格筛选"但实际未筛选
- ❌ 返回数据显示"Unknown"

**修复后**:
- ✅ 系统诚实说明"暂无价格信息"
- ✅ 多轮对话推荐不同商品（方案2）
- ✅ API返回完整的商品名称、类型等信息
- ✅ 用户体验更好，不被误导

---

需要我帮你实施这些改进吗？我建议先从方案1和方案4开始，这两个最快且影响最大。
