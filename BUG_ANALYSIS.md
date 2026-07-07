# Bug分析报告

## 测试中发现的重大Bug

### Bug #1: 三轮对话推荐相同的3个商品 ⚠️ **重大问题**

**现象**: 
- 第一轮：推荐商品A、B、C
- 第二轮（增加价格限制300-500元）：仍然推荐商品A、B、C
- 第三轮（要求耐克/阿迪达斯品牌，适合跑步）：仍然推荐商品A、B、C

**根本原因分析**:

1. **数据库缺少价格字段**
   ```sql
   -- 当前schema中没有price字段
   CREATE TABLE products (
       id TEXT PRIMARY KEY,
       title TEXT NOT NULL,
       brand TEXT,
       material TEXT,
       season TEXT,
       ...
       -- 缺少 price 字段！
   )
   ```

2. **价格过滤失效**
   - 用户第二轮要求"300-500元"
   - 系统在 `self_query_node.py` 中尝试提取价格条件
   - 但由于数据库表中根本没有 `price` 字段，SQL过滤条件无法生效
   - 结果：价格过滤被忽略，仍返回第一轮的相同商品

3. **品牌过滤也失效**
   - 用户第三轮要求"耐克或阿迪达斯"
   - 虽然 `brand` 字段存在，但：
     - 数据库中的商品品牌字段可能为空（从测试看到返回的brand都是空字符串）
     - 或者数据集本身就不包含耐克/阿迪达斯等大品牌商品

4. **检索逻辑的问题**
   - 当结构化过滤条件（price、brand）无法匹配时
   - 系统回退到仅基于语义向量检索
   - 而语义上"夏季透气运动鞋"、"夏季透气运动鞋+价格"、"夏季透气运动鞋+品牌+跑步"的向量相似度差别不大
   - 导致三轮都检索到相同的top-3商品

**代码位置**:
- 数据库schema定义: `src/database/schema.py`
- 商品插入逻辑: `src/database/seed_jsonl_data.py` 
- 过滤逻辑: `src/recommender/self_query_node.py`, `src/retriever/hybrid_retriever.py`

---

### Bug #2: 数据库缺少价格字段 ⚠️ **数据完整性问题**

**现象**: 
- 数据库 `products` 表中没有 `price` 字段
- API返回的商品metadata中 `price` 显示为 `"Unknown"`
- 系统无法进行任何价格相关的筛选

**根本原因**:
1. **Schema定义不完整**
   - `src/database/schema.py` 中创建表时未包含price字段
   - 可能在某次重构中被遗漏

2. **数据导入未处理价格**
   - `src/database/seed_jsonl_data.py` 在导入JSONL数据时
   - 即使源数据有价格，也没有字段映射到数据库

3. **检索代码假设price存在**
   - `src/recommender/self_query_node.py` 和其他代码尝试使用price进行过滤
   - 但实际表中不存在该字段

**影响范围**:
- ❌ 所有价格相关的筛选功能完全失效
- ❌ 用户询问"300元左右"、"便宜的"、"高端的"等价格相关需求无法满足
- ❌ 排序功能（按价格高低）无法实现
- ⚠️ 系统会生成关于价格的话术，但实际上没有基于价格进行筛选（误导用户）

---

### Bug #3: 品牌字段数据缺失 ⚠️ **数据质量问题**

**现象**:
- API返回的商品metadata中 `brand` 字段为空字符串
- 第三轮测试中，系统正确说明"没有耐克或阿迪达斯"，但实际上是因为品牌字段都是空的

**根本原因**:
1. **源数据可能缺少品牌信息**
   - JSONL源文件中的商品可能大多没有品牌字段
   - 或者品牌提取逻辑有问题

2. **品牌提取/清洗逻辑不完善**
   - 商品标题中可能包含品牌信息，但未提取
   - 例如: "细细条 拖鞋女夏..." - "细细条"可能是品牌，但未识别

**影响范围**:
- ❌ 品牌筛选功能基本失效（除非极少数商品有品牌）
- ⚠️ 用户询问特定品牌时，系统会说"没有"，但实际可能是数据缺失

---

### Bug #4: 关键商品属性返回"Unknown" ⚠️ **API响应问题**

**现象**:
```json
{
  "metadata": {
    "name": "Unknown",
    "price": "Unknown",
    "brand": "",
    "type": "Unknown"
  }
}
```

**根本原因**:
1. **商品文档构建逻辑问题**
   - `src/retriever/product_documents.py` 中的 `build_product_content()` 函数
   - 可能未正确将数据库字段映射到metadata

2. **字段名称不匹配**
   - API返回期望 `name` 字段，但数据库使用 `title` 字段
   - API返回期望 `type` 字段，但数据库使用 `shoe_type` 字段
   - 这些映射可能缺失或错误

**代码位置**:
- `src/retriever/product_documents.py`
- `src/recommender/self_query_node.py` 中的 `serialize_docs()` 函数

---

### Bug #5: 系统话术与实际筛选不一致 ⚠️ **用户体验问题**

**现象**:
- 第二轮系统说："已根据...300–500元...严格筛选"
- 但实际上由于price字段不存在，没有进行任何价格筛选

**根本原因**:
- LLM基于用户需求生成话术，假设系统已正确筛选
- 但实际检索层面的筛选失败了
- 缺少验证机制来确保话术与实际筛选结果一致

**影响**:
- ❌ 严重误导用户
- ❌ 降低系统可信度
- ❌ 用户可能基于错误信息做决策

---

## 为什么三轮推荐相同商品？流程分析

```
第一轮: "夏季透气运动鞋"
  ↓
  语义向量检索 → Top 20候选
  ↓
  RRF融合 → Top 5
  ↓
  Rerank → Top 3: [商品A, B, C]
  ✅ 返回: A, B, C

第二轮: "300-500元"
  ↓
  意图识别: 需要商品 + 价格筛选条件 {price: [300, 500]}
  ↓
  尝试SQL过滤: WHERE price >= 300 AND price <= 500
  ❌ 失败: price字段不存在
  ↓
  回退到语义检索: "夏季透气运动鞋 300-500元"
  ↓
  语义上与第一轮几乎相同 → 相同的Top 20候选
  ↓
  RRF融合 → Top 5（与第一轮相同）
  ↓
  Rerank → Top 3: [商品A, B, C]
  ⚠️ 返回: A, B, C（相同）

第三轮: "耐克/阿迪达斯 + 跑步"
  ↓
  意图识别: 需要商品 + 品牌筛选 {brand: ["耐克", "阿迪达斯"]} + 场景 {usage_scene: "跑步"}
  ↓
  尝试SQL过滤: WHERE brand IN ('耐克', '阿迪达斯')
  ❌ 失败: 所有商品的brand字段都是空或无匹配
  ↓
  回退到语义检索: "夏季透气运动鞋 跑步"
  ↓
  语义上仍与第一轮高度相似 → 相同或高度重叠的Top 20候选
  ↓
  RRF融合 → Top 5（高度重叠）
  ↓
  Rerank → Top 3: [商品A, B, C]
  ⚠️ 返回: A, B, C（相同）
```

**关键问题**:
1. **结构化过滤完全失效** - price不存在，brand为空
2. **纯语义检索的局限性** - 三轮查询的语义向量差异不大
3. **缺少回退机制** - 当筛选无结果时，应该明确告知用户，而不是忽略筛选条件

---

## 修复建议

### 优先级1: 添加价格字段

1. **修改schema添加price字段**
   ```sql
   ALTER TABLE products ADD COLUMN price REAL;
   CREATE INDEX idx_products_price ON products(price);
   ```

2. **修改数据导入逻辑**
   - 在 `src/database/seed_jsonl_data.py` 中添加价格字段映射
   - 从JSONL的 `price` 或 `original_price` 字段读取

3. **重新导入数据**
   ```bash
   conda run -n rag_env python -m src.database.seed_jsonl_data
   conda run -n rag_env python -m src.indexing.embedding
   ```

### 优先级2: 修复metadata映射

1. **修改 `src/retriever/product_documents.py`**
   ```python
   def build_metadata(row: dict) -> dict:
       return {
           "name": row.get("title"),  # 映射title到name
           "price": row.get("price"),
           "brand": row.get("brand") or "未知品牌",
           "type": row.get("shoe_type"),  # 映射shoe_type到type
           "season": row.get("season"),
           "material": row.get("material"),
           ...
       }
   ```

2. **确保所有检索路径都使用正确的字段名**

### 优先级3: 增强品牌数据

1. **从商品标题提取品牌**
   - 添加品牌识别逻辑
   - 使用正则或字典匹配常见品牌

2. **补充品牌数据**
   - 检查源JSONL是否有品牌信息
   - 考虑是否需要外部API或爬虫补充

### 优先级4: 改进筛选逻辑

1. **添加筛选结果验证**
   ```python
   if filter_applied and len(results) == 0:
       # 明确告知用户没有符合条件的商品
       # 而不是回退到无筛选的结果
   ```

2. **区分"无结果"和"忽略筛选"**
   - 当price字段不存在时，明确log警告
   - 在响应中说明哪些筛选条件生效了

### 优先级5: 改进话术生成

1. **验证筛选条件是否实际生效**
2. **在prompt中告知LLM哪些筛选条件生效了**
3. **避免生成无法验证的断言**（如"已严格筛选价格"）

---

## 数据完整性检查清单

建议执行以下检查：

```bash
# 1. 检查数据库有多少商品
sqlite3 src/database/enriched_products.db "SELECT COUNT(*) FROM products;"

# 2. 检查有多少商品有品牌
sqlite3 src/database/enriched_products.db "SELECT COUNT(*) FROM products WHERE brand IS NOT NULL AND brand != '';"

# 3. 检查品牌分布
sqlite3 src/database/enriched_products.db "SELECT brand, COUNT(*) FROM products WHERE brand IS NOT NULL GROUP BY brand LIMIT 20;"

# 4. 检查源JSONL是否有价格
head -5 src/indexing/data/data/processed/shoe_products.jsonl | jq '.price'

# 5. 检查材质、季节等关键字段的完整性
sqlite3 src/database/enriched_products.db "SELECT COUNT(*) FROM products WHERE material IS NOT NULL;"
sqlite3 src/database/enriched_products.db "SELECT COUNT(*) FROM products WHERE season IS NOT NULL;"
```

---

## 总结

**核心问题**: 系统的结构化筛选功能（价格、品牌）完全失效，导致多轮对话无法根据新增条件缩小商品范围。

**直接原因**:
1. 数据库缺少 `price` 字段
2. 品牌数据大量缺失或为空
3. 字段名映射不一致（title vs name, shoe_type vs type）

**间接原因**:
1. 数据导入流程未完整处理所有字段
2. 缺少数据完整性验证
3. 筛选失败时的回退逻辑不透明

**修复优先级**:
1. 🔴 **立即修复**: 添加price字段并重新导入数据
2. 🟡 **尽快修复**: 修正metadata字段映射
3. 🟢 **计划修复**: 改进品牌提取和筛选回退逻辑

修复后需要重新进行完整的三轮对话测试来验证。
