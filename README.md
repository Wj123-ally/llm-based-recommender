# API 后端接入指南

本文档面向后端接入、部署和维护，重点说明生成服务如何启动、如何配置、数据如何落库、各层代码职责，以及修改后应如何验证。前端 Streamlit 页面只作为可选调用方，不作为本文档重点。

## 1. 服务组成

当前后端由四类能力组成：

- `FastAPI API`：入口为 `src/api/main.py`，对外暴露健康检查、推荐生成、知识库管理接口。
- `LangGraph 推荐生成链路`：入口为 `src/recommender/graph.py`，负责意图分析、商品/知识检索和最终回答生成。
- `SQLite 商品结构化库`：默认文件为 `src/database/enriched_products.db`，保存商品主数据和 FTS5 全文索引。
- `Milvus 向量库`：保存商品文本向量、可选图片向量和知识库文档向量。

核心请求链路：

```text
POST /recommend/
  -> src/api/routers/recommender.py
  -> create_recommender_graph()
  -> combined_analysis_node
  -> parallel_retrieve_node
      -> self_query_retrieve -> hybrid_retriever -> Milvus + BM25 + SQLite metadata filter
      -> knowledge_retrieve_node -> Milvus knowledge collection
  -> rag_recommender -> DashScope ChatTongyi
  -> API 返回 answer + documents
```

## 2. API 启动方式

### Docker 启动

推荐优先使用 Docker Compose 启动完整依赖：

```powershell
docker compose up -d
```

启动后检查：

```powershell
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
```

API 容器启动命令来自 `scripts/docker-entrypoint.sh`：

```text
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API 启动前会执行两类检查：

- `ensure_database()`：检查 `/app/src/database/enriched_products.db` 是否存在；缺失且 `BOOTSTRAP=true` 时执行 `python -m src.database.seed_jsonl_data`。
- `ensure_indexes()`：检查 `/app/src/indexing/indexes/bm25.pkl` 是否存在；缺失且 `BOOTSTRAP=true` 时执行 `python -m src.indexing.embedding`。

注意：`BOOTSTRAP=true` 只覆盖 SQLite 商品库和商品检索索引的准备，不会自动索引 `src/knowledge_base/documents/*.md`。内置知识库需要额外执行 `python scripts/index_knowledge_base.py`。

### 本地开发启动

本地开发以 Conda 环境 `rag_env` 为准，不依赖 `uv.lock`。启动前需要先保证 Milvus 可用、依赖已安装，并在当前 PowerShell 会话里显式设置运行环境变量。

如果本机还没有 `rag_env`，先创建环境：

```powershell
conda env create -f environment.yml
```

如果已经存在 `rag_env`，但 `requirements.txt` 有更新，执行：

```powershell
conda run -n rag_env pip install -r requirements.txt
```

```powershell
conda activate rag_env

$env:DASHSCOPE_API_KEY="你的 DashScope Key"
$env:DASHSCOPE_CHAT_MODEL="qwen-plus"
$env:MILVUS_URI="http://127.0.0.1:19530"
$env:MILVUS_TEXT_COLLECTION_NAME="product_collection"
$env:MILVUS_IMAGE_COLLECTION_NAME="product_image_collection"
$env:MILVUS_KNOWLEDGE_COLLECTION_NAME="knowledge_base_collection"
$env:TEXT_EMBEDDING_PROVIDER="bge"
$env:TEXT_EMBEDDING_DEVICE="cpu"
$env:BGE_TEXT_MODEL_PATH="src/models/bge-m3"
$env:CROSS_ENCODER_MODEL_NAME="src/models/bge-reranker-v2-m3"
$env:ENABLE_MULTIMODAL_RETRIEVER="true"

python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

说明：

- `conda run -n rag_env ...` 适合一次性命令；长期开发建议先 `conda activate rag_env`，再运行 `python ...`。
- 本项目代码不会自动加载 `.env` 文件；本地启动必须在 shell 中设置关键环境变量，或用你自己的启动脚本提前注入。
- `TEXT_EMBEDDING_DEVICE` 本地建议固定为 `cpu`，除非当前 `rag_env` 已安装 CUDA 可用的 PyTorch。

## 3. 关键配置

配置集中在 `config.py` 和 `.env`。`config.py` 会读取环境变量，不存在时使用默认值。

必须配置：

```env
DASHSCOPE_API_KEY=你的 DashScope Key
DASHSCOPE_CHAT_MODEL=qwen-plus
```

常用运行配置：

```env
MILVUS_URI=http://localhost:19530
MILVUS_TEXT_COLLECTION_NAME=product_collection
MILVUS_IMAGE_COLLECTION_NAME=product_image_collection
MILVUS_KNOWLEDGE_COLLECTION_NAME=knowledge_base_collection

TEXT_EMBEDDING_PROVIDER=bge
TEXT_EMBEDDING_DEVICE=cpu
BGE_TEXT_MODEL_PATH=src/models/bge-m3
CROSS_ENCODER_MODEL_NAME=src/models/bge-reranker-v2-m3

DENSE_RETRIEVER_TOP_K=20
DENSE_SIMILARITY_THRESHOLD=0.2
BM25_RETRIEVER_TOP_K=20
BM25_MIN_SCORE=0
RERANK_TOP_N=5
RRF_K=60

ENABLE_MULTIMODAL_RETRIEVER=false
BOOTSTRAP=false
```

Docker 内部 API 连接 Milvus 时使用 `http://milvus:19530`，宿主机脚本或本地 API 连接 Milvus 时通常使用 `http://127.0.0.1:19530` 或 `http://localhost:19530`。

## 4. 数据如何落库

### 商品结构化数据：SQLite

SQLite 文件位置：

```text
src/database/enriched_products.db
```

建表和索引逻辑：

- `src/database/schema.py`：创建 `products` 主表、普通索引、`products_fts` FTS5 虚拟表和同步触发器。
- `src/database/connection.py`：提供 SQLite 单例连接，启用 WAL 和外键。
- `src/database/product_repo.py`：封装商品插入、更新、按 ID 查询、组合过滤、FTS 查询。

从 JSONL 初始化商品库：

```powershell
conda run -n rag_env python -m src.database.seed_jsonl_data
```

默认读取：

```text
src/indexing/data/data/processed/shoe_products.jsonl
```

转换逻辑在 `src/database/seed_jsonl_data.py`：

- 将 JSONL 商品字段映射到 `products` 表字段。
- 生成 `content_text`，供 BM25 和向量索引使用。
- 调用 `ProductRepo.insert_batch()` 批量写入 SQLite。
- `products_fts` 通过触发器随 `products` 自动更新。

### 商品检索索引：Milvus + BM25

商品索引构建入口：

```powershell
conda run -n rag_env python -m src.indexing.embedding
```

执行内容：

- 从 SQLite `products` 表读取商品。
- 使用 `src.shared.create_embedding_model()` 创建 embedding 模型。
- 写入 Milvus collection：默认 `product_collection`。
- 生成 BM25 文件：`src/indexing/indexes/bm25.pkl`。

检索时 `src/retriever/hybrid_retriever.py` 会同时使用：

- Milvus dense vector search。
- 本地 BM25 index。
- 可选图片向量检索。
- RRF 融合。
- Cross-Encoder rerank。
- 必要时回 SQLite 做 metadata 过滤。

### 知识库数据：上传文件 + Milvus

知识库文件上传接口：

```text
POST /knowledge/upload
GET /knowledge/files
DELETE /knowledge/files/{file_id}
```

文件存储逻辑在 `src/knowledge_base/file_store.py`：

- 原始文件保存到 `uploads/raw/{file_id}.{ext}`。
- 元数据保存到 `uploads/files.json`。
- 通过 MD5 防止重复上传。
- 支持 `pdf`、`docx`、`txt`、`md`。

上传成功后，`src/api/routers/knowledge_base.py` 会调用 `src.knowledge_base.document_processor.index_file()`：

- 解析文件文本。
- 按 500 字符、50 字符 overlap 切块。
- 使用与商品一致的 embedding 模型向量化。
- 写入 Milvus knowledge collection，默认 `knowledge_base_collection`。

内置 Markdown 知识库索引命令：

```powershell
conda run -n rag_env python scripts/index_knowledge_base.py
```

删除知识库文件时会先删除本地文件和 `uploads/files.json` 记录，再调用 `delete_file_index()` 清理 Milvus 中对应 `file_id` 的 chunk。

## 5. 各层代码职责

API 层：

- `src/api/main.py`：创建 FastAPI app，注册路由，提供 `/` 和 `/health`。
- `src/api/routers/recommender.py`：推荐接口，维护 `thread_id` cookie，调用 LangGraph。
- `src/api/routers/knowledge_base.py`：知识库上传、列表、删除接口。

推荐编排层：

- `src/recommender/graph.py`：定义 LangGraph 节点和路由。
- `src/recommender/state.py`：定义流程状态字段。
- `src/recommender/combined_analysis_node.py`：一次 LLM 调用完成主题判断、意图分析和部分结构化过滤条件提取。
- `src/recommender/self_query_node.py`：按意图执行商品检索，并格式化商品上下文。
- `src/recommender/knowledge_retrieve_node.py`：执行知识库检索。
- `src/recommender/rag_node.py`：基于商品和知识上下文调用 LLM 生成最终回答。
- `src/recommender/utils.py`：RAG prompt 模板等工具。

检索层：

- `src/retriever/hybrid_retriever.py`：商品混合检索、过滤、RRF 融合和 rerank。
- `src/retriever/milvus_store.py`：Milvus collection 创建、upsert、search、count。
- `src/retriever/product_documents.py`：商品内容拼接、中文分词、文档 key。
- `src/retriever/multimodal_retriever.py`：可选图片向量检索。

数据层：

- `src/database/schema.py`：SQLite DDL 和迁移式补列。
- `src/database/connection.py`：SQLite 连接。
- `src/database/product_repo.py`：商品读写仓库。
- `src/database/seed_jsonl_data.py`：JSONL 到 SQLite 的初始化脚本。

模型和共享工厂：

- `src/shared.py`：创建 Chat LLM、embedding 模型，确保知识库 collection 存在。
- `config.py`：路径、模型、Milvus、检索参数配置。

## 6. API 接入示例

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

推荐生成：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/recommend/" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body '{"question":"推荐一双适合通勤的黑色皮鞋"}'
```

正常返回结构：

```json
{
  "question": "推荐一双适合通勤的黑色皮鞋",
  "thread_id": "uuid",
  "answer": "生成的推荐回答",
  "documents": []
}
```

`thread_id` 会写入 cookie，用于多轮追问时保留 LangGraph memory 上下文。

## 7. 修改后如何验证

### 通用验证

```powershell
conda run -n rag_env pytest
```

当前已有测试重点覆盖：

- `tests/test_hybrid_retriever.py`：RRF、过滤条件合并、BM25 过滤等纯逻辑。
- `tests/test_product_documents.py`：商品文档拼接、中文分词、文档 key。
- `tests/test_file_store.py`：上传文件存储元数据和重复校验。
- `tests/test_state.py`：推荐状态结构。

### 修改 API 路由后

```powershell
conda run -n rag_env pytest
conda run -n rag_env python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
Invoke-RestMethod http://127.0.0.1:8000/health
```

再调用对应接口确认 HTTP 状态码和返回字段。

### 修改推荐生成链路后

至少验证：

```powershell
conda run -n rag_env pytest tests/test_hybrid_retriever.py tests/test_product_documents.py
```

然后启动 API，调用：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/recommend/" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body '{"question":"推荐一双适合夏天走路的透气运动鞋"}'
```

检查：

- `answer` 不为空。
- `documents` 有商品时包含 `page_content` 和 `metadata`。
- API 日志中没有 DashScope、Milvus、rerank、embedding 初始化异常。

### 修改数据字段或商品落库后

执行：

```powershell
conda run -n rag_env python -m src.database.seed_jsonl_data
conda run -n rag_env python -m src.indexing.embedding
```

然后检查 SQLite：

```powershell
conda run -n rag_env python -c "from src.database.product_repo import ProductRepo; repo=ProductRepo(); print(repo.count())"
```

检查 Milvus collection：

```powershell
conda run -n rag_env python -c "from src.retriever.milvus_store import count_collection; import config; print(count_collection(config.MILVUS_TEXT_COLLECTION_NAME))"
```

两边数量都应大于 0。字段结构变化时，要确认 `schema.py`、`seed_jsonl_data.py`、`product_documents.py`、`hybrid_retriever.py` 中字段名保持一致。

### 修改知识库上传或文档处理后

执行：

```powershell
conda run -n rag_env pytest tests/test_file_store.py
conda run -n rag_env python scripts/index_knowledge_base.py
```

检查 knowledge collection：

```powershell
conda run -n rag_env python -c "from src.knowledge_base.document_processor import get_knowledge_collection_stats; print(get_knowledge_collection_stats())"
```

再通过 `/knowledge/upload` 上传一个小的 `.md` 或 `.txt` 文件，确认返回中包含 `indexed_chunks`。

### Docker 验证

```powershell
docker compose up -d --force-recreate api
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
docker logs --tail 200 recommender-api
```

重点看 API 日志：

- 是否找到 SQLite 数据库。
- 是否找到 BM25 索引。
- 是否成功启动 Uvicorn。
- 推荐请求是否出现 DashScope Key、Milvus 连接、模型路径、CUDA/CPU 设备相关错误。

## 8. 常见接入注意事项

- `DASHSCOPE_API_KEY` 缺失时，推荐服务无法生成回答；Docker entrypoint 会直接阻止 API 启动。
- 默认 embedding provider 是本地 BGE。模型目录不存在时，商品索引、知识库索引和检索都会失败。
- `TEXT_EMBEDDING_DEVICE=cuda` 要求当前环境安装 CUDA 可用的 PyTorch；不确定时先用 `cpu`。
- 修改商品字段后，只改 SQLite 不够，还要重建 BM25 和 Milvus 商品向量索引。
- 上传知识库文件保存成功不等于索引一定成功；索引异常会记录日志，应检查 `indexed_chunks` 或 collection stats。
- `ENABLE_MULTIMODAL_RETRIEVER=true` 依赖 CLIP 模型和图片向量 collection；后端接入阶段如果只验证文本推荐，建议先设为 `false` 降低变量。
