# 启动、数据与验证 Runbook

本文档说明生成服务如何启动、如何配置、数据如何落库、各层代码职责，以及修改后应如何验证。

它同时覆盖两种运行方式：

- 本地终端运行：API 和 UI 直接跑在 `conda rag_env` 环境中，适合日常开发调试。
- Docker 运行：Milvus、API、UI、Admin 由 Compose 管理，适合联调和演示。

## 修改后是否需要重建镜像

结论先看这里：

| 修改内容 | 本地终端运行 | Docker 运行 |
| --- | --- | --- |
| `src/recommender/**` 后端推荐链路 | 重启 API，`--reload` 下通常自动生效 | 只需 `docker compose restart api`，因为 API 挂载了 `./src:/app/src` |
| `src/retriever/**` 检索逻辑 | 重启 API，`--reload` 下通常自动生效 | 只需 `docker compose restart api` |
| `src/api/**` API 路由 | `--reload` 自动生效或重启 API | 只需 `docker compose restart api` |
| `config.py` | 重启 API | 只需 `docker compose restart api`，因为挂载了 `./config.py:/app/config.py` |
| `scripts/**` | 直接重新运行脚本 | 不需要重建镜像，API 容器挂载了 `./scripts:/app/scripts` |
| `src/ui/app.py` 用户端 UI | Streamlit 通常自动刷新，必要时重启 UI | 不需要重建镜像；UI 已挂载 `./src:/app/src`，必要时 `docker compose restart ui` |
| `src/ui/admin_app.py` 管理端 UI | Streamlit 通常自动刷新，必要时重启 Admin | 不需要重建镜像；Admin 已挂载 `./src:/app/src`，必要时 `docker compose restart admin` |
| `requirements.txt` / `Dockerfile` | 重新安装依赖 | 必须 `docker compose build ...` |
| `docker-compose.yml` 端口、挂载、环境变量 | 不适用 | 需要 `docker compose up -d --force-recreate ...`，依赖变化可能还要 build |
| 商品字段、JSONL 原始数据、落库逻辑 | 重跑落库和索引脚本 | 重跑落库和索引脚本；如果容器内跑脚本，确认挂载路径 |

当前 `docker-compose.yml` 的关键挂载：

```yaml
api:
  volumes:
    - ./config.py:/app/config.py
    - ./src:/app/src
    - ./scripts:/app/scripts
    - ./src/indexing/indexes:/app/src/indexing/indexes
    - ./src/indexing/data:/app/src/indexing/data
    - ./uploads:/app/uploads

ui:
  volumes:
    - ./config.py:/app/config.py
    - ./src:/app/src
    - ./scripts:/app/scripts
    - ./src/models:/app/src/models:ro
    - ./src/database:/app/src/database
    - ./src/indexing/indexes:/app/src/indexing/indexes
    - ./src/indexing/data:/app/src/indexing/data
    - ./uploads:/app/uploads

admin:
  volumes:
    - ./config.py:/app/config.py
    - ./src:/app/src
    - ./scripts:/app/scripts
    - ./src/models:/app/src/models:ro
    - ./src/database:/app/src/database
    - ./src/indexing/indexes:/app/src/indexing/indexes
    - ./src/indexing/data:/app/src/indexing/data
    - ./uploads:/app/uploads
```

所以 API、UI、Admin 的项目源码都已挂载。改 Python 源码后通常不需要重建镜像；如进程未自动加载，重启对应服务即可。

## 本地终端启动 API 和 UI

本地开发以 Conda 环境 `rag_env` 为准。开两个 PowerShell 终端。

### 终端 1：启动 API

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

验证 API：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

正常返回：

```json
{"status":"healthy"}
```

### 终端 2：启动用户 UI

```powershell
conda activate rag_env

$env:API_URL="http://127.0.0.1:8000"

streamlit run src/ui/app.py --server.port 8501
```

访问：

```text
http://127.0.0.1:8501
```

### 可选：启动管理端 UI

```powershell
conda activate rag_env

$env:API_URL="http://127.0.0.1:8000"

streamlit run src/ui/admin_app.py --server.port 8502
```

访问：

```text
http://127.0.0.1:8502
```

本地 API 仍然需要 Milvus 可用。如果不使用 Docker 启动 Milvus，需要有外部 Milvus 服务监听 `127.0.0.1:19530`。

## 生成服务如何配置

生成服务入口是 FastAPI：

```text
src/api/main.py
```

推荐接口路由：

```text
src/api/routers/recommender.py
```

推荐生成链路入口：

```text
src/recommender/graph.py
```

关键环境变量：

| 变量 | 示例 | 作用 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | `sk-...` | DashScope LLM 调用密钥，缺失时无法生成回答 |
| `DASHSCOPE_CHAT_MODEL` | `qwen-plus` | 生成/意图分析使用的聊天模型 |
| `MILVUS_URI` | 本地 `http://127.0.0.1:19530`，Docker `http://milvus:19530` | Milvus 连接地址 |
| `MILVUS_TEXT_COLLECTION_NAME` | `product_collection` | 商品文本向量 collection |
| `MILVUS_IMAGE_COLLECTION_NAME` | `product_image_collection` | 商品图片向量 collection |
| `MILVUS_KNOWLEDGE_COLLECTION_NAME` | `knowledge_base_collection` | 知识库向量 collection |
| `TEXT_EMBEDDING_PROVIDER` | `bge` | 文本向量模型来源，默认本地 BGE |
| `TEXT_EMBEDDING_DEVICE` | `cpu` | 本地 BGE 使用设备 |
| `BGE_TEXT_MODEL_PATH` | `src/models/bge-m3` | 商品/知识文本 embedding 模型路径 |
| `CROSS_ENCODER_MODEL_NAME` | `src/models/bge-reranker-v2-m3` | reranker 模型路径 |
| `ENABLE_MULTIMODAL_RETRIEVER` | `true` / `false` | 是否启用图片召回 |

`src/shared.py` 负责创建：

- Chat LLM：`create_chat_llm()`
- 文本 embedding：`create_embedding_model()`
- 知识库 Milvus collection：`ensure_knowledge_collection()`

`config.py` 负责读取环境变量并集中管理路径、collection 名称和检索参数。

## 数据如何落库

系统有三类数据：商品结构化数据、商品检索索引、知识库数据。

### 商品结构化数据：JSONL -> SQLite

原始商品数据默认来自：

```text
src/indexing/data/data/processed/shoe_products.jsonl
```

落库命令：

```powershell
conda run -n rag_env python -m src.database.seed_jsonl_data
```

主要逻辑：

- `src/database/seed_jsonl_data.py` 读取 JSONL。
- `src/database/schema.py` 定义 SQLite 表结构。
- `src/database/product_repo.py` 封装商品写入和查询。
- 生成 `content_text`，供 BM25 和向量索引使用。
- 写入 `src/database/enriched_products.db`。

检查 SQLite 商品数：

```powershell
conda run -n rag_env python -c "from src.database.product_repo import ProductRepo; repo=ProductRepo(); print(repo.count())"
```

### 商品检索索引：SQLite -> Milvus + BM25

构建商品文本向量和 BM25：

```powershell
conda run -n rag_env python -m src.indexing.embedding
```

主要逻辑：

- 从 SQLite 读取商品。
- `src/retriever/product_documents.py` 把商品 metadata 转成检索文档。
- `src/shared.create_embedding_model()` 创建 embedding 模型。
- 写入 Milvus `product_collection`。
- 写入本地 BM25 文件 `src/indexing/indexes/bm25.pkl`。

如果启用图片召回，构建图片向量：

```powershell
conda run -n rag_env python -m src.indexing.multimodal_embedding
```

检查 Milvus collection：

```powershell
conda run -n rag_env python -c "from pymilvus import MilvusClient; c=MilvusClient(uri='http://127.0.0.1:19530'); [print(name, c.get_collection_stats(name)) for name in c.list_collections()]"
```

### 知识库数据：文件 -> chunks -> Milvus

内置知识库索引：

```powershell
conda run -n rag_env python scripts/index_knowledge_base.py
```

上传知识库文件时：

- API 路由：`src/api/routers/knowledge_base.py`
- 文件保存：`uploads/`
- 文件记录：`uploads/files.json`
- 文档解析、分块、向量化、写入 Milvus：`src/knowledge_base/document_processor.py`
- 检索：`src/knowledge_base/knowledge_retriever.py`

删除知识库文件时，会删除本地文件记录，并调用 `delete_file_index()` 清理 Milvus 中对应 `file_id` 的 chunks。

## 各层代码职责

| 层 | 文件/目录 | 职责 |
| --- | --- | --- |
| API 层 | `src/api/main.py` | FastAPI app 创建、路由注册、健康检查 |
| 推荐 API | `src/api/routers/recommender.py` | 接收用户问题，调用 LangGraph，返回 answer/documents |
| 知识库 API | `src/api/routers/knowledge_base.py` | 上传、列表、删除知识库文件并维护索引 |
| 图编排 | `src/recommender/graph.py` | 串联意图分析、商品检索、知识检索、生成节点 |
| 意图分析 | `src/recommender/combined_analysis_node.py` | 判断是否鞋类问题、是否需要商品/知识、是否继承上下文 |
| 商品检索节点 | `src/recommender/self_query_node.py` | 调用混合检索，当前只把 top3 商品交给 LLM 和 UI |
| 知识检索节点 | `src/recommender/knowledge_retrieve_node.py` | 检索知识库片段 |
| 生成节点 | `src/recommender/rag_node.py` | 基于本轮 top3 商品和知识片段生成最终回答，维护短期推荐记忆 |
| Prompt | `src/recommender/utils.py` | RAG prompt 模板和回答规则 |
| 混合检索 | `src/retriever/hybrid_retriever.py` | dense、BM25、可选图片召回、rerank、metadata filter 合并 |
| 商品文档 | `src/retriever/product_documents.py` | 商品 metadata 到检索文本的转换 |
| Milvus 封装 | `src/retriever/milvus_store.py` | collection 创建、写入、搜索、统计 |
| SQLite | `src/database/**` | 商品库 schema、连接、repo、JSONL 落库 |
| 索引脚本 | `src/indexing/**` | 商品文本/图片索引构建 |
| UI | `src/ui/app.py` | Streamlit 用户端，展示对话和商品卡 |
| Admin | `src/ui/admin_app.py` | Streamlit 管理端，管理知识库等功能 |

当前推荐链路要点：

```text
用户问题
  -> FastAPI /recommend/
  -> LangGraph
  -> combined_analysis_node
  -> self_query_node: 每个 need_products=True 的轮次都执行混合召回，并选择 top3
  -> knowledge_retrieve_node: 可选知识召回
  -> rag_node: LLM 只解释本轮 top3，不再从候选中 5 选 3
  -> API 返回 answer + documents
  -> UI 按 documents 顺序展示商品1/商品2/商品3
```

检索上下文继承规则：

- 每一轮商品推荐都会走检索。
- “有没有拖鞋”“有洞洞鞋吗”“想看板鞋”这类本轮明确给出鞋型/品类的请求，只用本轮 query 检索。
- “女款”“不要黑色”“便宜点”“可爱一点”这类没有新鞋型的补充条件，才会把上一轮 query 拼入检索词。
- 上一轮商品记忆只用于解析“刚才那双”“第二个”“这几款里”这类指代，不作为本轮候选池。

## 修改后应该如何验证

### 通用快速验证

```powershell
conda run -n rag_env python -m compileall src\recommender src\api src\ui
conda run -n rag_env python -m pytest
```

API 健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

推荐接口：

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8000/recommend/" `
  -Method Post `
  -Body '{"question":"有洞洞鞋吗？我喜欢洞洞鞋"}' `
  -ContentType "application/json; charset=utf-8" `
  -TimeoutSec 180 `
  -UseBasicParsing
```

### 修改推荐链路后

重点验证：

- API 返回 HTTP 200。
- `answer` 中只出现本轮 `商品1`、`商品2`、`商品3`。
- `documents` 长度最多为 3。
- UI 商品卡顺序和回答里的商品编号一致。
- 新需求不会被上一轮商品池污染。例如先问“女款拖鞋”，再问“有洞洞鞋吗？我喜欢洞洞鞋”，第二轮应重新召回洞洞鞋 top3。
- 只有用户明确说“上一轮/刚才/这几个/第几个/那双”等，才使用上一轮商品记忆。

推荐用脚本验证多轮：

```powershell
E:\anaconda\envs\rag_env\python.exe -c "import requests, uuid; sid='check-'+uuid.uuid4().hex[:8]; base='http://127.0.0.1:8000/recommend/'; qs=['女款拖鞋','有洞洞鞋吗？我喜欢洞洞鞋']; \
for q in qs: \
    r=requests.post(base,json={'question':q,'session_id':sid},timeout=180); r.raise_for_status(); data=r.json(); print('===',q,'==='); print((data.get('answer') or '')[:800]); print('---docs---'); [print(i+1,d.get('metadata',{}).get('_source_rank'),d.get('metadata',{}).get('title')) for i,d in enumerate(data.get('documents',[]))]"
```

### 修改 UI 后

本地终端运行：

- Streamlit 通常会自动刷新。
- 不生效时停止后重新执行 `streamlit run src/ui/app.py --server.port 8501`。

Docker 运行：

- 当前 UI 已挂载 `./src:/app/src`。
- 修改 `src/ui/app.py` 后通常只需要：

```powershell
docker compose restart ui
```

修改 `src/ui/admin_app.py` 后通常只需要：

```powershell
docker compose restart admin
```

### 修改商品字段或落库逻辑后

需要重建 SQLite、BM25、Milvus 商品索引：

```powershell
conda run -n rag_env python -m src.database.seed_jsonl_data
conda run -n rag_env python -m src.indexing.embedding
```

如果影响图片召回：

```powershell
conda run -n rag_env python -m src.indexing.multimodal_embedding
```

检查：

```powershell
conda run -n rag_env python -c "from src.database.product_repo import ProductRepo; repo=ProductRepo(); print(repo.count())"
conda run -n rag_env python -c "from pymilvus import MilvusClient; c=MilvusClient(uri='http://127.0.0.1:19530'); [print(name, c.get_collection_stats(name)) for name in c.list_collections()]"
```

### 修改知识库处理后

```powershell
conda run -n rag_env python scripts/index_knowledge_base.py
```

再通过 Admin 或 `/knowledge/upload` 上传一个小文件，确认返回包含 `indexed_chunks`，并检查 `knowledge_base_collection` 的 row_count。

---

以下章节保留 Docker 启动与健康检查细节。

## 0. 项目重要配置速查

### 默认访问地址

| 服务 | 宿主机访问地址 | 容器内端口 | 说明 |
| --- | --- | --- | --- |
| API | `http://127.0.0.1:8000` | `8000` | FastAPI 后端服务 |
| API 健康检查 | `http://127.0.0.1:8000/health` | `8000` | 返回 `{"status":"healthy"}` |
| UI | `http://127.0.0.1:8510` | `8501` | Streamlit 用户界面 |
| UI 健康检查 | `http://127.0.0.1:8510/_stcore/health` | `8501` | 返回 `200 ok` |
| Admin | `http://127.0.0.1:8511` | `8502` | 管理端界面 |
| Milvus | `http://127.0.0.1:19530` | `19530` | 向量数据库 |
| Milvus Metrics | `http://127.0.0.1:9091` | `9091` | Milvus 健康/指标端口 |

### 默认 Docker 容器

| 容器名 | 服务 | 作用 |
| --- | --- | --- |
| `recommender-api` | `api` | 后端 API |
| `recommender-ui` | `ui` | Streamlit 用户界面 |
| `recommender-admin` | `admin` | 管理端 |
| `recommender-milvus` | `milvus` | Milvus standalone |
| `recommender-etcd` | `etcd` | Milvus 元数据依赖 |
| `recommender-minio` | `minio` | Milvus 对象存储依赖 |

### `.env` 关键端口

| 变量 | 推荐值 | 说明 |
| --- | --- | --- |
| `API_PORT` | `8000` | 宿主机 API 端口 |
| `UI_PORT` | `8510` | 宿主机 UI 端口，映射到容器内 `8501` |
| `ADMIN_PORT` | `8511` | 宿主机 Admin 端口，映射到容器内 `8502` |
| `MILVUS_PORT` | `19530` | 宿主机 Milvus 端口 |
| `MILVUS_METRICS_PORT` | `9091` | 宿主机 Milvus metrics 端口 |
| `MILVUS_URI` | `http://milvus:19530` | Docker 容器内 API 连接 Milvus 的地址 |

注意：容器内服务之间访问 Milvus 时应使用 `http://milvus:19530`；宿主机上的脚本或浏览器访问 Milvus 时使用 `http://127.0.0.1:19530`。

## 1. 启动前检查

确认项目根目录存在 `.env`，并至少包含这些配置：

```env
DASHSCOPE_API_KEY=你的 DashScope Key
API_PORT=8000
UI_PORT=8510
ADMIN_PORT=8511
MILVUS_PORT=19530
MILVUS_METRICS_PORT=9091
```

如果修改过 `.env` 里的端口，必须重建对应容器，否则旧容器可能继续使用旧端口映射。

## 2. 标准启动命令

在项目根目录执行：

```powershell
docker compose up -d
```

如果使用 GPU API 覆盖配置，执行：

```powershell
docker compose -f docker-compose.yml -f docker-compose.api-gpu.yml up -d
```

首次启动或修改过 `.env`、`docker-compose*.yml` 后，建议强制重建 API 和 UI：

```powershell
docker compose up -d --force-recreate api ui
```

如果需要重建全部服务：

```powershell
docker compose up -d --force-recreate
```

## 3. 必查状态

启动后先查看容器状态：

```powershell
docker compose ps
```

正常状态应类似：

```text
recommender-milvus   Up ... healthy   0.0.0.0:19530->19530/tcp
recommender-api      Up ... healthy   0.0.0.0:8000->8000/tcp
recommender-ui       Up ... healthy   0.0.0.0:8510->8501/tcp
recommender-admin    Up ... healthy   0.0.0.0:8511->8502/tcp
```

重点确认：

- `recommender-milvus` 必须是 `healthy`
- `recommender-api` 必须是 `healthy`
- `recommender-ui` 必须有宿主机端口映射，例如 `8510->8501/tcp`

如果 UI 只显示 `8501/tcp`，没有 `0.0.0.0:8510->8501/tcp`，说明容器端口没有发布，需要重建 UI：

```powershell
docker compose up -d --force-recreate ui
```

## 4. 健康检查

### Milvus 端口

```powershell
Test-NetConnection -ComputerName 127.0.0.1 -Port 19530
```

`TcpTestSucceeded` 应为 `True`。

### API

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

正常返回：

```json
{"status":"healthy"}
```

### UI

```powershell
Invoke-WebRequest http://127.0.0.1:8510/_stcore/health -UseBasicParsing
```

正常返回 `200 ok`。

浏览器访问：

```text
http://127.0.0.1:8510
```

### Admin

浏览器访问：

```text
http://127.0.0.1:8511
```

## 5. 检查 Milvus 数据

在本机 `conda rag_env` 环境中执行：

```powershell
conda run -n rag_env python -c "from pymilvus import MilvusClient; c=MilvusClient(uri='http://127.0.0.1:19530'); print(c.list_collections()); [print(name, c.get_collection_stats(name)) for name in c.list_collections()]"
```

正常应至少看到：

```text
product_collection
product_image_collection
```

并且 `row_count` 大于 0。

如果 collection 不存在或 `row_count` 为 0，需要重新构建/导入索引数据。

## 6. 验证推荐接口

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8000/recommend/" `
  -Method Post `
  -Body '{"question":"推荐一双黑色通勤鞋"}' `
  -ContentType "application/json; charset=utf-8" `
  -TimeoutSec 180 `
  -UseBasicParsing
```

正常应返回 HTTP `200`，并包含 `answer` 字段。

注意：首次请求会加载本地 embedding、reranker、jieba 等模型，可能明显慢于后续请求。

## 7. 查看日志

### Milvus 日志

```powershell
docker logs --tail 200 recommender-milvus
```

Milvus 日志中出现 recovery、load collection、checkpoint 等信息通常是正常行为。重点关注 `ERROR`、`FATAL`、`panic`。

### API 日志

```powershell
docker logs --tail 200 recommender-api
```

重点检查：

- 是否找到 SQLite 数据库
- 是否找到 BM25 索引
- 是否能启动 FastAPI
- 推荐请求是否报 DashScope、Milvus、模型加载错误

### UI 日志

```powershell
docker logs --tail 120 recommender-ui
```

正常应看到 Streamlit 启动在容器内 `8501` 端口。

## 8. 常见问题

### UI 无法访问 `http://127.0.0.1:8510`

先看端口映射：

```powershell
docker compose ps
```

如果 UI 只显示：

```text
8501/tcp
```

而不是：

```text
0.0.0.0:8510->8501/tcp
```

执行：

```powershell
docker compose up -d --force-recreate ui
```

### API healthy，但推荐没有商品

检查 Milvus collection 数据量：

```powershell
conda run -n rag_env python -c "from pymilvus import MilvusClient; c=MilvusClient(uri='http://127.0.0.1:19530'); [print(name, c.get_collection_stats(name)) for name in c.list_collections()]"
```

如果 `row_count` 为 0，需要重新导入数据和索引。

### Milvus 端口不通

检查容器：

```powershell
docker compose ps
docker logs --tail 200 recommender-milvus
```

如果 Milvus 不是 healthy，通常需要先确认 etcd 和 minio 是否 healthy：

```powershell
docker compose ps etcd minio milvus
```

### 修改端口后不生效

Docker 不会自动修改已创建容器的端口映射。修改 `.env` 后执行：

```powershell
docker compose up -d --force-recreate api ui admin
```

## 9. 推荐的一键验证流程

每次启动后按顺序执行：

```powershell
docker compose up -d
docker compose ps
Test-NetConnection -ComputerName 127.0.0.1 -Port 19530
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-WebRequest http://127.0.0.1:8510/_stcore/health -UseBasicParsing
```

全部通过后，再打开：

```text
http://127.0.0.1:8510
```

## 10. 两个项目共用同一份 Milvus 数据

如果有两个重构自同一项目的代码仓库，需要共用同一份 Milvus 数据，推荐只启动一套 Milvus/etcd/minio，两个项目分别启动自己的 API/UI/Admin。

### 推荐端口规划

| 服务 | 项目 A | 项目 B |
| --- | --- | --- |
| API | `8000` | `8010` |
| UI | `8510` | `8520` |
| Admin | `8511` | `8521` |
| Milvus | `19530` | 共用项目 A 的 `19530` |
| Milvus Metrics | `9091` | 共用项目 A 的 `9091` |

### 项目 A：启动完整服务

项目 A 负责启动 Milvus、API、UI、Admin：

```powershell
docker compose -p recommender_a up -d
```

项目 A 的 `.env`：

```env
API_PORT=8000
UI_PORT=8510
ADMIN_PORT=8511
MILVUS_PORT=19530
MILVUS_METRICS_PORT=9091
MILVUS_URI=http://milvus:19530
```

### 项目 B：只启动业务服务

项目 B 不再启动 Milvus/etcd/minio，只启动 API、UI、Admin：

```powershell
docker compose -p recommender_b up -d api ui admin
```

项目 B 的 `.env`：

```env
API_PORT=8010
UI_PORT=8520
ADMIN_PORT=8521

MILVUS_URI=http://host.docker.internal:19530
```

说明：

- `host.docker.internal:19530` 表示项目 B 的容器通过宿主机端口访问项目 A 暴露出来的 Milvus。
- 如果两个项目在同一个 Docker network 内，也可以改为访问项目 A 的 Milvus 容器名或网络别名。
- 项目 B 不要执行完整的 `docker compose up -d`，否则会尝试再启动一套 Milvus/etcd/minio，导致端口或容器名冲突。

### 避免容器名冲突

当前 compose 文件里使用了固定 `container_name`，例如：

```text
recommender-api
recommender-ui
recommender-admin
recommender-milvus
```

如果两个项目都直接使用相同的 `container_name`，即使端口不同，也会发生容器名冲突。

推荐做法：

1. 使用不同 Compose project name：

```powershell
docker compose -p recommender_a up -d
docker compose -p recommender_b up -d api ui admin
```

2. 或者在项目 B 的 compose 文件中修改 `container_name`，例如：

```yaml
container_name: recommender-b-api
container_name: recommender-b-ui
container_name: recommender-b-admin
```

更长期的做法是移除 `container_name`，让 Docker Compose 自动按 project name 生成容器名。

### 双项目启动后检查

```powershell
docker compose -p recommender_a ps
docker compose -p recommender_b ps
```

项目 A 应包含完整依赖：

```text
milvus
etcd
minio
api
ui
admin
```

项目 B 应只包含：

```text
api
ui
admin
```

访问地址：

```text
项目 A UI: http://127.0.0.1:8510
项目 B UI: http://127.0.0.1:8520
项目 A API: http://127.0.0.1:8000/health
项目 B API: http://127.0.0.1:8010/health
```
