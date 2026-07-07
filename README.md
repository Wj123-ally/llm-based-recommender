# LLM-Based Shoe Recommender

> 基于大语言模型的智能鞋类推荐系统

一个结合 RAG（检索增强生成）技术的智能推荐系统，通过 LangGraph 编排多轮对话和混合检索，为用户提供个性化的鞋类商品推荐。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ 核心特性

- 🤖 **智能对话理解**：基于大语言模型的意图分析和多轮对话管理
- 🔍 **混合检索**：结合向量检索（Milvus）、BM25 和元数据过滤的混合检索策略
- 📚 **知识增强**：支持上传自定义知识库文档，增强推荐的专业性
- 🎯 **精准重排序**：使用 Cross-Encoder 模型对检索结果重排序
- 🔄 **上下文记忆**：支持多轮对话，保留用户偏好上下文
- 🐳 **容器化部署**：提供完整的 Docker Compose 配置，一键启动

## 🏗️ 技术架构

```
用户查询
    ↓
FastAPI 接口层
    ↓
LangGraph 编排层 ──→ 意图分析 ──→ 并行检索
                    ↓              ↓
                主题判断        商品检索 + 知识检索
                    ↓              ↓
                RAG 生成 ←─────── 上下文融合
                    ↓
                推荐结果
```

**技术栈：**
- **后端框架**：FastAPI
- **LLM**：通义千问（DashScope API）
- **编排引擎**：LangGraph
- **向量数据库**：Milvus
- **结构化数据库**：SQLite (FTS5)
- **Embedding 模型**：BGE-M3
- **重排序模型**：BGE-Reranker-V2-M3

## 🚀 快速开始

### 前置要求

- Python 3.10+
- Docker & Docker Compose
- 通义千问 API Key ([获取地址](https://dashscope.console.aliyun.com/))

### Docker 部署（推荐）

1. **克隆项目**
   ```bash
   git clone https://github.com/Wj123-ally/llm-based-recommender.git
   cd llm-based-recommender
   ```

2. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，填入你的 DASHSCOPE_API_KEY
   ```

3. **下载模型文件**
   
   模型文件较大，需要从 Hugging Face 下载：
   ```bash
   # BGE-M3 Embedding 模型
   huggingface-cli download BAAI/bge-m3 --local-dir src/models/bge-m3
   
   # BGE-Reranker-V2-M3 重排序模型
   huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir src/models/bge-reranker-v2-m3
   ```

4. **启动服务**
   ```bash
   docker compose up -d
   ```

5. **验证部署**
   ```bash
   curl http://localhost:8000/health
   ```

### 本地开发

详细的本地开发指南请参考 [DOCKER_RUNBOOK.md](./DOCKER_RUNBOOK.md)

1. **创建 Conda 环境**
   ```bash
   conda env create -f environment.yml
   conda activate rag_env
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置环境变量**
   ```bash
   export DASHSCOPE_API_KEY="your-api-key"
   export MILVUS_URI="http://localhost:19530"
   # ... 其他配置见 .env.example
   ```

4. **初始化数据**
   ```bash
   # 初始化商品数据库
   python -m src.database.seed_jsonl_data
   
   # 构建检索索引
   python -m src.indexing.embedding
   
   # 索引知识库文档
   python scripts/index_knowledge_base.py
   ```

5. **启动 API 服务**
   ```bash
   python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
   ```

## 📖 API 使用

### 推荐接口

```bash
curl -X POST http://localhost:8000/recommend/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "推荐一双适合通勤的黑色皮鞋"
  }'
```

**响应示例：**
```json
{
  "question": "推荐一双适合通勤的黑色皮鞋",
  "thread_id": "uuid-string",
  "answer": "根据您的需求，为您推荐以下几款...",
  "documents": [
    {
      "page_content": "商品详情...",
      "metadata": {
        "product_id": "123",
        "name": "经典商务皮鞋",
        "price": 299.0
      }
    }
  ]
}
```

### 知识库管理

```bash
# 上传知识文档
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@guide.pdf"

# 查询已上传文件
curl http://localhost:8000/knowledge/files

# 删除知识文档
curl -X DELETE http://localhost:8000/knowledge/files/{file_id}
```

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_hybrid_retriever.py
```

## 📁 项目结构

```
llm-based-recommender/
├── src/
│   ├── api/                 # FastAPI 接口层
│   │   ├── main.py         # 应用入口
│   │   └── routers/        # 路由定义
│   ├── recommender/         # LangGraph 编排层
│   │   ├── graph.py        # 推荐流程图
│   │   └── *_node.py       # 各个节点实现
│   ├── retriever/           # 检索层
│   │   ├── hybrid_retriever.py  # 混合检索
│   │   └── milvus_store.py      # Milvus 操作
│   ├── database/            # 数据层
│   │   ├── schema.py       # 数据库表结构
│   │   └── product_repo.py # 商品仓库
│   ├── knowledge_base/      # 知识库
│   │   ├── documents/      # 内置知识文档
│   │   └── document_processor.py
│   └── shared.py           # 共享工具
├── tests/                   # 测试用例
├── scripts/                 # 工具脚本
├── docker-compose.yml       # Docker 编排
├── Dockerfile              # 镜像构建
├── requirements.txt        # Python 依赖
└── README.md              # 项目说明
```

## ⚙️ 配置说明

主要配置项（在 `.env` 中设置）：

```bash
# LLM 配置
DASHSCOPE_API_KEY=your-key        # 必填
DASHSCOPE_CHAT_MODEL=qwen-plus

# Milvus 配置
MILVUS_URI=http://localhost:19530
MILVUS_TEXT_COLLECTION_NAME=product_collection
MILVUS_KNOWLEDGE_COLLECTION_NAME=knowledge_base_collection

# Embedding 配置
TEXT_EMBEDDING_PROVIDER=bge
TEXT_EMBEDDING_DEVICE=cpu         # cpu 或 cuda
BGE_TEXT_MODEL_PATH=src/models/bge-m3

# 检索参数
DENSE_RETRIEVER_TOP_K=20
BM25_RETRIEVER_TOP_K=20
RERANK_TOP_N=5
```

完整配置项请参考 [.env.example](./.env.example)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📮 联系方式

- 作者：Wangjuan
- GitHub: [@Wj123-ally](https://github.com/Wj123-ally)

---

⭐ 如果这个项目对你有帮助，欢迎 Star！
