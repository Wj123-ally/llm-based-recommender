"""
共享工厂模块。

统一项目中重复出现的工厂函数：
- create_chat_llm()          → 通义千问 LLM（3 处重复已消除）
- create_embedding_model()   → 文本 embedding（BGE 本地模型 / DashScope 可选）

所有函数均带 @lru_cache 单例缓存和重试逻辑。
"""

import logging
import os
import time
from functools import lru_cache
from typing import Any

import config as settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Chat LLM
# ─────────────────────────────────────────────────────────────


def create_chat_llm(temperature: float = 0.0, max_tokens: int | None = None, streaming: bool = False) -> Any:
    """
    创建通义千问聊天模型（DashScope ChatTongyi）。

    项目中统一使用此函数创建 LLM 实例，不再在各模块中重复。

    Args:
        temperature: 生成温度。分类任务用 0（默认），生成任务用 0.3。
        max_tokens: 最大输出 token 数。None 表示使用模型默认值。
        streaming: 是否启用流式输出。

    Returns:
        ChatTongyi 实例。

    Raises:
        EnvironmentError: 未设置 DASHSCOPE_API_KEY 环境变量。
    """
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise EnvironmentError("请先设置环境变量 DASHSCOPE_API_KEY")

    try:
        from langchain_community.chat_models import ChatTongyi
    except ImportError:
        from langchain_community.chat_models.tongyi import ChatTongyi

    kwargs: dict[str, Any] = {
        "model": os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
        "temperature": temperature,
        "streaming": streaming,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    logger.debug("创建 ChatTongyi: model=%s temperature=%s max_tokens=%s streaming=%s",
                 kwargs["model"], temperature, max_tokens, streaming)
    return ChatTongyi(**kwargs)


# ─────────────────────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def create_embedding_model() -> Any:
    """
    创建文本 embedding 模型。

    默认使用本地 BGE 模型，避免商品索引和知识库索引构建依赖外部
    embedding 网络接口。必要时可通过 TEXT_EMBEDDING_PROVIDER=dashscope
    切回 DashScope。

    Returns:
        embedding 模型实例。

    Raises:
        EnvironmentError: 使用 DashScope 时未设置 DASHSCOPE_API_KEY。
        RuntimeError: 初始化失败。
    """
    provider = getattr(settings, "TEXT_EMBEDDING_PROVIDER", "bge")

    if provider == "bge":
        model_path = getattr(settings, "BGE_TEXT_MODEL_PATH")
        if not model_path.exists():
            raise FileNotFoundError(f"BGE embedding model not found: {model_path}")

        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            configured_device = getattr(settings, "TEXT_EMBEDDING_DEVICE", "auto")
            if configured_device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    logger.exception("检测 CUDA 失败，BGE embedding 回退到 CPU")
                    device = "cpu"
            else:
                device = configured_device
                if device == "cuda":
                    try:
                        import torch

                        if not torch.cuda.is_available():
                            raise RuntimeError(
                                "TEXT_EMBEDDING_DEVICE=cuda, but CUDA is not available. "
                                "Install a CUDA-enabled PyTorch build in rag_env first."
                            )
                    except RuntimeError:
                        raise
                    except Exception as exc:
                        raise RuntimeError(
                            "TEXT_EMBEDDING_DEVICE=cuda, but CUDA availability could not be checked."
                        ) from exc

            logger.info("创建 BGE embedding: model=%s device=%s", model_path, device)
            return HuggingFaceEmbeddings(
                model_name=str(model_path),
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            raise RuntimeError(f"初始化 BGE embedding 模型失败: {model_path}") from exc

    if provider != "dashscope":
        raise ValueError(f"Unsupported TEXT_EMBEDDING_PROVIDER: {provider}")

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise EnvironmentError("请先设置环境变量 DASHSCOPE_API_KEY")

    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            try:
                from langchain_community.embeddings import DashScopeEmbeddings
            except ImportError:
                from langchain_dashscope import DashScopeEmbeddings

            try:
                model = DashScopeEmbeddings(model=settings.DASHSCOPE_EMBEDDING_MODEL)
            except TypeError:
                model = DashScopeEmbeddings(
                    model_name=settings.DASHSCOPE_EMBEDDING_MODEL
                )

            logger.debug("Embedding 模型初始化成功: %s", settings.DASHSCOPE_EMBEDDING_MODEL)
            return model

        except Exception as exc:
            last_error = exc
            logger.warning("初始化 embedding 失败，第 %s/3 次重试: %s", attempt, exc)
            if attempt < 3:
                time.sleep(2)

    raise RuntimeError("初始化 embedding 模型失败（3 次重试后）") from last_error


# ─────────────────────────────────────────────────────────────
# Knowledge Base Helpers
# ─────────────────────────────────────────────────────────────


def get_knowledge_collection_name() -> str:
    """获取知识库 Milvus collection 名称。"""
    return settings.MILVUS_KNOWLEDGE_COLLECTION_NAME


def ensure_knowledge_collection(dimension: int) -> None:
    """
    确保知识库 Milvus collection 存在，不存在则创建。

    Args:
        dimension: 向量维度（需与 embedding 模型输出维度一致）。
    """
    from src.retriever.milvus_store import (
        create_collection,
        get_milvus_client,
    )

    client = get_milvus_client()
    collection_name = get_knowledge_collection_name()
    if not client.has_collection(collection_name):
        create_collection(collection_name, dimension)
        logger.info("创建知识库 Milvus collection: %s (dim=%s)", collection_name, dimension)
