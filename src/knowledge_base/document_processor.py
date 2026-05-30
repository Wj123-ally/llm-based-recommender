"""
知识库文档处理器。

负责：
- 解析不同格式的上传文件（PDF、DOCX、TXT、MD）
- 将文档切分为适合检索的文本块
- 向量化并存入 Chroma 知识库专用 collection
- 删除文件时清理对应索引

设计原则：
- 索引失败不影响文件本身的保存（记录日志并向后兼容）
- 使用与商品检索相同的 embedding 模型，保持向量空间一致
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

import config as settings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 知识库专用的 Chroma collection 名，与商品 collection 隔离
KNOWLEDGE_COLLECTION_NAME = "knowledge_base_collection"
# 文档分块大小（字符数）
CHUNK_SIZE = 500
# 分块之间的重叠字符数
CHUNK_OVERLAP = 50


def _get_embedding_model():
    """延迟加载 embedding 模型，与商品检索共用同一模型。"""
    import os
    import time

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
                return DashScopeEmbeddings(model=settings.DASHSCOPE_EMBEDDING_MODEL)
            except TypeError:
                return DashScopeEmbeddings(
                    model_name=settings.DASHSCOPE_EMBEDDING_MODEL
                )
        except Exception as exc:
            last_error = exc
            logger.warning("初始化 embedding 失败，第 %s 次重试", attempt)
            if attempt < 3:
                time.sleep(2)

    raise RuntimeError("初始化 embedding 模型失败") from last_error


def _get_knowledge_chroma():
    """获取知识库专用 Chroma collection。"""
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    # 知识库索引存放在与商品索引相同的目录下，但使用独立的 collection 名
    persist_dir = str(settings.CHROMA_INDEX_PATH)

    return Chroma(
        collection_name=KNOWLEDGE_COLLECTION_NAME,
        embedding_function=_get_embedding_model(),
        persist_directory=persist_dir,
    )


def _parse_file(file_path: Path) -> str:
    """
    根据文件扩展名解析文件内容，返回纯文本。
    支持的格式：PDF、DOCX、TXT、MD。
    """
    extension = file_path.suffix.lower().lstrip(".")
    file_path_str = str(file_path)

    if extension == "txt" or extension == "md":
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return file_path.read_text(encoding="gbk")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"文件编码无法识别，请转换为 UTF-8：{file_path.name}"
                ) from exc

    if extension == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("请安装 pypdf：pip install pypdf>=3.0.0")

        reader = PdfReader(file_path_str)
        text_parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        if not text_parts:
            raise ValueError(f"PDF 文件没有可提取的文本：{file_path.name}")
        return "\n".join(text_parts)

    if extension == "docx":
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError("请安装 python-docx：pip install python-docx>=0.8.0")

        doc = DocxDocument(file_path_str)
        text_parts: list[str] = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)

        # 也提取表格中的文本
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    text_parts.append(row_text)

        if not text_parts:
            raise ValueError(f"DOCX 文件没有可提取的文本：{file_path.name}")
        return "\n".join(text_parts)

    raise ValueError(f"不支持的文件类型：.{extension}")


def _chunk_text(text: str, file_id: str, filename: str) -> list[Document]:
    """
    将文本切分为固定大小的块，每条携带 file_id 和来源文件名。
    """
    try:
        from langchain_core.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[
            {
                "file_id": file_id,
                "source_filename": filename,
                "chunk_size": CHUNK_SIZE,
            }
        ],
    )

    # 为每个 chunk 生成唯一 ID，格式：{file_id}_{chunk_index}
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{file_id}_{index}"

    return chunks


def _chunk_id_to_md5(chunk_id: str) -> str:
    """将 chunk_id 转为 MD5 哈希，作为 Chroma 的 doc ID（Chroma 要求 ASCII 兼容）。"""
    return hashlib.md5(chunk_id.encode("utf-8")).hexdigest()


def index_file(file_path: Path, file_id: str, filename: str) -> int:
    """
    解析文件、分块、向量化、写入知识库 Chroma collection。

    返回值：成功索引的文本块数量。

    异常：解析或向量化失败时抛出，调用方应捕获并记录日志。
    """
    try:
        text = _parse_file(file_path)
    except ImportError as exc:
        logger.warning("解析文件 %s 缺少依赖：%s", filename, exc)
        raise
    except Exception as exc:
        logger.exception("解析文件 %s 失败", filename)
        raise

    if not text.strip():
        raise ValueError(f"文件内容为空：{filename}")

    chunks = _chunk_text(text, file_id, filename)
    logger.info("文件 %s 解析出 %s 个文本块", filename, len(chunks))

    chroma = _get_knowledge_chroma()
    ids = [_chunk_id_to_md5(chunk.metadata["chunk_id"]) for chunk in chunks]

    # 分批写入，每批 20 条，避免一次性写入过多
    batch_size = 20
    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        chroma.add_documents(documents=batch_chunks, ids=batch_ids)

    logger.info("文件 %s 索引完成，共 %s 块", filename, len(chunks))
    return len(chunks)


def delete_file_index(file_id: str) -> int:
    """
    删除某个 file_id 在知识库 Chroma collection 中的所有 chunk。

    返回值：删除的文本块数量（可能为 0，表示该文件未被索引过）。

    实现方式：用 Chroma 的 metadata 过滤查询所有属于该 file_id 的 chunk ID，
    然后批量删除。
    """
    chroma = _get_knowledge_chroma()

    # 通过 metadata 过滤找到所有属于该文件的所有 chunk
    try:
        results = chroma.get(where={"file_id": file_id})
    except Exception:
        logger.exception("查询知识库 collection 失败")
        return 0

    chunk_ids = results.get("ids", [])
    if not chunk_ids:
        logger.info("文件 %s 没有可清理的索引记录", file_id)
        return 0

    # Chroma 通过 ID 列表删除
    chroma.delete(ids=chunk_ids)
    logger.info("已从知识库索引中删除文件 %s 的 %s 个文本块", file_id, len(chunk_ids))
    return len(chunk_ids)


def get_knowledge_collection_stats() -> dict[str, Any]:
    """
    返回知识库 collection 的统计信息。
    用于管理界面或健康检查。
    """
    try:
        chroma = _get_knowledge_chroma()
        count = chroma._collection.count()
        return {
            "collection_name": KNOWLEDGE_COLLECTION_NAME,
            "chunk_count": count,
        }
    except Exception:
        logger.exception("读取知识库统计信息失败")
        return {"collection_name": KNOWLEDGE_COLLECTION_NAME, "chunk_count": 0}
