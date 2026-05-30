from pathlib import Path
import os
import pickle
import shutil
import sys
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BM25_INDEX_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_INDEX_PATH,
    DASHSCOPE_EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
from src.retriever.product_documents import build_product_content, chinese_tokenize


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗列名。
    """
    rename_dict = {
        "title": "商品标题",
        "image_url": "商品图片",
        "industry": "商品领域",
        "category1": "商品大类",
        "category2": "商品类别",
        "category3": "商品子类",
        "category4": "商品细分类",
        "attributes": "商品属性",
    }

    return df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})


def load_and_preprocess_data(n_samples: Optional[int] = 2000) -> pd.DataFrame:
    """
    加载并预处理原始数据。

    主要步骤：
    1. 检查原始 CSV 是否存在
    2. 读取 CSV
    3. 清洗列名
    4. 只保留推荐系统需要的字段
    5. 删除空值
    6. 可选抽样
    7. 保存处理后的 CSV
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")

    df = clean_column_names(df)

    valid_columns = [
        "id",
        "商品标题",
        "商品图片",
        "商品领域",
        "商品大类",
        "商品类别",
        "商品子类",
        "商品细分类",
        "商品属性",
    ]
    df = df[[col for col in valid_columns if col in df.columns]]
    df.dropna(subset=["商品标题"], inplace=True)
    df.fillna("", inplace=True)

    if n_samples and n_samples < len(df):
        df = df.sample(n_samples, random_state=42)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")

    return df


def generate_documents(use_csv_loader: bool = False) -> list:
    """
    把处理后的 CSV 数据转换成 LangChain Document。

    Document 可以理解成 LangChain 的标准资料格式：
    - page_content：主要文本内容，用于向量化和喂给大模型
    - metadata：结构化字段，用于过滤、排序、辅助检索
    """
    if use_csv_loader:
        from langchain_community.document_loaders import CSVLoader

        loader = CSVLoader(str(PROCESSED_DATA_PATH), encoding="utf-8-sig")
        return loader.load()

    from langchain_core.documents import Document

    df = pd.read_csv(PROCESSED_DATA_PATH, encoding="utf-8-sig")

    documents = []
    for index, row in df.iterrows():
        metadata = row.to_dict()
        product_id = str(metadata.get("id") or index)
        documents.append(
            Document(
                page_content=build_product_content(metadata),
                metadata=metadata,
                id=product_id,
            )
        )

    return documents


def initialize_embeddings_model():
    """
    初始化 DashScope embedding 模型。

    embedding 模型的作用：
    把文本转换成向量。

    例如：
    "红色连衣裙"
    -> [0.12, -0.08, 0.33, ...]

    后续 FAISS 和 Chroma 都需要用这个模型生成向量。
    """
    from src.shared import create_embedding_model

    return create_embedding_model()


def create_faiss_index(embeddings, documents: list) -> None:
    """
    创建并保存 FAISS 向量索引。
    """
    from langchain_community.vectorstores import FAISS

    faiss_index = FAISS.from_documents(documents, embeddings)
    faiss_index.save_local(str(FAISS_INDEX_PATH))


def create_chroma_index(embeddings, documents: list) -> None:
    """
    创建并保存 Chroma 向量库。
    """
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    if CHROMA_INDEX_PATH.exists():
        shutil.rmtree(CHROMA_INDEX_PATH)

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_INDEX_PATH),
    )
    vector_store.add_documents(documents)

    if hasattr(vector_store, "persist"):
        vector_store.persist()


def create_bm25_index(documents: list) -> None:
    """
    创建并保存 BM25 关键词索引。
    """
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    bm25_index = {
        "version": 2,
        "documents": documents,
        "tokenized_corpus": [
            chinese_tokenize(document.page_content) for document in documents
        ],
        "tokenizer": "jieba_or_cjk_fallback",
    }

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_index, f)


def embedding_pipeline(n_samples: Optional[int] = 100) -> None:
    """
    整个离线索引构建流程的入口函数。

    完整流程：
    1. 如果原始 CSV 不存在，则从 Hugging Face Datasets 下载数据
    2. 加载并预处理数据
    3. 把商品数据转换成 Document
    4. 初始化 DashScope embedding 模型
    5. 创建 FAISS 向量索引
    6. 创建 BM25 关键词索引
    7. 创建 Chroma 向量库
    """
    try:
        if not RAW_DATA_PATH.exists():
            from src.indexing.data_loader import download_data

            download_data()

        load_and_preprocess_data(n_samples)
        documents = generate_documents()
        embeddings = initialize_embeddings_model()

        create_faiss_index(embeddings, documents)
        create_bm25_index(documents)
        create_chroma_index(embeddings, documents)

        print("Embedding pipeline completed successfully.")
    except Exception as e:
        print(f"Failed to run embedding pipeline: {e}")
        raise

if __name__ == "__main__":
    embedding_pipeline(n_samples=2000)
