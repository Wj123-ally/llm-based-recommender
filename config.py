import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
INDEXING_DIR = PROJECT_ROOT / "src" / "indexing"
DATA_DIR = INDEXING_DIR / "data"

RAW_DATA_PATH = DATA_DIR / "products.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_products.csv"
INDEX_DIR = INDEXING_DIR / "indexes"
FAISS_INDEX_PATH = INDEX_DIR / "faiss"
CHROMA_INDEX_PATH = INDEX_DIR / "chroma"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
CROSS_ENCODER_RERANKER_PATH = INDEX_DIR / "hybrid_retriever.pkl"

DATABASE_PATH = PROJECT_ROOT / "src" / "database" / "enriched_products.db"

DASHSCOPE_EMBEDDING_MODEL = os.getenv(
    "DASHSCOPE_EMBEDDING_MODEL",
    "text-embedding-v4",
)
CHROMA_COLLECTION_NAME = "product_collection"
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL_NAME",
    "BAAI/bge-reranker-base",
)
CHROMA_RETRIEVER_TOP_K = int(os.getenv("CHROMA_RETRIEVER_TOP_K", "20"))
BM25_RETRIEVER_TOP_K = int(os.getenv("BM25_RETRIEVER_TOP_K", "20"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
CHROMA_SIMILARITY_THRESHOLD = float(
    os.getenv("CHROMA_SIMILARITY_THRESHOLD", "0.2")
)
BM25_MIN_SCORE = float(os.getenv("BM25_MIN_SCORE", "0"))
