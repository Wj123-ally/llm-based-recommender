import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
INDEXING_DIR = PROJECT_ROOT / "src" / "indexing"
DATA_DIR = INDEXING_DIR / "data"
MODELS_DIR = PROJECT_ROOT / "src" / "models"

PROCESSED_JSONL_DATA_PATH = DATA_DIR / "data" / "processed" / "shoe_products_only_shoes_cleaned.jsonl"
INDEX_DIR = INDEXING_DIR / "indexes"
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_TEXT_COLLECTION_NAME = os.getenv(
    "MILVUS_TEXT_COLLECTION_NAME",
    "product_collection",
)
MILVUS_IMAGE_COLLECTION_NAME = os.getenv(
    "MILVUS_IMAGE_COLLECTION_NAME",
    "product_image_collection",
)
MILVUS_KNOWLEDGE_COLLECTION_NAME = os.getenv(
    "MILVUS_KNOWLEDGE_COLLECTION_NAME",
    "knowledge_base_collection",
)
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
CROSS_ENCODER_RERANKER_PATH = INDEX_DIR / "hybrid_retriever.pkl"

DATABASE_PATH = PROJECT_ROOT / "src" / "database" / "enriched_products.db"

DASHSCOPE_EMBEDDING_MODEL = os.getenv(
    "DASHSCOPE_EMBEDDING_MODEL",
    "text-embedding-v4",
)
TEXT_EMBEDDING_PROVIDER = os.getenv("TEXT_EMBEDDING_PROVIDER", "bge").lower()
BGE_TEXT_MODEL_PATH = Path(
    os.getenv("BGE_TEXT_MODEL_PATH", str(MODELS_DIR / "bge-m3"))
)
TEXT_EMBEDDING_DEVICE = os.getenv("TEXT_EMBEDDING_DEVICE", "cuda").lower()
CLIP_IMAGE_MODEL_NAME = os.getenv(
    "CLIP_IMAGE_MODEL_NAME",
    "sentence-transformers/clip-ViT-B-32",
)
CLIP_TEXT_MODEL_NAME = os.getenv(
    "CLIP_TEXT_MODEL_NAME",
    "sentence-transformers/clip-ViT-B-32-multilingual-v1",
)
CLIP_MODEL_CACHE_DIR = Path(
    os.getenv("CLIP_MODEL_CACHE_DIR", str(INDEX_DIR / "hf_cache"))
)
CLIP_LOCAL_FILES_ONLY = os.getenv(
    "CLIP_LOCAL_FILES_ONLY",
    "true",
).lower() in {"1", "true", "yes", "on"}
ENABLE_MULTIMODAL_RETRIEVER = os.getenv(
    "ENABLE_MULTIMODAL_RETRIEVER",
    "true",
).lower() in {"1", "true", "yes", "on"}
MULTIMODAL_RETRIEVER_TOP_K = int(os.getenv("MULTIMODAL_RETRIEVER_TOP_K", "20"))
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL_NAME",
    str(MODELS_DIR / "bge-reranker-v2-m3"),
)
DENSE_RETRIEVER_TOP_K = int(os.getenv("DENSE_RETRIEVER_TOP_K", "20"))
DENSE_SIMILARITY_THRESHOLD = float(os.getenv("DENSE_SIMILARITY_THRESHOLD", "0.2"))
BM25_RETRIEVER_TOP_K = int(os.getenv("BM25_RETRIEVER_TOP_K", "20"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_MIN_SCORE = float(os.getenv("BM25_MIN_SCORE", "0"))
