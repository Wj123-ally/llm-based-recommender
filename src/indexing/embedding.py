import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as settings  # noqa: E402
from config import BM25_INDEX_PATH  # noqa: E402
from src.retriever.product_documents import build_product_content, chinese_tokenize  # noqa: E402
from src.retriever.milvus_store import count_collection, reset_collection, upsert_documents  # noqa: E402

def generate_documents_from_db() -> list:
    from langchain_core.documents import Document
    from src.database.product_repo import ProductRepo

    repo = ProductRepo()
    rows = repo.conn.execute("SELECT * FROM products ORDER BY id").fetchall()

    documents = []
    for row in rows:
        metadata = dict(row)
        product_id = str(metadata.get("id", ""))
        content = metadata.get("content_text", "") or build_product_content(metadata)
        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
                id=product_id,
            )
        )

    return documents


def initialize_embeddings_model():
    from src.shared import create_embedding_model

    return create_embedding_model()


def create_milvus_text_index(embeddings, documents: list) -> None:
    batch_size = 100
    total = len(documents)
    if not documents:
        return

    first_vector = embeddings.embed_documents([documents[0].page_content])[0]
    reset_collection(settings.MILVUS_TEXT_COLLECTION_NAME, len(first_vector))
    upsert_documents(
        settings.MILVUS_TEXT_COLLECTION_NAME,
        [documents[0]],
        [first_vector],
    )
    print(f"  Milvus write progress: 1/{total}")

    for start in range(0, total, batch_size):
        if start == 0:
            start = 1
        end = min(start + batch_size, total)
        batch = documents[start:end]
        if not batch:
            continue
        contents = [doc.page_content for doc in batch]
        vectors = embeddings.embed_documents(contents)
        upsert_documents(settings.MILVUS_TEXT_COLLECTION_NAME, batch, vectors)
        print(f"  Milvus write progress: {end}/{total}")

    print(
        "  Milvus collection count: "
        f"{count_collection(settings.MILVUS_TEXT_COLLECTION_NAME)}"
    )

def create_bm25_index(documents: list) -> None:
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    bm25_index = {
        "version": 2,
        "documents": documents,
        "tokenized_corpus": [
            chinese_tokenize(document.page_content) for document in documents
        ],
        "tokenizer": "jieba_or_cjk_fallback",
    }

    with open(BM25_INDEX_PATH, "wb") as file:
        pickle.dump(bm25_index, file)


def embedding_pipeline() -> None:
    documents = generate_documents_from_db()
    print(f"[1/4] Loaded {len(documents)} product documents from SQLite")

    print("[2/4] Initializing text embedding model")
    embeddings = initialize_embeddings_model()

    print("[3/4] Building Milvus text index")
    create_milvus_text_index(embeddings, documents)

    print("[4/4] Building BM25 text index")
    create_bm25_index(documents)
    print("Embedding pipeline completed successfully.")


if __name__ == "__main__":
    embedding_pipeline()
