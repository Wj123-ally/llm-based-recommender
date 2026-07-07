"""
一键索引知识库文档 → Milvus knowledge_base_collection.

用法: conda run -n rag_env python scripts/index_knowledge_base.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.knowledge_base.document_processor import (
    index_file,
    get_knowledge_collection_stats,
)

DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "src" / "knowledge_base" / "documents"


def main():
    md_files = sorted(DOCUMENTS_DIR.glob("*.md"))
    if not md_files:
        print("No .md files found in", DOCUMENTS_DIR)
        return

    print(f"Found {len(md_files)} documents to index\n")

    total_chunks = 0
    for i, file_path in enumerate(md_files, 1):
        file_id = file_path.stem  # e.g. "01_dataset_shoe_overview"
        filename = file_path.name

        try:
            chunks = index_file(file_path, file_id, filename)
            total_chunks += chunks
            print(f"  [{i}/{len(md_files)}] {filename}: {chunks} chunks indexed")
        except Exception as e:
            print(f"  [{i}/{len(md_files)}] {filename}: FAILED - {e}")

    print(f"\n{'='*50}")
    print(f"Done. Total chunks indexed: {total_chunks}")
    stats = get_knowledge_collection_stats()
    print(f"Collection: {stats['collection_name']}")
    print(f"Total chunks: {stats['chunk_count']}")


if __name__ == "__main__":
    main()
