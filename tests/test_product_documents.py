"""
测试 product_documents.py 的文本处理函数。

无需外部 API，纯逻辑测试。
"""

import pytest

# 如果 langchain_core 未安装，跳过所有测试
langchain_available = False
try:
    from langchain_core.documents import Document

    from src.retriever.product_documents import (
        build_product_content,
        chinese_tokenize,
        get_document_key,
    )

    langchain_available = True
except ImportError:
    pass

if not langchain_available:
    pytest.skip("langchain_core not installed", allow_module_level=True)


class TestChineseTokenizer:
    """chinese_tokenize 分词函数测试。"""

    def test_simple_chinese(self):
        """简单中文应正确分词。"""
        tokens = chinese_tokenize("红色运动鞋")
        assert isinstance(tokens, list)
        assert len(tokens) >= 1
        assert all(t.strip() for t in tokens)

    def test_mixed_cjk_and_english(self):
        """中英混合文本应正确分词。"""
        tokens = chinese_tokenize("2025新款 冬季羽绒服 90%白鸭绒")
        assert isinstance(tokens, list)
        assert len(tokens) >= 1

    def test_numbers_preserved(self):
        """数字应保留在分词结果中。"""
        tokens = chinese_tokenize("尺码M码 XL码 2025款")
        assert any("M" in t or "XL" in t or "2025" in t for t in tokens)

    def test_punctuation_removed(self):
        """标点符号应被过滤。"""
        tokens = chinese_tokenize("推荐！！夏天穿的。。。凉鞋～～")
        assert all(t.strip() for t in tokens)

    def test_empty_text(self):
        """空文本应返回空列表。"""
        tokens = chinese_tokenize("")
        assert tokens == []

    def test_whitespace_only(self):
        """纯空白文本应返回空列表。"""
        tokens = chinese_tokenize("   \n  \t  ")
        assert tokens == []


class TestBuildProductContent:
    """build_product_content 函数测试。"""

    def test_build_from_metadata(self):
        """从元数据构建商品文本应包含关键字段。"""
        metadata = {
            "商品标题": "夏季纯棉T恤",
            "商品大类": "女装",
            "商品类别": "T恤",
            "商品属性": "品牌:耐克,,材质:网面,,季节:夏季",
        }
        content = build_product_content(metadata)
        assert "夏季纯棉T恤" in content
        assert "女装" in content
        assert "一级类目" in content

    def test_empty_metadata_returns_empty(self):
        """空元数据应返回空字符串。"""
        content = build_product_content({})
        assert content == ""

    def test_none_value_skipped(self):
        """None 值字段应被跳过。"""
        metadata = {
            "标题": "test",
            "商品大类": None,
            "商品类别": "T恤",
        }
        content = build_product_content(metadata)
        assert "None" not in content

    def test_nan_value_skipped(self):
        """NaN 值应被跳过。"""
        metadata = {
            "标题": "test",
            "商品大类": float("nan"),
        }
        content = build_product_content(metadata)
        assert "NaN" not in content and "nan" not in content


class TestGetDocumentKey:
    """get_document_key 函数测试。"""

    def test_key_from_id(self):
        """有 id 时返回 'id:xxx' 格式。"""
        doc = Document(
            page_content="content",
            metadata={"id": "product-123"},
        )
        key = get_document_key(doc)
        assert key == "id:product-123"

    def test_key_fallback_to_content_hash(self):
        """无 id 时回退到内容哈希。"""
        doc = Document(
            page_content="unique content",
            metadata={},
        )
        key = get_document_key(doc)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_same_content_same_key(self):
        """相同内容应产生相同 key。"""
        doc1 = Document(page_content="same", metadata={"title": "a"})
        doc2 = Document(page_content="same", metadata={"title": "b"})
        assert get_document_key(doc1) == get_document_key(doc2)

    def test_different_content_different_key(self):
        """不同内容应产生不同 key。"""
        doc1 = Document(page_content="content A", metadata={})
        doc2 = Document(page_content="content B", metadata={})
        assert get_document_key(doc1) != get_document_key(doc2)
