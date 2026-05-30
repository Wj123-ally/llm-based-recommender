"""
测试 file_store.py 的文件存储逻辑。

纯逻辑测试，无外部依赖。使用 io.BytesIO 模拟文件上传流。
"""

import hashlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import src.knowledge_base.file_store as file_store


def _make_file_obj(content: bytes) -> SimpleNamespace:
    """创建模拟的文件上传对象。

    使用 io.BytesIO 模拟文件流，天然支持 seek/tell/read，
    与 save_upload_file 内部 hash(读)+seek(0)+copyfileobj(读) 的流程兼容。
    """
    return SimpleNamespace(
        filename=None,
        file=io.BytesIO(content),
    )


def _make_file_obj_named(filename: str, content: bytes) -> SimpleNamespace:
    """创建带文件名的模拟上传对象。"""
    obj = _make_file_obj(content)
    obj.filename = filename
    return obj


@pytest.fixture
def temp_upload_dirs():
    """创建临时上传目录并 patch 模块常量。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        upload_dir = root / "uploads"
        raw_dir = upload_dir / "raw"
        metadata_path = upload_dir / "files.json"

        with (
            mock.patch.object(file_store, "PROJECT_ROOT", root),
            mock.patch.object(file_store, "UPLOAD_DIR", upload_dir),
            mock.patch.object(file_store, "RAW_UPLOAD_DIR", raw_dir),
            mock.patch.object(file_store, "METADATA_PATH", metadata_path),
        ):
            yield {
                "root": root,
                "upload_dir": upload_dir,
                "raw_dir": raw_dir,
                "metadata_path": metadata_path,
            }


class TestSafeFilename:
    """_safe_filename 函数测试。"""

    def test_normal_filename(self):
        assert file_store._safe_filename("test.pdf") == "test.pdf"

    def test_path_traversal_stripped(self):
        """路径穿越字符应被清理。"""
        assert file_store._safe_filename("../../../etc/passwd") == "passwd"

    def test_empty_filename_raises(self):
        with pytest.raises(ValueError, match="文件名不能为空"):
            file_store._safe_filename("")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="文件名不能为空"):
            file_store._safe_filename(None)

    def test_dot_only_raises(self):
        with pytest.raises(ValueError, match="文件名不能为空"):
            file_store._safe_filename(".")

    def test_backslash_converted(self):
        result = file_store._safe_filename("folder\\test.pdf")
        assert result == "test.pdf"


class TestGetExtension:
    """_get_extension 函数测试。"""

    def test_allowed_extension(self):
        assert file_store._get_extension("test.pdf") == "pdf"

    def test_uppercase_extension(self):
        assert file_store._get_extension("test.PDF") == "pdf"

    def test_disallowed_extension_raises(self):
        with pytest.raises(ValueError, match="不支持的文件类型"):
            file_store._get_extension("test.exe")

    def test_no_extension_raises(self):
        with pytest.raises(ValueError, match="不支持的文件类型"):
            file_store._get_extension("testfile")


class TestIsAllowedFile:
    """is_allowed_file 函数测试。"""

    def test_pdf_allowed(self):
        assert file_store.is_allowed_file("test.pdf") is True

    def test_docx_allowed(self):
        assert file_store.is_allowed_file("test.docx") is True

    def test_txt_allowed(self):
        assert file_store.is_allowed_file("test.txt") is True

    def test_exe_not_allowed(self):
        assert file_store.is_allowed_file("test.exe") is False

    def test_none_not_allowed(self):
        assert file_store.is_allowed_file(None) is False


class TestMD5Hashing:
    """MD5 哈希函数测试。"""

    def test_hash_str_consistent(self):
        file_obj = io.BytesIO(b"hello")
        result1 = file_store._hash_file_obj(file_obj)
        result2 = hashlib.md5(b"hello").hexdigest()
        assert result1 == result2

    def test_hash_bytes(self):
        h = file_store.upload_by_str(b"hello world")
        assert h == hashlib.md5(b"hello world").hexdigest()

    def test_hash_str_content(self):
        h = file_store.upload_by_str("你好世界")
        assert h == hashlib.md5("你好世界".encode("utf-8")).hexdigest()


class TestFileOperations:
    """文件保存/列表/删除集成测试。"""

    def test_save_and_list(self, temp_upload_dirs):
        """保存文件后应出现在列表中，且内容完整。"""
        file_obj = _make_file_obj_named("test.txt", b"hello content")
        saved = file_store.save_upload_file(file_obj)

        assert saved["original_filename"] == "test.txt"
        assert saved["extension"] == "txt"
        assert saved["md5"] == hashlib.md5(b"hello content").hexdigest()
        assert "file_id" in saved
        assert "saved_path" in saved

        # 列表应包含该文件
        files = file_store.list_uploaded_files()
        assert len(files) == 1
        assert files[0]["file_id"] == saved["file_id"]

        # 文件应已写入磁盘
        saved_path = Path(temp_upload_dirs["root"]) / saved["saved_path"]
        assert saved_path.exists()
        assert saved_path.read_bytes() == b"hello content"

    def test_duplicate_detection(self, temp_upload_dirs):
        """重复上传相同内容应抛出 DuplicateFileError。"""
        content = b"duplicate test content"

        file1 = _make_file_obj_named("test1.txt", content)
        file_store.save_upload_file(file1)

        file2 = _make_file_obj_named("test2.txt", content)
        with pytest.raises(file_store.DuplicateFileError, match="文件已存在"):
            file_store.save_upload_file(file2)

    def test_delete(self, temp_upload_dirs):
        """删除文件后列表应为空。"""
        file_obj = _make_file_obj_named("to_delete.txt", b"content")
        saved = file_store.save_upload_file(file_obj)

        # 删除成功
        assert file_store.delete_uploaded_file(saved["file_id"]) is True
        # 列表为空
        assert len(file_store.list_uploaded_files()) == 0
        # 再次删除失败
        assert file_store.delete_uploaded_file(saved["file_id"]) is False

    def test_delete_nonexistent(self, temp_upload_dirs):
        """删除不存在的文件应返回 False。"""
        assert file_store.delete_uploaded_file("nonexistent-id") is False

    def test_multi_file_list(self, temp_upload_dirs):
        """多文件上传应全部出现。"""
        for i in range(3):
            content = f"content-{i}".encode()
            file_store.save_upload_file(
                _make_file_obj_named(f"test{i}.txt", content)
            )
        assert len(file_store.list_uploaded_files()) == 3


class TestMetadata:
    """元数据读写测试。"""

    def test_empty_metadata_returns_list(self, temp_upload_dirs):
        """无元数据文件时应返回空列表。"""
        assert file_store._load_metadata() == []

    def test_write_and_read_metadata(self, temp_upload_dirs):
        """写入后应正确读取。"""
        file_store._write_metadata([
            {"file_id": "abc", "original_filename": "test.pdf"}
        ])
        loaded = file_store._load_metadata()
        assert len(loaded) == 1
        assert loaded[0]["file_id"] == "abc"

    def test_corrupt_metadata_returns_empty(self, temp_upload_dirs):
        """损坏的 JSON 返回空列表。"""
        temp_upload_dirs["metadata_path"].parent.mkdir(parents=True, exist_ok=True)
        temp_upload_dirs["metadata_path"].write_text(
            "not valid json", encoding="utf-8"
        )
        assert file_store._load_metadata() == []

    def test_non_list_metadata_returns_empty(self, temp_upload_dirs):
        """非列表格式的 JSON 返回空列表。"""
        temp_upload_dirs["metadata_path"].parent.mkdir(parents=True, exist_ok=True)
        temp_upload_dirs["metadata_path"].write_text(
            '{"not": "a list"}', encoding="utf-8"
        )
        assert file_store._load_metadata() == []
