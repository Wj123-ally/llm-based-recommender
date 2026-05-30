import logging
from email.parser import BytesParser
from email.policy import default
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.knowledge_base.file_store import (
    ALLOWED_EXTENSIONS,
    PROJECT_ROOT,
    DuplicateFileError,
    check_md5,
    delete_uploaded_file,
    is_allowed_file,
    list_uploaded_files,
    save_upload_file,
    upload_by_str,
)

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])


def _extract_upload_files(content_type: str, body: bytes) -> list[Any]:
    if not content_type.lower().startswith("multipart/form-data"):
        raise HTTPException(status_code=400, detail="请使用 multipart/form-data 上传文件")

    header = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8")
    message = BytesParser(policy=default).parsebytes(header + body)
    if not message.is_multipart():
        raise HTTPException(status_code=400, detail="上传表单格式不正确")

    files: list[Any] = []
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue

        filename = part.get_filename()
        if not filename:
            continue

        content = part.get_payload(decode=True) or b""
        files.append(
            SimpleNamespace(
                filename=filename,
                file=BytesIO(content),
            )
        )

    return files


@router.post("/upload")
async def upload_files(request: Request) -> dict[str, list[dict[str, Any]]]:
    files = _extract_upload_files(
        request.headers.get("content-type", ""),
        await request.body(),
    )
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件")

    invalid_files = [
        getattr(file, "filename", "")
        for file in files
        if not is_allowed_file(getattr(file, "filename", None))
    ]
    if invalid_files:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{', '.join(invalid_files)}。仅支持：{allowed}",
        )

    duplicated_files: list[str] = []
    seen_md5: dict[str, str] = {}
    for file in files:
        filename = getattr(file, "filename", "")
        source = getattr(file, "file", None)
        if source is None:
            raise HTTPException(status_code=400, detail="上传文件内容不能为空")

        file_md5 = upload_by_str(source)
        if file_md5 in seen_md5:
            duplicated_files.append(f"{filename} 与 {seen_md5[file_md5]} 内容重复")
            continue

        existing_file = check_md5(file_md5)
        if existing_file is not None:
            existing_name = existing_file.get("original_filename", "已上传文件")
            duplicated_files.append(f"{filename} 已存在（{existing_name}）")
            continue

        seen_md5[file_md5] = filename

    if duplicated_files:
        raise HTTPException(
            status_code=409,
            detail="；".join(duplicated_files),
        )

    try:
        saved_files = [save_upload_file(file) for file in files]
    except DuplicateFileError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 文件保存成功后，异步索引到知识库
    # 索引失败不影响上传成功，仅记录警告
    for entry in saved_files:
        try:
            file_path = PROJECT_ROOT / entry["saved_path"]
            from src.knowledge_base.document_processor import index_file

            chunk_count = index_file(
                file_path=file_path,
                file_id=entry["file_id"],
                filename=entry["original_filename"],
            )
            entry["indexed_chunks"] = chunk_count
        except ImportError as exc:
            logger.warning(
                "文件 %s 索引失败（缺少依赖）：%s", entry["original_filename"], exc
            )
        except Exception:
            logger.exception(
                "文件 %s 索引失败，文件本身已保存", entry["original_filename"]
            )

    return {"files": saved_files}


@router.get("/files")
def get_uploaded_files() -> dict[str, list[dict[str, Any]]]:
    return {"files": list_uploaded_files()}


@router.delete("/files/{file_id}")
def delete_file(file_id: str) -> dict[str, Any]:
    deleted = delete_uploaded_file(file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="文件不存在")

    # 同步清理知识库索引
    try:
        from src.knowledge_base.document_processor import delete_file_index

        deleted_chunks = delete_file_index(file_id)
        logger.info("文件 %s 已删除，清理了 %s 条索引", file_id, deleted_chunks)
    except ImportError:
        logger.warning("文件 %s 已删除，但缺少索引清理依赖", file_id)
    except Exception:
        logger.exception("清理文件 %s 的知识库索引失败", file_id)

    return {"file_id": file_id, "deleted": True}
