import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "uploads"
RAW_UPLOAD_DIR = UPLOAD_DIR / "raw"
METADATA_PATH = UPLOAD_DIR / "files.json"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
_METADATA_LOCK = Lock()


class DuplicateFileError(ValueError):
    def __init__(self, file: dict[str, Any]) -> None:
        self.file = file
        filename = file.get("original_filename", "unknown")
        super().__init__(f"文件已存在：{filename}")


def _safe_filename(filename: str | None) -> str:
    normalized = (filename or "").replace("\\", "/").split("/")[-1].strip()
    normalized = normalized.replace("\x00", "")
    if not normalized or normalized in {".", ".."}:
        raise ValueError("文件名不能为空")

    return normalized


def _get_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"不支持的文件类型，仅支持：{allowed}")

    return extension


def is_allowed_file(filename: str | None) -> bool:
    try:
        safe_name = _safe_filename(filename)
        _get_extension(safe_name)
    except ValueError:
        return False

    return True


def _ensure_upload_dirs() -> None:
    RAW_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _load_metadata() -> list[dict[str, Any]]:
    if not METADATA_PATH.exists():
        return []

    try:
        data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []

    return [item for item in data if isinstance(item, dict)]


def _write_metadata(files: list[dict[str, Any]]) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = METADATA_PATH.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(files, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(METADATA_PATH)


def _hash_file_obj(file_obj: Any) -> str:
    md5 = hashlib.md5()
    while True:
        chunk = file_obj.read(1024 * 1024)
        if not chunk:
            break
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        md5.update(chunk)

    return md5.hexdigest()


def _hash_path(path: Path) -> str:
    with path.open("rb") as file_obj:
        return _hash_file_obj(file_obj)


def upload_by_str(content: bytes | str | Any) -> str:
    if isinstance(content, str):
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    if isinstance(content, bytes):
        return hashlib.md5(content).hexdigest()

    try:
        position = content.tell()
    except (AttributeError, OSError):
        position = None

    try:
        content.seek(0)
    except (AttributeError, OSError):
        pass

    file_md5 = _hash_file_obj(content)

    if position is not None:
        try:
            content.seek(position)
        except OSError:
            pass

    return file_md5


def _backfill_missing_md5(files: list[dict[str, Any]]) -> bool:
    changed = False
    for item in files:
        if item.get("md5"):
            continue

        saved_path = item.get("saved_path")
        if not isinstance(saved_path, str) or not saved_path:
            continue

        try:
            path = _resolve_saved_path(saved_path)
        except ValueError:
            continue

        if not path.exists():
            continue

        item["md5"] = _hash_path(path)
        changed = True

    return changed


def _find_by_md5(files: list[dict[str, Any]], file_md5: str) -> dict[str, Any] | None:
    return next(
        (item for item in files if item.get("md5") == file_md5),
        None,
    )


def check_md5(file_md5: str) -> dict[str, Any] | None:
    with _METADATA_LOCK:
        files = _load_metadata()
        if _backfill_missing_md5(files):
            _write_metadata(files)

        return _find_by_md5(files, file_md5)


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _resolve_saved_path(saved_path: str) -> Path:
    path = Path(saved_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    resolved_path = path.resolve()
    raw_dir = RAW_UPLOAD_DIR.resolve()
    if resolved_path != raw_dir and raw_dir not in resolved_path.parents:
        raise ValueError("保存路径不合法")

    return resolved_path


def save_upload_file(file: Any) -> dict[str, Any]:
    original_filename = _safe_filename(getattr(file, "filename", None))
    extension = _get_extension(original_filename)

    _ensure_upload_dirs()
    source = getattr(file, "file", None)
    if source is None:
        raise ValueError("上传文件内容不能为空")

    file_md5 = upload_by_str(source)
    file_id = uuid.uuid4().hex
    saved_path = RAW_UPLOAD_DIR / f"{file_id}.{extension}"

    metadata = {
        "file_id": file_id,
        "original_filename": original_filename,
        "saved_path": _relative_to_project(saved_path),
        "extension": extension,
        "size": 0,
        "md5": file_md5,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    with _METADATA_LOCK:
        files = _load_metadata()
        changed = _backfill_missing_md5(files)
        duplicate = _find_by_md5(files, file_md5)
        if duplicate is not None:
            if changed:
                _write_metadata(files)
            raise DuplicateFileError(duplicate)

        try:
            source.seek(0)
        except (AttributeError, OSError):
            pass

        try:
            with saved_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            metadata["size"] = saved_path.stat().st_size
        except Exception:
            try:
                saved_path.unlink()
            except FileNotFoundError:
                pass
            raise

        files.append(metadata)
        _write_metadata(files)

    return metadata


def list_uploaded_files() -> list[dict[str, Any]]:
    with _METADATA_LOCK:
        files = _load_metadata()
        if _backfill_missing_md5(files):
            _write_metadata(files)

        return files


def delete_uploaded_file(file_id: str) -> bool:
    with _METADATA_LOCK:
        files = _load_metadata()
        file_to_delete = next(
            (item for item in files if item.get("file_id") == file_id),
            None,
        )
        if file_to_delete is None:
            return False

        saved_path = file_to_delete.get("saved_path")
        if isinstance(saved_path, str) and saved_path:
            path = _resolve_saved_path(saved_path)
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        remaining_files = [
            item for item in files if item.get("file_id") != file_id
        ]
        _write_metadata(remaining_files)

    return True
