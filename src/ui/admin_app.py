import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")
ALLOWED_FILE_TYPES = ["pdf", "docx", "txt", "md"]


st.set_page_config(
    page_title="资料管理后台",
    page_icon=":file_folder:",
    layout="wide",
)


def build_api_url(path: str) -> str:
    return f"{API_URL.rstrip('/')}/{path.lstrip('/')}"


def format_file_size(size: int | float | str | None) -> str:
    if size is None:
        return "-"

    try:
        value = float(size)
    except (TypeError, ValueError):
        return "-"

    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024

    return f"{value:.1f} GB"


def format_uploaded_at(value: Any) -> str:
    if not value:
        return "-"

    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text

    if dt.tzinfo is not None:
        dt = dt.astimezone()

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def get_uploaded_file_size(uploaded_file: Any) -> int:
    size = getattr(uploaded_file, "size", None)
    if isinstance(size, int):
        return size

    return len(uploaded_file.getvalue())


def extract_error_detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text

    if isinstance(detail, str):
        return detail

    return json.dumps(detail, ensure_ascii=False)


def fetch_files() -> tuple[list[dict[str, Any]], str | None]:
    try:
        response = requests.get(build_api_url("/knowledge/files"), timeout=20)
    except requests.exceptions.RequestException as exc:
        return [], f"读取文件列表失败：{exc}"

    if response.status_code != 200:
        return [], f"读取文件列表失败：{extract_error_detail(response)}"

    try:
        data = response.json()
    except ValueError:
        return [], "读取文件列表失败：后端返回格式不正确"

    files = data.get("files", [])
    if not isinstance(files, list):
        return [], "读取文件列表失败：后端返回格式不正确"

    return files, None


def upload_files(uploaded_files: list[Any]) -> tuple[bool, int, int | None, str | None]:
    files = [
        (
            "files",
            (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            ),
        )
        for uploaded_file in uploaded_files
    ]

    try:
        response = requests.post(
            build_api_url("/knowledge/upload"),
            files=files,
            timeout=120,
        )
    except requests.exceptions.RequestException as exc:
        return False, 0, None, str(exc)

    if response.status_code != 200:
        return False, 0, response.status_code, extract_error_detail(response)

    try:
        data = response.json()
    except ValueError:
        return True, len(uploaded_files), response.status_code, None

    saved_files = data.get("files", [])
    count = len(saved_files) if isinstance(saved_files, list) else len(uploaded_files)
    return True, count, response.status_code, None


def delete_file(file_id: str) -> tuple[bool, str | None]:
    try:
        response = requests.delete(
            build_api_url(f"/knowledge/files/{file_id}"),
            timeout=20,
        )
    except requests.exceptions.RequestException as exc:
        return False, str(exc)

    if response.status_code != 200:
        return False, extract_error_detail(response)

    return True, None


def filter_files(
    files: list[dict[str, Any]],
    search_text: str,
    extension_filter: str,
) -> list[dict[str, Any]]:
    keyword = search_text.strip().lower()
    filtered = files

    if keyword:
        filtered = [
            item
            for item in filtered
            if keyword in str(item.get("original_filename", "")).lower()
        ]

    if extension_filter != "全部":
        filtered = [
            item
            for item in filtered
            if str(item.get("extension", "")).lower() == extension_filter
        ]

    return filtered


def show_flash_message() -> None:
    message = st.session_state.pop("flash_message", None)
    if not message:
        return

    level, text = message
    if level == "success":
        st.success(text)
    elif level == "warning":
        st.warning(text)
    else:
        st.error(text)


def set_flash_message(level: str, text: str) -> None:
    st.session_state["flash_message"] = (level, text)


st.session_state.setdefault("upload_widget_key", 0)
st.session_state.setdefault("pending_delete_id", None)
st.session_state.setdefault("pending_delete_name", "")

files, files_error = fetch_files()

st.title("资料管理后台")
show_flash_message()

with st.sidebar:
    st.subheader("后端服务")
    st.caption("当前 FastAPI 地址")
    st.code(API_URL)
    if files_error:
        st.error("连接异常")
    else:
        st.success("连接正常")

st.subheader("上传资料")
uploaded_files = st.file_uploader(
    "选择文件",
    accept_multiple_files=True,
    key=f"knowledge_upload_{st.session_state['upload_widget_key']}",
)

supported_files: list[Any] = []
unsupported_files: list[str] = []
if uploaded_files:
    preview_rows = []
    for uploaded_file in uploaded_files:
        extension = get_file_extension(uploaded_file.name)
        is_supported = extension in ALLOWED_FILE_TYPES
        if is_supported:
            supported_files.append(uploaded_file)
        else:
            unsupported_files.append(uploaded_file.name)

        preview_rows.append(
            {
                "文件名": uploaded_file.name,
                "类型": extension or "-",
                "大小": format_file_size(get_uploaded_file_size(uploaded_file)),
                "状态": "待上传" if is_supported else "不支持",
            }
        )

    st.dataframe(preview_rows, hide_index=True, use_container_width=True)
    if unsupported_files:
        st.warning("不支持的文件类型不会上传。支持类型：pdf、docx、txt、md。")

upload_disabled = not supported_files
if st.button("上传", type="primary", disabled=upload_disabled):
    ok, success_count, status_code, detail = upload_files(supported_files)
    if ok:
        set_flash_message("success", f"上传成功：{success_count} 个文件")
        st.session_state["upload_widget_key"] += 1
        st.rerun()

    if status_code == 409:
        st.warning(f"文件已存在，无需重复上传。详情：{detail}")
    else:
        st.error(f"上传失败：{detail}")

st.divider()
st.subheader("已上传资料")

if files_error:
    st.error(files_error)
else:
    search_col, filter_col = st.columns([3, 1])
    search_text = search_col.text_input("按文件名搜索", placeholder="输入文件名关键词")
    extension_filter = filter_col.selectbox(
        "按类型筛选",
        ["全部", *ALLOWED_FILE_TYPES],
    )

    filtered_files = filter_files(files, search_text, extension_filter)
    st.caption(f"共 {len(filtered_files)} / {len(files)} 个文件")

    if not filtered_files:
        st.info("暂无匹配资料")
    else:
        header = st.columns([4, 1, 1, 2, 1])
        header[0].markdown("**文件名**")
        header[1].markdown("**类型**")
        header[2].markdown("**大小**")
        header[3].markdown("**上传时间（本地）**")
        header[4].markdown("**操作**")

        for item in filtered_files:
            file_id = str(item.get("file_id", ""))
            filename = str(item.get("original_filename", "-"))
            row = st.columns([4, 1, 1, 2, 1])
            row[0].write(filename)
            row[1].write(item.get("extension", "-"))
            row[2].write(format_file_size(item.get("size")))
            row[3].write(format_uploaded_at(item.get("uploaded_at")))

            if row[4].button("删除", key=f"delete_{file_id}", disabled=not file_id):
                st.session_state["pending_delete_id"] = file_id
                st.session_state["pending_delete_name"] = filename
                st.rerun()

pending_delete_id = st.session_state.get("pending_delete_id")
pending_delete_name = st.session_state.get("pending_delete_name")
if pending_delete_id:
    st.warning(f"确认删除「{pending_delete_name}」？此操作不可撤销。")
    confirm_col, cancel_col = st.columns([1, 5])
    if confirm_col.button("确认删除", type="primary"):
        ok, detail = delete_file(str(pending_delete_id))
        if ok:
            st.session_state["pending_delete_id"] = None
            st.session_state["pending_delete_name"] = ""
            set_flash_message("success", "删除成功")
            st.rerun()

        st.error(f"删除失败：{detail}")

    if cancel_col.button("取消"):
        st.session_state["pending_delete_id"] = None
        st.session_state["pending_delete_name"] = ""
        st.rerun()
