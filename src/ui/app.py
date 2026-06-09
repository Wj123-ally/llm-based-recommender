import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")
PRODUCT_IMAGE_WIDTH = int(os.getenv("PRODUCT_IMAGE_WIDTH", "220"))
MAX_PRODUCT_IMAGES = int(os.getenv("MAX_PRODUCT_IMAGES", "5"))
MAX_HISTORY_RECORDS = int(os.getenv("MAX_HISTORY_RECORDS", "30"))
MEMORY_PATH = Path(
    os.getenv(
        "CHAT_MEMORY_PATH",
        str(Path(__file__).resolve().parent / "chat_memory.json"),
    )
).expanduser()
NUMBERED_SECTION_RE = re.compile(r"(?m)(?=^\s*\d+\.\s+)")


st.set_page_config(
    page_title="服装智能推荐",
    page_icon=":dress:",
    layout="centered",
)


def build_api_url(path: str) -> str:
    return f"{API_URL.rstrip('/')}/{path.lstrip('/')}"


def get_api_session() -> requests.Session:
    if "api_session" not in st.session_state:
        st.session_state.api_session = requests.Session()

    return st.session_state.api_session


def load_history_records() -> list[dict[str, Any]]:
    if not MEMORY_PATH.exists():
        return []

    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []

    return [
        record
        for record in data
        if isinstance(record, dict) and isinstance(record.get("messages"), list)
    ]


def save_history_records(records: list[dict[str, Any]]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(
        json.dumps(records[:MAX_HISTORY_RECORDS], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_history_title(question: str) -> str:
    title = re.sub(r"\s+", " ", question).strip()
    if not title:
        return "未命名对话"

    return title if len(title) <= 28 else f"{title[:28]}..."


def set_api_thread_cookie(thread_id: str | None) -> None:
    session = get_api_session()
    session.cookies.clear()

    if thread_id:
        session.cookies.set("thread_id", thread_id)


def init_memory_state() -> None:
    if "history_records" not in st.session_state:
        st.session_state.history_records = load_history_records()

    if "current_history_id" not in st.session_state:
        st.session_state.current_history_id = None

    if "current_thread_id" not in st.session_state:
        st.session_state.current_thread_id = None

    if "messages" not in st.session_state:
        if st.session_state.history_records:
            latest = st.session_state.history_records[0]
            st.session_state.messages = latest.get("messages", [])
            st.session_state.current_history_id = latest.get("id")
            st.session_state.current_thread_id = latest.get("thread_id")
            set_api_thread_cookie(st.session_state.current_thread_id)
        else:
            st.session_state.messages = []
            st.session_state.current_history_id = None
            st.session_state.current_thread_id = None
    elif st.session_state.current_thread_id:
        set_api_thread_cookie(st.session_state.current_thread_id)


def remember_current_chat(question: str) -> None:
    if not st.session_state.messages:
        return

    history_id = st.session_state.current_history_id or str(uuid.uuid4())
    st.session_state.current_history_id = history_id

    records = [
        record
        for record in st.session_state.history_records
        if record.get("id") != history_id
    ]
    records.insert(
        0,
        {
            "id": history_id,
            "title": format_history_title(question),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "thread_id": st.session_state.current_thread_id,
            "messages": st.session_state.messages,
        },
    )

    st.session_state.history_records = records[:MAX_HISTORY_RECORDS]
    save_history_records(st.session_state.history_records)


def load_history_record(record: dict[str, Any]) -> None:
    st.session_state.messages = record.get("messages", [])
    st.session_state.current_history_id = record.get("id")
    st.session_state.current_thread_id = record.get("thread_id")
    set_api_thread_cookie(st.session_state.current_thread_id)
    st.rerun()


def start_new_chat() -> None:
    st.session_state.messages = []
    st.session_state.current_history_id = None
    st.session_state.current_thread_id = None
    set_api_thread_cookie(None)
    st.rerun()


def clear_long_term_memory() -> None:
    st.session_state.history_records = []
    st.session_state.messages = []
    st.session_state.current_history_id = None
    st.session_state.current_thread_id = None
    set_api_thread_cookie(None)

    try:
        MEMORY_PATH.unlink(missing_ok=True)
    except OSError:
        pass

    st.rerun()


def call_recommend_api(question: str) -> dict[str, Any]:
    if st.session_state.current_thread_id:
        set_api_thread_cookie(st.session_state.current_thread_id)

    try:
        response = get_api_session().post(
            build_api_url("/recommend/"),
            json={"question": question},
            timeout=120,
        )
    except requests.exceptions.RequestException:
        return {
            "answer": "无法连接后端服务，请确认 FastAPI 已启动。",
            "documents": [],
            "thread_id": st.session_state.current_thread_id,
        }

    if response.status_code != 200:
        return {
            "answer": "后端返回错误，请稍后再试。",
            "documents": [],
            "thread_id": st.session_state.current_thread_id,
        }

    try:
        data = response.json()
    except ValueError:
        return {
            "answer": "后端返回格式异常，请稍后再试。",
            "documents": [],
            "thread_id": st.session_state.current_thread_id,
        }

    return {
        "answer": data.get("answer", "暂无推荐结果"),
        "documents": data.get("documents", []),
        "thread_id": data.get("thread_id", st.session_state.current_thread_id),
    }


def metadata_value(metadata: dict[str, Any], key: str, default: str = "") -> str:
    value = metadata.get(key, default)
    if value is None:
        return default

    return str(value)


def get_image_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_documents: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for document in documents:
        metadata = document.get("metadata", {})
        image_url = metadata_value(metadata, "商品图片")
        if not image_url or image_url in seen_urls:
            continue

        seen_urls.add(image_url)
        image_documents.append(document)
        if len(image_documents) >= MAX_PRODUCT_IMAGES:
            break

    return image_documents


def split_answer_sections(answer: str) -> tuple[str, list[str]]:
    chunks = [
        chunk.strip()
        for chunk in NUMBERED_SECTION_RE.split(answer.strip())
        if chunk.strip()
    ]
    intro_parts: list[str] = []
    sections: list[str] = []

    for chunk in chunks:
        if re.match(r"^\s*\d+\.\s+", chunk):
            sections.append(chunk)
        else:
            intro_parts.append(chunk)

    if not sections:
        return "", [answer]

    return "\n\n".join(intro_parts), sections


def render_product_image(document: dict[str, Any]) -> None:
    metadata = document.get("metadata", {})
    image_url = metadata_value(metadata, "商品图片")
    if not image_url:
        return

    title = metadata_value(metadata, "商品标题", "未命名商品")
    categories = [
        metadata_value(metadata, "商品大类"),
        metadata_value(metadata, "商品类别"),
        metadata_value(metadata, "商品子类"),
    ]
    category_text = " / ".join(item for item in categories if item)

    st.image(image_url, caption=title, width=PRODUCT_IMAGE_WIDTH)
    if category_text:
        st.caption(category_text)


def render_assistant_response(
    answer: str,
    documents: list[dict[str, Any]],
) -> None:
    image_documents = get_image_documents(documents)
    intro, sections = split_answer_sections(answer)

    if intro:
        st.write(intro)

    for index, section in enumerate(sections):
        st.write(section)
        if index < len(image_documents):
            render_product_image(image_documents[index])


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("设置")
        st.write("当前后端地址：")
        st.code(API_URL)

        if st.button("新建对话", use_container_width=True):
            start_new_chat()

        if st.button("清空长期记忆", use_container_width=True):
            clear_long_term_memory()

        st.subheader("历史记录")
        if not st.session_state.history_records:
            st.caption("暂无历史记录")
            return

        for record in st.session_state.history_records:
            title = record.get("title") or "未命名对话"
            updated_at = record.get("updated_at", "")
            label = f"{title}\n{updated_at}" if updated_at else title
            if st.button(
                label,
                key=f"history_{record.get('id')}",
                use_container_width=True,
            ):
                load_history_record(record)


def send_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("正在为你挑选合适商品..."):
            result = call_recommend_api(question)
            answer = result["answer"]
            documents = result.get("documents", [])
            st.session_state.current_thread_id = result.get("thread_id")
            render_assistant_response(answer, documents)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "documents": documents,
        }
    )
    remember_current_chat(question)


init_memory_state()

st.title("服装智能推荐")
st.write("输入你的穿搭需求，我会根据商品库为你推荐合适的服装。")

render_sidebar()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_response(
                message["content"],
                message.get("documents", []),
            )
        else:
            st.write(message["content"])


user_question = st.chat_input("请输入你的穿搭需求")

if user_question:
    send_question(user_question)
