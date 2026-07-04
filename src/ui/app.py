from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import requests
import streamlit as st


st.set_page_config(
    page_title="AI 鞋类导购",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
MAX_PRODUCTS = int(os.getenv("MAX_PRODUCT_IMAGES", "6"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY_RECORDS", "30"))
MEMORY_PATH = Path(
    os.getenv(
        "CHAT_MEMORY_PATH",
        str(Path(__file__).resolve().parent / "chat_memory.json"),
    )
).expanduser()

PRODUCT_ATTRS = [
    ("shoe_type", "鞋型"),
    ("color", "颜色"),
    ("material", "材质"),
    ("season", "季节"),
    ("brand", "品牌"),
    ("gender", "人群"),
    ("usage_scene", "场景"),
    ("heel_type", "鞋跟"),
    ("functionality", "特点"),
]

WELCOME_MESSAGE = (
    "您好，我是您的专属鞋类导购顾问，拥有丰富的产品知识和导购经验！\n\n"
    "我可以根据您的使用场景、搭配需求、颜色偏好和脚感要求进行针对性推荐，"
    "帮您挑选最合适的鞋类产品。\n\n"
    "无论您是想了解产品对比、鞋类搭配技巧、材质特点，还是想找通勤、运动、休闲、"
    "配裙子等具体场景的鞋，都可以随时向我提问。"
)


def welcome_message() -> dict[str, Any]:
    return {"role": "assistant", "content": WELCOME_MESSAGE, "documents": []}


CSS = """
<style>
    :root {
        --header-h: 78px;
        --max-w: 1180px;
        --user-bg: #E3F2FD;
        --user-border: #B8DCF5;
        --ai-bg: #F3EEFF;
        --ai-border: #DDD2FF;
        --ink: #182033;
        --muted: #687083;
        --brand: #6F63D9;
    }

    .main .block-container {
        max-width: var(--max-w);
        padding-top: 0.8rem;
        padding-bottom: 7rem;
    }

    div[data-testid="stVerticalBlock"] { gap: 0.75rem; }

    .app-header {
        position: sticky;
        top: 0;
        z-index: 20;
        background: rgba(255, 255, 255, 0.97);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid #E9ECF3;
        padding: 14px 0 12px;
        margin: -0.8rem 0 18px;
    }

    .app-title {
        margin: 0;
        color: var(--ink);
        font-size: 1.35rem;
        line-height: 1.2;
        font-weight: 850;
        letter-spacing: 0;
    }

    .app-subtitle {
        margin-top: 4px;
        color: var(--muted);
        font-size: 0.92rem;
    }

    .chat-row {
        display: flex;
        width: 100%;
        margin: 16px 0;
        align-items: flex-start;
        gap: 10px;
    }

    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }

    .avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 34px;
        font-size: 12px;
        font-weight: 800;
    }

    .avatar-ai {
        color: var(--brand);
        background: #FFFFFF;
        border: 1px solid #DDDFF0;
    }

    .avatar-user {
        color: #1E6BAA;
        background: #DDF0FF;
        border: 1px solid #B8DCF5;
    }

    .bubble {
        max-width: min(86%, 980px);
        border-radius: 18px;
        padding: 13px 16px;
        box-shadow: 0 3px 14px rgba(17, 24, 39, 0.05);
        word-break: break-word;
        overflow-wrap: anywhere;
    }

    .bubble-user {
        background: var(--user-bg);
        border: 1px solid var(--user-border);
        border-bottom-right-radius: 6px;
        max-width: min(72%, 760px);
    }

    .bubble-ai {
        background: var(--ai-bg);
        border: 1px solid var(--ai-border);
        border-bottom-left-radius: 6px;
    }

    .speaker {
        font-size: 0.78rem;
        font-weight: 800;
        color: var(--brand);
        margin-bottom: 7px;
    }

    .bubble-user .speaker { color: #1E6BAA; }

    .message-text {
        color: var(--ink);
        line-height: 1.7;
        font-size: 0.95rem;
        white-space: normal;
    }

    .product-section-title {
        margin: 14px 0 10px;
        color: #3F3A75;
        font-weight: 850;
        font-size: 0.95rem;
    }

    .product-list {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 18px;
        margin-top: 10px;
    }

    .product-entry {
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
    }

    .guide-product-card {
        width: min(100%, 380px);
        background: #FFFFFF;
        border: 1px solid #E4E7F1;
        border-radius: 8px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }

    .guide-product-image {
        width: 100%;
        aspect-ratio: 1 / 1;
        background: #F7F8FB;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }

    .guide-product-image img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        object-position: center;
        display: block;
    }

    .product-placeholder {
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #8A8FA3;
        font-size: 0.9rem;
    }

    .product-body {
        display: flex;
        flex-direction: column;
        gap: 8px;
        min-height: 0;
        flex: 1;
        padding: 10px 12px 12px;
    }

    .product-title {
        color: var(--ink);
        font-weight: 800;
        font-size: 0.94rem;
        line-height: 1.55;
        overflow-wrap: anywhere;
    }

    .product-reason {
        width: 100%;
        box-sizing: border-box;
        color: #4338A3;
        background: #F4F0FF;
        border: 1px solid #E4DCFF;
        border-radius: 6px;
        padding: 10px 11px;
        font-size: 0.88rem;
        line-height: 1.65;
        overflow-wrap: anywhere;
    }

    .product-details {
        border-top: 1px solid #EEF0F6;
        padding-top: 6px;
    }

    .product-details summary {
        width: fit-content;
        cursor: pointer;
        user-select: none;
        color: #3F3A75;
        background: #F7F5FF;
        border: 1px solid #E3DFFF;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 0.82rem;
        font-weight: 800;
    }

    .product-meta {
        color: #5C6475;
        font-size: 0.82rem;
        line-height: 1.6;
        margin-top: 8px;
        overflow: visible;
    }

    .product-meta-row {
        margin: 3px 0;
        overflow-wrap: anywhere;
    }

    section[data-testid="stSidebar"] .stButton button {
        border-radius: 8px;
        text-align: left;
        min-height: 38px;
    }

    div[data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.98);
        border-top: 1px solid #E8EBF2;
        padding-top: 10px;
    }

    @media (max-width: 1080px) {
        .bubble { max-width: 92%; }
        .bubble-user { max-width: 82%; }
    }

    @media (max-width: 720px) {
        .bubble, .bubble-user { max-width: 92%; }
        .avatar { display: none; }
        .product-entry { width: 100%; }
    }
</style>
"""


def _e(value: Any) -> str:
    return escape(str(value or ""), quote=True)


def _html_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"&lt;/?div[^&]*&gt;", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?div[^>]*>", "", text, flags=re.IGNORECASE)
    text = text.replace("</div>", "").replace("<div>", "")
    return _e(text).replace("\n", "<br>")


def _metadata_value(metadata: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return _clean_display_value(value)
    return default


def _clean_display_value(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        items = re.findall(r"['\"]([^'\"]+)['\"]", text)
        if items:
            return "、".join(item for item in items if item)
    return text


def _api(path: str) -> str:
    return f"{API_URL.rstrip('/')}/{path.lstrip('/')}"


def _session() -> requests.Session:
    if "api_session" not in st.session_state:
        st.session_state.api_session = requests.Session()
    return st.session_state.api_session


def _set_thread_cookie(thread_id: str | None) -> None:
    session = _session()
    session.cookies.clear()
    if thread_id:
        session.cookies.set("thread_id", thread_id)


def call_api(question: str) -> dict[str, Any]:
    if st.session_state.get("thread_id"):
        _set_thread_cookie(st.session_state.thread_id)

    try:
        response = _session().post(
            _api("/recommend/"),
            json={"question": question},
            timeout=120,
        )
    except requests.exceptions.RequestException:
        return {
            "answer": "无法连接后端服务，请确认 FastAPI 已启动。",
            "documents": [],
            "thread_id": st.session_state.get("thread_id"),
        }

    if response.status_code != 200:
        return {
            "answer": f"服务异常：HTTP {response.status_code}",
            "documents": [],
            "thread_id": st.session_state.get("thread_id"),
        }

    try:
        data = response.json()
    except ValueError:
        return {
            "answer": "服务返回的数据格式异常。",
            "documents": [],
            "thread_id": st.session_state.get("thread_id"),
        }

    return {
        "answer": data.get("answer", ""),
        "documents": data.get("documents", []),
        "thread_id": data.get("thread_id", st.session_state.get("thread_id")),
    }


def load_records() -> list[dict[str, Any]]:
    if not MEMORY_PATH.exists():
        return []
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [
        item
        for item in data
        if isinstance(item, dict) and isinstance(item.get("messages"), list)
    ]


def save_records(records: list[dict[str, Any]]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(
        json.dumps(records[:MAX_HISTORY], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def history_title(question: str) -> str:
    title = re.sub(r"\s+", " ", question).strip() or "新对话"
    return title if len(title) <= 22 else f"{title[:22]}..."


def init_state() -> None:
    if "records" not in st.session_state:
        st.session_state.records = load_records()
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "messages" not in st.session_state:
        if st.session_state.records:
            record = st.session_state.records[0]
            st.session_state.messages = record.get("messages", [])
            st.session_state.conversation_id = record.get("id")
            st.session_state.thread_id = record.get("thread_id")
            _set_thread_cookie(st.session_state.thread_id)
        else:
            st.session_state.messages = [welcome_message()]
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def persist_conversation(last_question: str) -> None:
    if not st.session_state.messages:
        return

    conversation_id = st.session_state.conversation_id or str(uuid.uuid4())
    st.session_state.conversation_id = conversation_id
    records = [
        record
        for record in st.session_state.records
        if record.get("id") != conversation_id
    ]
    records.insert(
        0,
        {
            "id": conversation_id,
            "title": history_title(last_question),
            "updated_at": datetime.now().strftime("%m-%d %H:%M"),
            "thread_id": st.session_state.thread_id,
            "messages": st.session_state.messages,
        },
    )
    st.session_state.records = records[:MAX_HISTORY]
    save_records(st.session_state.records)


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">AI 鞋类导购</h1>
            <div class="app-subtitle">根据你的场景、风格和颜色偏好推荐合适的鞋类商品</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("历史对话")
        st.caption("侧栏可通过左上角箭头折叠")

        col_new, col_clear = st.columns(2)
        if col_new.button("新对话", use_container_width=True):
            st.session_state.messages = [welcome_message()]
            st.session_state.conversation_id = None
            st.session_state.thread_id = None
            _set_thread_cookie(None)
            st.rerun()

        if col_clear.button("清空", use_container_width=True):
            st.session_state.records = []
            st.session_state.messages = [welcome_message()]
            st.session_state.conversation_id = None
            st.session_state.thread_id = None
            _set_thread_cookie(None)
            try:
                MEMORY_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            st.rerun()

        st.divider()
        if not st.session_state.records:
            st.caption("暂无历史记录")
            return

        for record in st.session_state.records:
            label = record.get("title", "未命名对话")
            updated_at = record.get("updated_at", "")
            if updated_at:
                label = f"{label}\n{updated_at}"
            if st.button(label, key=f"history_{record.get('id')}", use_container_width=True):
                st.session_state.messages = record.get("messages", [])
                st.session_state.conversation_id = record.get("id")
                st.session_state.thread_id = record.get("thread_id")
                _set_thread_cookie(st.session_state.thread_id)
                st.rerun()


def render_user_bubble(content: str) -> None:
    st.markdown(
        f"""
        <div class="chat-row user">
            <div class="bubble bubble-user">
                <div class="speaker">你</div>
                <div class="message-text">{_html_text(content)}</div>
            </div>
            <div class="avatar avatar-user">你</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pick_product_docs(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document in documents:
        metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
        product_id = _metadata_value(metadata, ["id", "product_id"])
        image_url = _metadata_value(metadata, ["image_url", "商品图片"])
        key = product_id or image_url
        if not key or key in seen:
            continue
        seen.add(key)
        products.append(document)
        if len(products) >= MAX_PRODUCTS:
            break
    return products


def extract_recommend_reason(title: str, answer: str, metadata: dict[str, Any]) -> str:
    scene = _metadata_value(metadata, ["usage_scene"])
    shoe_type = _metadata_value(metadata, ["shoe_type"])
    color = _metadata_value(metadata, ["color", "text_color", "image_color"])
    material = _metadata_value(metadata, ["material"])
    style = _metadata_value(metadata, ["style"])
    parts = [value for value in [color, scene, shoe_type] if value]
    if parts:
        base = " / ".join(parts)
        extra = "，".join(value for value in [material, style] if value)
        return (
            f"这款{title}是我们店里很适合优先看的款式，{base}这些特点和您的需求比较贴合。"
            + (f" 同时它采用{extra}，日常搭配和上脚质感都会更稳妥。" if extra else "")
        )
    return f"这款{title}是我们为您优先挑出的候选商品，整体风格和使用场景都比较百搭，适合继续重点了解。"


def split_answer_for_products(answer: str) -> tuple[str, list[str]]:
    text = str(answer or "").strip()
    if not text:
        return "", []

    # 优先匹配 "商品N：" 格式（LLM生成的格式）
    product_matches = list(re.finditer(r"(?m)^\s*商品\s*[0-9]+\s*[：:]\s*", text))
    if product_matches:
        intro = text[: product_matches[0].start()].strip()
        sections = []
        for index, match in enumerate(product_matches):
            start = match.end()
            end = product_matches[index + 1].start() if index + 1 < len(product_matches) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)
        return intro, sections

    # 回退：匹配数字编号格式 "1." "2."
    number_matches = list(re.finditer(r"(?m)^\s*\d+[\.、]\s*", text))
    if number_matches:
        intro = text[: number_matches[0].start()].strip()
        sections = []
        for index, match in enumerate(number_matches):
            start = match.end()
            end = number_matches[index + 1].start() if index + 1 < len(number_matches) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)
        return intro, sections

    # 没有找到任何格式，返回全文作为intro
    return text, []


def section_source_rank(section: str) -> int | None:
    match = re.search(r"商品\s*(\d+)\s*[:：]", str(section or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def product_source_rank(document: dict[str, Any]) -> int | None:
    metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
    value = metadata.get("_source_rank") if isinstance(metadata, dict) else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_product_reason(title: str, reason: str) -> str:
    text = str(reason or "").strip()
    if not text:
        return ""

    escaped_title = re.escape(title.strip())
    if escaped_title:
        text = re.sub(rf"^{escaped_title}\s*", "", text).strip()
        text = re.sub(rf"^{escaped_title}\s*", "", text).strip()
    return text


def build_product_item(document: dict[str, Any], answer: str, reason: str = "") -> str:
    metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
    image_url = _metadata_value(metadata, ["image_url", "商品图片"])
    title = _metadata_value(metadata, ["title", "商品标题"], "未命名商品")
    reason = clean_product_reason(title, reason) or extract_recommend_reason(
        title, answer, metadata
    )

    details = []
    for key, label in PRODUCT_ATTRS:
        value = _metadata_value(metadata, [key])
        if value:
            details.append(
                f'<div class="product-meta-row"><b>{_e(label)}:</b> {_e(value)}</div>'
            )
    detail_html = "".join(details) or '<div class="product-meta-row">暂无更多属性</div>'

    if image_url:
        image_html = f'<div class="guide-product-image"><img src="{_e(image_url)}" alt="{_e(title)}"></div>'
    else:
        image_html = '<div class="guide-product-image"><div class="product-placeholder">暂无图片</div></div>'

    details_html = (
        '<details class="product-details">'
        '<summary>商品详情</summary>'
        f'<div class="product-meta">{detail_html}</div>'
        "</details>"
        if details
        else ""
    )

    return (
        '<div class="product-entry">'
        '<div class="guide-product-card">'
        f"{image_html}"
        '<div class="product-body">'
        f'<div class="product-title">{_e(title)}</div>'
        f"{details_html}"
        "</div>"
        "</div>"
        f'<div class="product-reason">{_e(reason)}</div>'
        "</div>"
    )


def build_product_grid(documents: list[dict[str, Any]], answer: str) -> str:
    products = pick_product_docs(documents)
    if not products:
        return ""

    _, sections = split_answer_for_products(answer)
    items_to_render: list[tuple[dict[str, Any], str]] = []
    if sections:
        products_by_source_rank = {
            rank: product
            for product in products
            if (rank := product_source_rank(product)) is not None
        }
        used_keys: set[str] = set()
        for index, section in enumerate(sections):
            source_rank = section_source_rank(section)
            product = (
                products_by_source_rank.get(source_rank)
                if source_rank is not None
                else products[index] if index < len(products) else None
            )
            if product is None:
                continue
            metadata = product.get("metadata", {}) if isinstance(product, dict) else {}
            key = str(metadata.get("id") or metadata.get("product_id") or source_rank or index)
            if key in used_keys:
                continue
            used_keys.add(key)
            items_to_render.append((product, section))
    else:
        items_to_render = [(product, "") for product in products]

    items = "".join(
        build_product_item(document, answer, reason)
        for document, reason in items_to_render
    )
    return (
        '<div class="product-section-title">推荐商品</div>'
        f'<div class="product-list">{items}</div>'
    )


def render_ai_bubble(answer: str, documents: list[dict[str, Any]]) -> None:
    intro, _ = split_answer_for_products(answer)
    product_grid = build_product_grid(documents, answer)
    if product_grid:
        answer_block = f'<div class="message-text">{_html_text(intro)}</div>' if intro else ""
    else:
        answer_html = _html_text(answer) if answer else "暂时没有生成推荐说明。"
        answer_block = f'<div class="message-text">{answer_html}</div>'
    html = (
        '<div class="chat-row assistant">'
        '<div class="avatar avatar-ai">AI</div>'
        '<div class="bubble bubble-ai">'
        '<div class="speaker">AI 导购</div>'
        f"{answer_block}"
        f"{product_grid}"
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def request_answer(question: str) -> None:
    placeholder = st.empty()

    try:
        placeholder.markdown("正在为您挑选合适商品...")
        result = call_api(question)
        answer = result.get("answer", "")
        documents = result.get("documents", [])
        st.session_state.thread_id = result.get(
            "thread_id",
            st.session_state.get("thread_id"),
        )
        placeholder.empty()
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer or "抱歉，暂时没有生成推荐结果。",
                "documents": documents,
            }
        )

    except Exception as e:
        placeholder.empty()
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"抱歉，发生了意外错误：{str(e)}",
                "documents": [],
            }
        )

    persist_conversation(question)
    st.session_state.pending_question = None


init_state()
st.markdown(CSS, unsafe_allow_html=True)
render_sidebar()
render_header()

for message in st.session_state.messages:
    if message.get("role") == "assistant":
        render_ai_bubble(
            message.get("content", ""),
            message.get("documents", []),
        )
    else:
        render_user_bubble(message.get("content", ""))

prompt = st.chat_input("描述你想要的鞋类商品，例如：黑色通勤皮鞋、夏季防滑凉拖")
if prompt and prompt.strip():
    question = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.pending_question = question
    st.rerun()

if st.session_state.pending_question:
    request_answer(st.session_state.pending_question)
    st.rerun()
