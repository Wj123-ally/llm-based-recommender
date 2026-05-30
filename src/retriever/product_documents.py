import re
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from langchain_core.documents import Document


PRODUCT_TEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("商品标题", "标题"),
    ("商品大类", "一级类目"),
    ("商品类别", "二级类目"),
    ("商品子类", "三级类目"),
    ("商品细分类", "四级类目"),
    ("商品属性", "属性"),
)

_ASCII_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")


def _clean_value(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""

    return text


def build_product_content(row: Mapping[str, Any]) -> str:
    parts: list[str] = []

    for field_name, label in PRODUCT_TEXT_FIELDS:
        value = _clean_value(row.get(field_name))
        if value:
            parts.append(f"{label}: {value}")

    return "\n".join(parts)


def get_document_key(document: "Document") -> str:
    metadata = document.metadata or {}

    for field_name in ("id", "商品标题"):
        value = _clean_value(metadata.get(field_name))
        if value:
            return f"{field_name}:{value}"

    return f"content:{document.page_content}"


def chinese_tokenize(text: str) -> list[str]:
    normalized = _clean_value(text).lower()
    if not normalized:
        return []

    try:
        import jieba

        tokens = [token.strip() for token in jieba.lcut(normalized) if token.strip()]
    except ImportError:
        tokens = _fallback_chinese_tokenize(normalized)

    return [token for token in tokens if token]


def _fallback_chinese_tokenize(text: str) -> list[str]:
    tokens: list[str] = []

    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if _ASCII_WORD_RE.fullmatch(token):
            tokens.append(token)
            continue

        chars = _CJK_CHAR_RE.findall(token)
        tokens.extend(chars)
        tokens.extend(
            "".join(chars[index : index + 2])
            for index in range(max(len(chars) - 1, 0))
        )

    return tokens
