"""
数据播种脚本 —— 解析 products.csv → 增强字段 → 写入 SQLite。

流程：
1. 读取 products.csv（HuggingFace 下载的原始数据）
2. 解析 attributes 字段 → material / season / style / color / gender
3. 按品类分配价格区间、随机品牌、随机评分/销量
4. 可选：LLM 补充缺失品类（运动、童装、配饰、汉服）
5. 写入 enriched_products.db
"""

import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as settings  # noqa: E402
from src.database.product_repo import ProductRepo  # noqa: E402

logger = logging.getLogger(__name__)

# ── 品牌库（50+ 中文服饰品牌） ────────────────────────

BRANDS_BY_CATEGORY: dict[str, list[str]] = {
    "女装": [
        "伊芙丽", "太平鸟", "欧时力", "ONLY", "VERO MODA", "乐町",
        "茵曼", "裂帛", "韩都衣舍", "秋水伊人", "歌莉娅", "红袖",
        "MO&Co.", "地素", "江南布衣", "妖精的口袋", "衣香丽影",
        "雪中飞", "艾格", "拉夏贝尔", "ZARA", "H&M", "优衣库",
    ],
    "男装": [
        "海澜之家", "七匹狼", "劲霸", "柒牌", "雅戈尔", "报喜鸟",
        "杉杉", "九牧王", "利郎", "罗蒙", "恒源祥", "红豆",
        "GXG", "卡宾", "马克华菲", "杰克琼斯", "SELECTED",
    ],
    "男女鞋": [
        "百丽", "达芙妮", "红蜻蜓", "奥康", "康奈", "意尔康",
        "天美意", "思加图", "卓诗尼", "他她", "斯凯奇", "热风",
        "NIKE", "Adidas", "安踏", "李宁", "特步", "361°",
    ],
    "箱包服配": [
        "金利来", "稻草人", "七匹狼皮具", "皮尔卡丹", "啄木鸟",
        "CHARLES & KEITH", "蔻驰", "MK", "芙拉", "DISSONA",
    ],
    "内衣": [
        "爱慕", "曼妮芬", "古今", "安莉芳", "黛安芬", "华歌尔",
        "都市丽人", "歌瑞尔", "芬狄诗", "维多利亚的秘密",
    ],
    "手表眼镜": [
        "天王", "飞亚达", "罗西尼", "依波", "海鸥",
        "暴龙", "帕莎", "陌森", "雷朋", "木九十",
    ],
    "运动服饰": [
        "NIKE", "Adidas", "安踏", "李宁", "特步",
        "FILA", "彪马", "UNDER ARMOUR", "lululemon", "迪桑特",
    ],
    "童装": [
        "巴拉巴拉", "安奈儿", "小猪班纳", "英氏", "丽婴房",
        "Mini Peace", "gxg kids", "笛莎", "巴布豆", "ABC KIDS",
    ],
    "配饰": [
        "施华洛世奇", "潘多拉", "APM Monaco", "周大福", "周生生",
        "老凤祥", "六福", "I DO", "DR", "卡地亚",
    ],
}

# ── 品类价格区间（元） ────────────────────────────────

PRICE_RANGES: dict[str, tuple[float, float]] = {
    # 女装
    "女士毛衣/针织衫": (99, 599),
    "女士T恤": (39, 199),
    "女士衬衫": (59, 299),
    "女士连衣裙": (99, 499),
    "女士半身裙": (79, 349),
    "女士裤子": (89, 399),
    "女士外套": (199, 899),
    "女士羽绒服": (399, 1999),
    "女士风衣": (299, 999),
    "女士棉衣": (199, 799),
    "女士西装": (199, 699),
    # 男装
    "男士T恤": (49, 249),
    "男士衬衫": (69, 349),
    "男士Polo衫": (79, 349),
    "男士裤子": (99, 449),
    "男士牛仔裤": (129, 499),
    "男士外套": (249, 999),
    "男士羽绒服": (449, 2499),
    "男士西装": (299, 1499),
    "男士毛衣/针织衫": (129, 599),
    # 鞋
    "女鞋": (99, 599),
    "女靴": (199, 899),
    "男鞋": (129, 699),
    "男靴": (249, 999),
    "运动鞋": (199, 999),
    "帆布鞋": (69, 299),
    "凉鞋": (49, 299),
    # 箱包
    "女包": (99, 999),
    "男包": (149, 899),
    "双肩包": (79, 499),
    "钱包/卡包": (49, 349),
    "拉杆箱": (199, 1499),
    # 内衣
    "文胸": (59, 349),
    "内裤": (19, 99),
    "睡衣/家居服": (69, 349),
    "袜子": (9, 49),
    # 配饰
    "围巾": (29, 299),
    "帽子": (29, 199),
    "手套": (19, 149),
    # 手表眼镜
    "手表": (299, 2999),
    "太阳镜": (99, 599),
    "光学镜": (199, 999),
    # 运动
    "运动上衣": (79, 399),
    "运动裤": (99, 449),
    "运动内衣": (59, 299),
    "泳衣": (69, 349),
    # 童装
    "童装上衣": (39, 199),
    "童装裤子": (39, 179),
    "童装裙子": (49, 249),
    "童装外套": (89, 499),
    "婴儿连体衣": (29, 129),
}

# ── 属性解析 ─────────────────────────────────────────

_ATTRIBUTE_PATTERNS: dict[str, re.Pattern] = {
    "人群": re.compile(r"人群[:：]([^,;！]+)"),
    "时间季节": re.compile(r"时间季节[:：]([^,;！]+)"),
    "品类": re.compile(r"品类[:：]([^,;！]+)"),
    "风格": re.compile(r"风格[:：]([^,;！]+)"),
    "品牌": re.compile(r"品牌[:：]([^,;！]+)"),
    "材质": re.compile(r"材质[:：]([^,;！]+)"),
    "颜色": re.compile(r"颜色[:：]([^,;！]+)"),
    "功能功效": re.compile(r"功能功效[:：]([^,;！]+)"),
    "款式元素": re.compile(r"款式元素[:：]([^,;！]+)"),
}

# 季节关键词映射
SEASON_KEYWORDS = {
    "春": "春", "夏": "夏", "秋": "秋", "冬": "冬",
    "春夏": "春夏", "夏秋": "夏秋", "秋冬": "秋冬",
    "春秋": "春秋", "四季": "四季",
}

# 人群 → gender
GENDER_MAP = {
    "女": "女", "男": "男",
    "情侣": "通用", "中性": "通用", "通用": "通用",
    "儿童": "儿童", "女童": "儿童", "男童": "儿童",
    "婴幼儿": "婴幼儿", "婴儿": "婴幼儿",
}


def _extract_attribute(raw: str, key: str) -> str:
    """从原始属性字符串中提取指定键的值。"""
    if not raw or pd.isna(raw):
        return ""
    pattern = _ATTRIBUTE_PATTERNS.get(key)
    if not pattern:
        return ""
    match = pattern.search(str(raw))
    return match.group(1).strip() if match else ""


def _clean_list_value(value: str, sep: str = "!!!") -> str:
    """取列表值的第一项（去除 !! 分隔的多值）。"""
    if not value:
        return ""
    return value.split(sep)[0].strip()


def parse_attributes(raw: str) -> dict[str, str]:
    """
    解析原始属性字符串，返回结构化字段。
    """
    result: dict[str, str] = {}

    raw_material = _extract_attribute(raw, "材质")
    result["material"] = _clean_list_value(raw_material)

    raw_style = _extract_attribute(raw, "风格")
    result["style"] = _clean_list_value(raw_style)

    raw_color = _extract_attribute(raw, "颜色")
    result["color"] = _clean_list_value(raw_color)

    raw_season = _extract_attribute(raw, "时间季节")
    season_clean = raw_season.replace(" ", "").replace(",,", ",").strip(",")
    result["season"] = SEASON_KEYWORDS.get(season_clean, season_clean or "四季")

    raw_gender = _extract_attribute(raw, "人群")
    result["gender"] = GENDER_MAP.get(raw_gender, raw_gender or "通用")

    return result


# ── 品牌 / 价格分配 ───────────────────────────────────


def _get_all_brands() -> list[str]:
    """获取所有品牌的平铺列表。"""
    all_brands: list[str] = []
    for brands in BRANDS_BY_CATEGORY.values():
        all_brands.extend(brands)
    return list(dict.fromkeys(all_brands))  # 去重保序


def _pick_brand(category_l1: str, category_l2: str) -> str:
    """根据类目选择一个合理的品牌。"""
    # 先在 L1 品牌库中找
    pool = BRANDS_BY_CATEGORY.get(category_l1, [])
    if pool:
        return random.choice(pool)

    # 回退到全部品牌
    return random.choice(_get_all_brands())


def _pick_price(category_l2: str, category_l3: str) -> float:
    """根据二级/三级类目生成合理价格。"""
    # 先精确匹配 L3
    for key, (lo, hi) in PRICE_RANGES.items():
        if key in (category_l3 or ""):
            return round(random.uniform(lo, hi), 2)

    # 再模糊匹配 L2
    for key, (lo, hi) in PRICE_RANGES.items():
        if key in (category_l2 or ""):
            return round(random.uniform(lo, hi), 2)

    # 回退
    return round(random.uniform(49, 499), 2)


# ── 内容文本构建 ──────────────────────────────────────


def build_content_text(row: dict[str, Any]) -> str:
    """构建用于检索和嵌入的文本。"""
    parts: list[str] = []

    title = row.get("title", "")
    if title:
        parts.append(f"标题: {title}")

    for field, label in [
        ("category_l1", "一级类目"),
        ("category_l2", "二级类目"),
        ("category_l3", "三级类目"),
    ]:
        val = row.get(field, "")
        if val:
            parts.append(f"{label}: {val}")

    for field, label in [
        ("brand", "品牌"),
        ("material", "材质"),
        ("season", "季节"),
        ("style", "风格"),
        ("color", "颜色"),
    ]:
        val = row.get(field, "")
        if val:
            parts.append(f"{label}: {val}")

    return "\n".join(parts)


# ── 主流程 ────────────────────────────────────────────


def enrich_and_seed(
    products_csv_path: Optional[Path] = None,
    use_llm_for_missing_categories: bool = False,
) -> int:
    """
    读取 CSV → 增强 → 写入 SQLite。

    Args:
        products_csv_path: CSV 路径，默认 config.RAW_DATA_PATH
        use_llm_for_missing_categories: 是否用 LLM 补充缺失品类

    Returns:
        写入的商品总数。
    """
    if products_csv_path is None:
        products_csv_path = settings.RAW_DATA_PATH

    if not products_csv_path.exists():
        raise FileNotFoundError(f"原始商品 CSV 不存在: {products_csv_path}")

    logger.info("读取原始商品数据: %s", products_csv_path)
    df = pd.read_csv(products_csv_path, encoding="utf-8-sig")

    # 重命名列（英→中→统一）
    rename_map = {
        "title": "商品标题",
        "image_url": "商品图片",
        "industry": "商品领域",
        "category1": "商品大类",
        "category2": "商品类别",
        "category3": "商品子类",
        "category4": "商品细分类",
        "attributes": "商品属性",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    repo = ProductRepo()
    total_before = repo.count()

    products: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        meta = row.to_dict()
        product_id = str(meta.get("id", ""))

        # 解析属性
        attr_raw = str(meta.get("商品属性", ""))
        parsed = parse_attributes(attr_raw)

        # 类目
        cat_l1 = str(meta.get("商品大类", "") or meta.get("category1", ""))
        cat_l2 = str(meta.get("商品类别", "") or meta.get("category2", ""))
        cat_l3 = str(meta.get("商品子类", "") or meta.get("category3", ""))
        cat_l4 = str(meta.get("商品细分类", "") or meta.get("category4", ""))

        # 赋值
        title = str(meta.get("商品标题", ""))
        brand = parsed.get("brand") or _pick_brand(cat_l1, cat_l2)
        material = parsed.get("material", "")
        season = parsed.get("season", "四季")
        style = parsed.get("style", "")
        color = parsed.get("color", "")
        gender = parsed.get("gender", "通用")

        product = {
            "id": product_id,
            "title": title,
            "image_url": str(meta.get("商品图片", "")),
            "category_l1": cat_l1,
            "category_l2": cat_l2,
            "category_l3": cat_l3,
            "category_l4": cat_l4,
            "price": _pick_price(cat_l2, cat_l3),
            "brand": brand,
            "material": material,
            "season": season,
            "style": style,
            "color": color,
            "gender": gender,
            "rating": round(random.uniform(3.5, 5.0), 1),
            "sales_count": int(random.expovariate(1 / 500) + 10),
            "stock_status": random.choices(
                ["有货", "预售", "售罄"], weights=[0.80, 0.15, 0.05]
            )[0],
            "attributes_raw": attr_raw,
            "content_text": "",  # 下面填充
        }

        product["content_text"] = build_content_text(product)
        products.append(product)

    # 批量写入
    count = repo.insert_batch(products)
    total_after = repo.count()

    logger.info("增强完成：写入 %s 条，数据库共 %s 条", count, total_after)
    logger.info("一级类目分布：")
    for key in sorted(repo.get_categories()):
        if key.startswith("category_l1:"):
            logger.info("  %s: %s", key.split(":", 1)[1], repo.get_categories()[key])

    # 价格分布
    price_stats = repo.get_price_stats()
    logger.info(
        "价格分布: min=¥%.0f max=¥%.0f avg=¥%.0f",
        price_stats.get("min", 0),
        price_stats.get("max", 0),
        price_stats.get("avg", 0),
    )

    # 可选：LLM 补充缺失品类
    if use_llm_for_missing_categories:
        _generate_missing_categories(repo)

    return total_after


def _generate_missing_categories(repo: ProductRepo) -> None:
    """
    用 LLM 生成缺失品类的商品数据。

    覆盖：运动服饰、童装、配饰首饰、汉服/国风
    """
    logger.info("使用 LLM 补充缺失品类 ...")
    # TODO: 用 qwen-plus 批量生成每类 100 条
    # 当前先跳过，用户可手动触发
    logger.warning("LLM 补充暂未实现，跳过")


# ── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    enrich_and_seed()
