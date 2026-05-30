import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DATA_PATH = DATA_DIR / "products.csv"


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    ds = load_dataset("Daoze/MM-Bench-E-Commerce")
    data = ds["test"]

    df = data.to_pandas()

    print("原始数据量：", len(df))
    print("字段：", df.columns.tolist())

    # 筛选服饰时尚
    fashion_df = df[df["doc_industry_name"] == "服饰时尚"].copy()

    print("服饰时尚数据量：", len(fashion_df))

    # 保留适合推荐系统使用的字段
    products = fashion_df[
        [
            "id",
            "doc_title",
            "doc_image",
            "doc_industry_name",
            "doc_cate1_name",
            "doc_cate2_name",
            "doc_cate3_name",
            "doc_cate4_name",
            "doc_attributes",
        ]
    ].copy()

    # 改成更容易理解的字段名
    products = products.rename(
        columns={
            "doc_title": "title",
            "doc_image": "image_url",
            "doc_industry_name": "industry",
            "doc_cate1_name": "category1",
            "doc_cate2_name": "category2",
            "doc_cate3_name": "category3",
            "doc_cate4_name": "category4",
            "doc_attributes": "attributes",
        }
    )

    # 删除没有标题的数据
    products = products.dropna(subset=["title"])

    # 先抽样 8000 条，方便快速实验
    products = products.sample(min(8000, len(products)), random_state=42)

    # 保存成 CSV，utf-8-sig 方便 Excel 打开不乱码
    products.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")

    print(f"已保存：{RAW_DATA_PATH}")
    print(products.head())


if __name__ == "__main__":
    download_data()
