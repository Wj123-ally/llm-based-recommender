from src.database.product_repo import ProductRepo
repo = ProductRepo()
products = repo.find_all(limit=1)
if products:
    print('商品字段示例:')
    for k, v in products[0].items():
        print(f'  {k}: {v}')
