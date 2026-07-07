from fastapi import FastAPI

from src.api.routers.knowledge_base import router as knowledge_base_router
from src.api.routers.recommender import router as recommender_router


app = FastAPI(
    title="AI Shoe Recommender API",
    version="1.0",
)

# 注册推荐接口路由
app.include_router(recommender_router)
app.include_router(knowledge_base_router)


@app.get("/")
def root() -> dict[str, str]:
    """
    根路径欢迎信息。
    """
    return {"message": "Welcome to AI Shoe Recommender API"}


@app.get("/health")
def health() -> dict[str, str]:
    """
    健康检查。
    """
    return {"status": "healthy"}
