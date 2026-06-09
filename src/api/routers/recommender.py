import logging
import uuid
from typing import Any

from fastapi import APIRouter, Cookie, HTTPException, Response
from pydantic import BaseModel

from src.recommender.graph import create_recommender_graph


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommend", tags=["Recommender"])


class QuestionRequest(BaseModel):
    """
    推荐请求体。
    """

    question: str


# 应用启动时创建 LangGraph app，避免每次请求重复构建图
try:
    graph_app: Any | None = create_recommender_graph()
except Exception:
    logger.exception("初始化推荐图失败")
    graph_app = None


@router.post("/")
def recommend(
    body: QuestionRequest,
    response: Response,
    thread_id: str | None = Cookie(default=None),
) -> dict[str, Any]:
    """
    服装推荐接口。
    """
    if graph_app is None:
        raise HTTPException(status_code=503, detail="推荐服务暂时不可用")

    if not thread_id:
        thread_id = str(uuid.uuid4())
        response.set_cookie(
            key="thread_id",
            value=thread_id,
            httponly=True,
            samesite="lax",
        )

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph_app.invoke(
            {"query": body.question},
            config=config,
        )
        answer = result.get("recommendation", "暂无推荐结果")
        documents = result.get("documents", [])

        return {
            "question": body.question,
            "thread_id": thread_id,
            "answer": answer,
            "documents": documents,
        }
    except Exception as exc:
        logger.exception("推荐流程执行失败")
        raise HTTPException(status_code=500, detail="推荐流程执行失败") from exc
