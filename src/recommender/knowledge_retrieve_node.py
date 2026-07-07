"""
知识库检索节点模块。

从 graph.py 提取，独立为一个模块以便并行检索时复用。
"""

import logging

from src.recommender.state import RecState

logger = logging.getLogger(__name__)


def knowledge_retrieve_node(state: RecState) -> RecState:
    """
    LangGraph 知识库检索节点。

    从用户上传的知识库文件中检索与 query 相关的文档片段。
    如果知识库为空或检索失败，不影响后续流程。
    """
    query = state["query"]

    logger.info("-" * 50)
    logger.info("[知识检索] 开始检索知识库")
    logger.info("[知识检索] 查询语句: %s", query)

    try:
        from src.knowledge_base.knowledge_retriever import retrieve_knowledge

        knowledge_docs = retrieve_knowledge(state["query"])

        if knowledge_docs:
            state["knowledge_docs"] = knowledge_docs
            state["knowledge_retrieval_state"] = "success"
            logger.info("[知识检索] 状态: 成功，找到相关文档片段")
            logger.info("[知识检索] 内容预览: %s...", knowledge_docs[:200])
        else:
            state["knowledge_docs"] = ""
            state["knowledge_retrieval_state"] = "empty"
            logger.info("[知识检索] 状态: 知识库为空或无匹配内容")

    except ImportError:
        logger.warning("[知识检索] 模块不可用，跳过")
        state["knowledge_docs"] = ""
        state["knowledge_retrieval_state"] = "skipped"
    except Exception:
        logger.exception("[知识检索] 检索异常，跳过")
        state["knowledge_docs"] = ""
        state["knowledge_retrieval_state"] = "skipped"

    logger.info("-" * 50)
    return state
