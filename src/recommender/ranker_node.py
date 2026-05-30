from src.recommender.self_query_node import self_query_retrieve
from src.recommender.state import RecState


def ranker_node(state: RecState) -> RecState:
    return self_query_retrieve(state)
