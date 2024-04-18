from query_engines import CustomQueryEngine
from typing import Dict, Any, List

custom_engine = CustomQueryEngine()
simple_engine = custom_engine.create_simple_query_engine(
    "rh-portal-vector-db-dev", "azuresearch", retriever_type="router")
sub_question_engine = custom_engine.create_sub_questions_query_engine()
multi_step_engine = custom_engine.create_multi_step_query_engine(
    "rh-portal-vector-db-dev", "azuresearch", retriever_type="router")

def answer_simple_questions(question) -> Dict[str, Dict[str, Any]]:
    response = simple_engine.query(question)
    res = dict(
        response_text=response.response,
        metadatas=[v for _, v in response.metadata.items()]
    )
    return res

def answer_complex_questions(question) -> Dict[str, Dict[str, Any]]:
    response = sub_question_engine.query(question)
    res = dict(
        response_text=response.response,
        metadatas=[v for _, v in response.metadata.items()]
    )
    return res