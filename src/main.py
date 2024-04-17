from typing import Union, List, Tuple, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from api.data_ingestion import greedy_ingest_data_from_urls, ingest_data_from_urls
from api.report_writing import Impax10StepWriter
from api.info_retrieval import answer_complex_questions, answer_simple_questions
# from utils import DBConfig
from typing import List, Dict, Any, Literal
import os


app = FastAPI()


@app.get("/")
def root():
    return {"intro": 
            "Welcome to Impax's AI assistant API.\n" 
            "This AI is built with langchain, langgrah, and Azure OpenAI.",
            "version":
            "alpha v0.2\n"}


#########################################
##### constructs for the api inputs #####
#########################################
class UrlLoaderContainer(BaseModel):
    urls: list
    db_name: str
    adddtional_metadata: Dict[str, Any]

class GreedyUrlLoaderContainer(BaseModel):
    urls: list
    db_name: str
    depth: int
    adddtional_metadata: Dict[str, Any]
    browser: Literal["selenium", "requests"]
class InfoRetrieverParamsContainer(BaseModel):
    query: str
    recursion_limit: int
    db_names: List[str]
    filters: List[Dict[str, Any]]

class TenStepParamsContainer(BaseModel):
    company_name: str
    # recursion_limit: int
    # db_names: List[str]
    # filters: List[Dict[str, Any]]

class LlamaIndexRetrievalContainer(BaseModel):
    question: str

#########################################
############### API methods #############
#########################################

#########################################
# question answer - get methods

# @app.post("/v0/langchain/info-retrieval/get-answer/")
# def get_answer_from_db(item: InfoRetrieverParamsContainer):
#     query = item.query
#     recursion_limit = item.recursion_limit
#     db_names = list(item.db_names)
#     filters = list(item.filters)
#     db_configs = [DBConfig(db_name=db_name, filter=filter)
#                   for db_name, filter in zip(db_names, filters)]
#     graph = get_research_graph(db_configs)
#     answer = graph.invoke(
#            query, {"recursion_limit": recursion_limit}
#         )
#     return answer

@app.post("/v0/langchain/report-writing/ten-step-writer/market-overview/")
def get_answer_from_db(item: TenStepParamsContainer):
    company_name = item.company_name
    # recursion_limit = item.recursion_limit
    # db_names = list(item.db_names)
    # filters = list(item.filters)
    # db_configs = [DBConfig(db_name=db_name, filter=filter)
    #               for db_name, filter in zip(db_names, filters)]
    # impax_10_step_writer = Impax10StepWriter(
    #     company_name=company_name,
    #     dbconfigs=db_configs,
    #     recursion_limit=recursion_limit
    # )
    answer = Impax10StepWriter.get_market_overview(company_name)
    return answer

@app.post("/v0/langchain/report-writing/ten-step-writer/cit-tse/")
def get_answer_from_db(item: TenStepParamsContainer):
    company_name = item.company_name
    # recursion_limit = item.recursion_limit
    # db_names = list(item.db_names)
    # filters = list(item.filters)
    # db_configs = [DBConfig(db_name=db_name, filter=filter)
    #               for db_name, filter in zip(db_names, filters)]
    # impax_10_step_writer = Impax10StepWriter(
    #     company_name=company_name,
    #     dbconfigs=db_configs,
    #     recursion_limit=recursion_limit
    # )
    answer = Impax10StepWriter.get_cit(company_name)
    return answer

@app.post("/v0/langchain/report-writing/ten-step-writer/business-model/")
def get_answer_from_db(item: TenStepParamsContainer):
    company_name = item.company_name
    # recursion_limit = item.recursion_limit
    # db_names = list(item.db_names)
    # filters = list(item.filters)
    # db_configs = [DBConfig(db_name=db_name, filter=filter)
    #               for db_name, filter in zip(db_names, filters)]
    # impax_10_step_writer = Impax10StepWriter(
    #     company_name=company_name,
    #     dbconfigs=db_configs,
    #     recursion_limit=recursion_limit
    # )
    answer = Impax10StepWriter.get_business_model(company_name)
    return answer

@app.post("/v0/langchain/report-writing/ten-step-writer/competitive-advantage/")
def get_answer_from_db(item: TenStepParamsContainer):
    company_name = item.company_name
    # recursion_limit = item.recursion_limit
    # db_names = list(item.db_names)
    # filters = list(item.filters)
    # db_configs = [DBConfig(db_name=db_name, filter=filter)
    #               for db_name, filter in zip(db_names, filters)]
    # impax_10_step_writer = Impax10StepWriter(
    #     company_name=company_name,
    #     dbconfigs=db_configs,
    #     recursion_limit=recursion_limit
    # )
    answer = Impax10StepWriter.get_competitive_advantage(company_name)
    return answer

@app.post("/v0/langchain/report-writing/ten-step-writer/risks/")
def get_answer_from_db(item: TenStepParamsContainer):
    company_name = item.company_name
    # recursion_limit = item.recursion_limit
    # db_names = list(item.db_names)
    # filters = list(item.filters)
    # db_configs = [DBConfig(db_name=db_name, filter=filter)
    #               for db_name, filter in zip(db_names, filters)]
    # impax_10_step_writer = Impax10StepWriter(
    #     company_name=company_name,
    #     dbconfigs=db_configs,
    #     recursion_limit=recursion_limit
    # )
    answer = Impax10StepWriter.get_risks(company_name)
    return answer

@app.post("/v0/llamaindex/get-simple-answers/")
def get_simple_answers(item: LlamaIndexRetrievalContainer):
    question = item.question
    response = answer_simple_questions(question)
    return response

@app.post("/v0/llamaindex/get-complex-answers/")
def get_simple_answers(item: LlamaIndexRetrievalContainer):
    question = item.question
    response = answer_complex_questions(question)
    return response


#########################################
# data ingestion - put methods
#########################################

# @app.put("/data-ingestion/greedy-load-url/")
# def greedy_ingest_data(item: GreedyUrlLoaderContainer):
#     """Greedy ingest data from a list of urls, going x layers deep in a 
#     breadth-first search fashion, according to the depth set."""
#     greedy_ingest_data_from_urls(
#         item.urls,
#         item.db_name,
#         item.depth,
#         item.adddtional_metadata,
#         item.browser
#     )
#     return "success!\n" +\
#     f"{';'.join(item.urls)} processed"

# @app.put("/data-ingestion/ingest-from-urls/")
# def ingest_from_urls(item: UrlLoaderContainer):
#     """Ingest data from a list of urls to vector database."""
#     ingest_data_from_urls(
#         item.urls,
#         item.db_name,
#         item.adddtional_metadata
#     )
#     return "success!\n " +\
#     f"{';'.join(item.urls)} processed"