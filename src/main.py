from typing import Union, List, Tuple, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from api.data_ingestion import greedy_ingest_data_from_urls, ingest_data_from_urls
from api.report_writing import get_research_graph, Impax10StepWriter
from utils import DBConfig
from typing import List, Dict, Any, Literal
import os


app = FastAPI()


@app.get("/")
def root():
    return {"intro": "Welcome to Impax's AI assistant API. This AI is built with langchain, langgrah, and Azure OpenAI."}


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
    recursion_limit: int
    db_names: List[str]
    filters: List[Dict[str, Any]]


#########################################
############### API methods #############
#########################################

#########################################
# question answer - get methods

@app.post("/info-retrieval/get-answer/")
def get_answer_from_db(item: InfoRetrieverParamsContainer):
    query = item.query
    recursion_limit = item.recursion_limit
    db_names = list(item.db_names)
    filters = list(item.filters)
    db_configs = [DBConfig(db_name=db_name, filter=filter)
                  for db_name, filter in zip(db_names, filters)]
    graph = get_research_graph(db_configs)
    answer = graph.invoke(
           query, {"recursion_limit": recursion_limit}
        )
    return answer

@app.post("/ten-step-writer/market-overview/")
def get_answer_from_db(item: TenStepParamsContainer):
    company_name = item.company_name
    recursion_limit = item.recursion_limit
    db_names = list(item.db_names)
    filters = list(item.filters)
    db_configs = [DBConfig(db_name=db_name, filter=filter)
                  for db_name, filter in zip(db_names, filters)]
    impax_10_step_writer = Impax10StepWriter(
        company_name=company_name,
        dbconfigs=db_configs,
        recursion_limit=recursion_limit
    )
    answer = impax_10_step_writer.market_overview()
    return answer

@app.post("/ten-step-writer/cit-tse/")
def get_answer_from_db(item: TenStepParamsContainer):
    company_name = item.company_name
    recursion_limit = item.recursion_limit
    db_names = list(item.db_names)
    filters = list(item.filters)
    db_configs = [DBConfig(db_name=db_name, filter=filter)
                  for db_name, filter in zip(db_names, filters)]
    impax_10_step_writer = Impax10StepWriter(
        company_name=company_name,
        dbconfigs=db_configs,
        recursion_limit=recursion_limit
    )
    answer = impax_10_step_writer.cit_tse()
    return answer

@app.post("/ten-step-writer/business-model/")
def get_answer_from_db(item: TenStepParamsContainer):
    company_name = item.company_name
    recursion_limit = item.recursion_limit
    db_names = list(item.db_names)
    filters = list(item.filters)
    db_configs = [DBConfig(db_name=db_name, filter=filter)
                  for db_name, filter in zip(db_names, filters)]
    impax_10_step_writer = Impax10StepWriter(
        company_name=company_name,
        dbconfigs=db_configs,
        recursion_limit=recursion_limit
    )
    answer = impax_10_step_writer.business_model()
    return answer

@app.post("/ten-step-writer/competitive-advantage/")
def get_answer_from_db(item: TenStepParamsContainer):
    company_name = item.company_name
    recursion_limit = item.recursion_limit
    db_names = list(item.db_names)
    filters = list(item.filters)
    db_configs = [DBConfig(db_name=db_name, filter=filter)
                  for db_name, filter in zip(db_names, filters)]
    impax_10_step_writer = Impax10StepWriter(
        company_name=company_name,
        dbconfigs=db_configs,
        recursion_limit=recursion_limit
    )
    answer = impax_10_step_writer.competitive_advantage()
    return answer

#########################################
# data ingestion - put methods

@app.put("/data-ingestion/greedy-load-url/")
def greedy_ingest_data(item: GreedyUrlLoaderContainer):
    """Greedy ingest data from a list of urls, going x layers deep in a 
    breadth-first search fashion, according to the depth set."""
    greedy_ingest_data_from_urls(
        item.urls,
        item.db_name,
        item.depth,
        item.adddtional_metadata,
        item.browser
    )
    return "success!\n" +\
    f"{';'.join(item.urls)} processed"

@app.put("/data-ingestion/ingest-from-urls/")
def ingest_from_urls(item: UrlLoaderContainer):
    """Ingest data from a list of urls to vector database."""
    ingest_data_from_urls(
        item.urls,
        item.db_name,
        item.adddtional_metadata
    )
    return "success!\n " +\
    f"{';'.join(item.urls)} processed"