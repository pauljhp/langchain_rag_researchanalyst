from typing import Union, List, Tuple, Optional
from fastapi import (FastAPI, File, UploadFile, APIRouter, Depends,
                    Header, HTTPException, Security)
from pydantic import BaseModel
# from api.data_ingestion import greedy_ingest_data_from_urls, ingest_data_from_urls
from api.report_writing import Impax10StepWriter
from api.info_retrieval import answer_complex_questions, answer_simple_questions
from api.chat_agent import ChatSession
from fastapi.responses import HTMLResponse
# from utils import DBConfig
from typing import List, Dict, Any, Literal, Annotated
import os
from pathlib import Path


app = FastAPI()
print("Token:", os.environ.get("IMPAX_AI_ASSISTANT_TOKEN"))
##################
# token auth
##################
# TODO - change to 2-factor auth in the next version


# def get_token_header(x_token: str = Header(None)):
#     if x_token != os.environ.get("IMPAX_AI_ASSISTANT_TOKEN") or x_token is None:
#         raise HTTPException(status_code=400, detail="Invalid X-Token header")
#     return x_token

# router = APIRouter(dependencies=[Depends(get_token_header)])

# @router.post("/v0/")
# async def protected_route():
#     return {"message": "Protected routes"}

# @app.get("/")
# def root():
#     return {"intro": 
#             "Welcome to Impax's AI assistant API.\n" 
#             "This AI is built with langchain, langgrah, and Azure OpenAI.",
#             "version":
#             "alpha v0.2\n"}

# app.include_router(router)


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

class ChatSessionContainer(BaseModel):
    session_id: str
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
def get_complex_answers(item: LlamaIndexRetrievalContainer):
    question = item.question
    response = answer_complex_questions(question)
    return response

@app.post("/v0/llamaindex/chat/start-chat-session/")
def start_session(item:ChatSessionContainer):
    session_id = item.session_id
    chat_session = ChatSession(session_id)
    return {"status": "success"}

@app.post("/v0/llamaindex/chat/chat/")
def chat(item: ChatSessionContainer):
    session_id = item.session_id
    # chat_session = ChatSession(session_id)
    response = ChatSession.get_response(session_id, item.question)
    return response

@app.post("/v0/llamaindex/chat/clear-chat-session/")
def chat(item: ChatSessionContainer):
    session_id = item.session_id
    # chat_session = ChatSession(session_id)
    ChatSession.clear_session(session_id)
    return {"status": "success"}

@app.post("/v0/llamaindex/chat-with-uploads/chat/")
def chat(item: ChatSessionContainer):
    session_id = item.session_id
    # chat_session = ChatSession(session_id)
    response = ChatSession.get_response(
        session_id, item.question,
        chat_agent_type="upload_file",
        upload_file_dir=f"./temp/uploads/{session_id}")
    return response

@app.post("/v0/llamaindex/chat-with-uploads/clear-chat-session/")
def chat(item: ChatSessionContainer):
    session_id = item.session_id
    # chat_session = ChatSession(session_id)
    ChatSession.clear_session(session_id)
    return {"status": "success"}

#########################################
# data ingestion - put methods
#########################################

@app.post("/v0/local-ingestion/{session_id}/files/")
async def create_files(session_id: str, files: List[UploadFile] = File(...)):
    upload_dir = Path(f"./temp/uploads/{session_id}/")
    if not upload_dir.exists():
        upload_dir.mkdir(parents=True, exist_ok=True)

    file_sizes = []
    for file in files:
        file_path = upload_dir / file.filename
        # Asynchronously write file to the designated directory
        with open(file_path, "wb") as buffer:
            data = await file.read()  # Read file data
            buffer.write(data)
            file_sizes.append(len(data))  # Append the size of the file

    return {"file_sizes": file_sizes, "filenames": [file.filename for file in files]}

@app.post("/v0/local-ingestion/{session_id}/uploadfile/")
async def create_upload_files(session_id: str, files: List[UploadFile]):
    upload_dir = Path(f"./temp/uploads/{session_id}/")
    if not upload_dir.exists():
        upload_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
            
    return {"filenames": [file.filename for file in files]}
            

@app.get("/v0/local-ingestion/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
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