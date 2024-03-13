from typing import Union, List, Tuple, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from api.data_ingestion import greedy_ingest_data_from_urls
import os

api_type = "azure"
api_key = "d8a6014348be4cc7951de2428dd55594"
api_version = "2023-07-01-preview"
api_base = "https://ipx-neo4j-openai-test.openai.azure.com/"
deployment_name = "gpt35-test"
model_name = "gpt-35-turbo"

os.environ['OPENAI_API_KEY'] = "d8a6014348be4cc7951de2428dd55594"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = api_version
os.environ["AZURE_OPENAI_ENDPOINT"] = api_base
os.environ["GOOGLE_CSE_ID"] = "040ca56f062be4799"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAzIpgVYFkbF-jcgVdIflJ5HeBJ-Jl9E7A"
os.environ["TAVILY_API_KEY"] = "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"
os.environ["CHROMADB_ENDPOINT"] = "https://impax-chromadb-test.azurewebsites.net/"
os.environ["CHROMADB_PORT"] = "8000"
os.environ["DEFAULT_EMBEDDING_MODEL"] = "text-embedding"

app = FastAPI()

@app.get("/")
def root():
    return {"message": "hello"}

class UrlLoaderContainer(BaseModel):
    urls: list
    db_name: str
    depth: int

@app.put("/data-ingestion/greedy-load-url/")
def greedy_ingest_data(item: UrlLoaderContainer):
    greedy_ingest_data_from_urls(
        item.urls,
        item.db_name,
        item.depth
    )
    print("success!")
    print(f"{item.urls} processed")