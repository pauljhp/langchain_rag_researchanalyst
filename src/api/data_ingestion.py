from utils.doc_loaders import CustomDocumentLoaders
import utils
from tools.web_browsing import URLCrawl, UrlContainer
import drivers
import chromadb
from langchain_community.vectorstores import Chroma, neo4j_vector
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from typing import Literal, List, Dict, Any
import os



embedding_model = AzureOpenAIEmbeddings(model=os.environ.get("DEFAULT_EMBEDDING_MODEL"))

chroma_client = chromadb.HttpClient(
    host=os.environ.get("CHROMADB_ENDPOINT"), 
    port=os.environ.get("CHROMADB_PORT"))

VectorStore = Literal["chroma", "neo4j"]

def recursive_ingest_data_from_urls(
        urls: List[str], 
        db_name: str, 
        depth: int=2,
        additional_metadata: Dict[str, Any]={}):
    url_containers = [
        UrlContainer(url, None, None, 0, i, None) 
        for i, url in enumerate(urls)]
    try:
        data = URLCrawl.greedy_get_all_links(url_containers, "requests", depth)
    except:
        data = URLCrawl.greedy_get_all_links(url_containers, "selenium", depth)
    
    drivers.write_docs_to_db(
        db_name,
        data,
        db_driver=chroma_client,
        additional_metadata=additional_metadata
        )
    