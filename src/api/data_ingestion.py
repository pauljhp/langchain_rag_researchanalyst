from utils.doc_loaders import CustomDocumentLoaders
import utils
from tools.web_browsing import URLCrawl, UrlContainer
import drivers
import chromadb
from langchain_community.vectorstores import Chroma, neo4j_vector
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from typing import Literal, List, Dict, Any
import os
from langchain_community.document_loaders.merge import MergedDataLoader


embedding_model = AzureOpenAIEmbeddings(model=os.environ.get("DEFAULT_EMBEDDING_MODEL"))

chroma_client = chromadb.HttpClient(
    host=os.environ.get("CHROMADB_ENDPOINT"), 
    port=os.environ.get("CHROMADB_PORT"))

VectorStore = Literal["chroma", "neo4j"]

def greedy_ingest_data_from_urls(
        urls: List[str], 
        db_name: str, 
        depth: int=2,
        additional_metadata: Dict[str, Any]={},
        browser: Literal["selenium", "requests"]="selenium"):
    url_containers = [
        UrlContainer(url, None, None, 0, i, None) 
        for i, url in enumerate(urls)]
    match browser:
        case "requests": data = URLCrawl.greedy_get_all_links(url_containers, "requests", depth)
        case "selenium": data = URLCrawl.greedy_get_all_links(url_containers, "selenium", depth)
    
    drivers.write_doc_to_db(
        db_name,
        data,
        db_driver=chroma_client,
        additional_metadata=additional_metadata
        )
    
def load_data_from_urls(urls: List[str]):
    web_pdfs, web_pages, local_pdfs = [], [], []
    for url in urls:
        if utils.detect_url_type(url) == "url":
            web_pages.append(url)
        else:
            if utils.detect_url_type(url) == "webpdf":
                web_pdfs.append(url)
            elif utils.detect_url_type(url) == "localpdf":
                local_pdfs(url)
    web_loader = utils.doc_loaders.CustomDocumentLoaders.load_urls(web_pages)
    data = web_loader.load()
    for pdf_url in web_pdfs:
        try:
            loaded = utils.doc_loaders.CustomDocumentLoaders.load_web_pdf(pdf_url)
            data += loaded
        except Exception as e:
            print(f"{url} not loaded, {e}")
            pass
    return data

def ingest_data_from_urls(
        urls: List[str], 
        db_name: str, 
        additional_metadata: Dict[str, Any]={}):
    data = load_data_from_urls(urls)
    drivers.write_doc_to_db(
        db_name,
        data,
        db_driver=chroma_client,
        additional_metadata=additional_metadata
        )
    