from utils.doc_loaders import CustomDocumentLoaders
import utils
from tools.web_browsing import URLCrawl, UrlContainer
import drivers
from langchain_community.vectorstores import Chroma, neo4j_vector
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.docstore.document import Document
from typing import Literal, List, Dict, Any
import os
from langchain_community.document_loaders.merge import MergedDataLoader
from qdrant_client.http.models import Distance, VectorParams


embedding_model = AzureOpenAIEmbeddings(model=os.environ.get("DEFAULT_EMBEDDING_MODEL"))

# chroma_client = drivers.VectorDBClients.chroma_client

VectorStore = Literal["chroma", "neo4j"]


    
def load_data_from_urls(
        urls: List[str],
        chunk_size: int=1000):
    web_pdfs, web_pages, local_pdfs = [], [], []
    for url in urls:
        if utils.detect_url_type(url) == "url":
            web_pages.append(url)
        else:
            if utils.detect_url_type(url) == "webpdf":
                web_pdfs.append(url)
            elif utils.detect_url_type(url) == "localpdf":
                local_pdfs.append(url)
    web_loader = utils.doc_loaders.CustomDocumentLoaders.load_urls(web_pages)
    data = web_loader.load()
    chunker = utils.Chunker(chunk_size=chunk_size)
    # TODO - handel mixed inputs with documentloader and unstructured
    for pdf_url in web_pdfs:
        try:
            loaded = utils.doc_loaders.CustomDocumentLoaders.load_web_pdf(pdf_url)
            data += loaded
        except Exception as e:
            print(f"{url} not loaded, {e}")
            pass
    chunked_data = []
    for entry in data:
        chunked_texts = chunker.split_text(entry.page_content)
        chunked_docs = [Document(page_content=text, metadata=entry.metadata) 
                        for text in chunked_texts]
        chunked_data += chunked_docs
    return chunked_data

def ingest_data_from_urls(
        urls: List[str], 
        db_name: str, 
        db_driver: drivers.VectorDBTypes="qdrant",
        chunk_size: int=1000,
        additional_metadata: Dict[str, Any]={}):
    data = load_data_from_urls(urls, chunk_size=chunk_size)
    match db_driver:
        case "chromadb":
            raise NotImplementedError
            drivers.write_doc_to_chromadb(
                db_name,
                data,
                db_driver=chroma_client,
                additional_metadata=additional_metadata
                )
        case "qdrant":
            metadatas = [d.metadata | additional_metadata for d in data]
            ids = [str(utils.get_random_uuid()) for _ in data]
            texts = [d.page_content for d in data]
            data = [d.dict() for d in data]
            embeddings = embedding_model.embed_documents(texts)
            embedding_dims = len(embeddings[0])
            existing_collections = [d.get("name") for d in 
                    drivers.VectorDBClients.qdrant_client\
                        .get_collections().model_dump()\
                        .get("collections")]
            if db_name not in existing_collections:
                drivers.VectorDBClients.qdrant_client.create_collection(
                    collection_name=db_name,
                    vectors_config=VectorParams(size=embedding_dims, distance=Distance.DOT)
                ) # TODO - refactor configs into utils
            drivers.write_doc_to_qdrant(
                db_name,
                metadatas,
                data,
                embeddings,
                ids
                )

def greedy_load_data_from_urls(
        urls: List[str], 
        depth: int=2,
        additional_metadata: Dict[str, Any]={},
        browser: Literal["selenium", "requests"]="selenium",
        chunk_size: int=1000):
    url_containers = [
        UrlContainer(url, None, None, 0, i, None) 
        for i, url in enumerate(urls)]
    match browser:
        case "requests": data = URLCrawl.greedy_get_all_links(url_containers, "requests", depth)
        case "selenium": data = URLCrawl.greedy_get_all_links(url_containers, "selenium", depth)
    chunker = utils.Chunker(chunk_size=chunk_size)
    chunked_data = []
    for entry in data:
        chunked_texts = chunker.split_text(entry.page_content)
        chunked_docs = [Document(page_content=text, metadata=entry.metadata | additional_metadata) 
                        for text in chunked_texts]
        chunked_data += chunked_docs
    return chunked_data

def greedy_ingest_data_from_urls(
        urls: List[str], 
        db_name: str, 
        depth: int=2,
        additional_metadata: Dict[str, Any]={},
        db_driver: drivers.VectorDBTypes="qdrant",
        browser: Literal["selenium", "requests"]="selenium",
        chunk_size: int=1000):
    data = greedy_load_data_from_urls(
        urls, depth, additional_metadata, 
        browser, chunk_size)
    match db_driver:
        case "chromadb":
            raise NotImplementedError
            drivers.write_doc_to_chromadb(
                db_name,
                data,
                db_driver=chroma_client,
                additional_metadata=additional_metadata
                )
        case "qdrant":
            client = drivers.VectorDBClients.qdrant_client
            metadatas = [additional_metadata for d in data]
            ids = [str(utils.get_random_uuid()) for _ in data]
            texts = [d.page_content for d in data]
            embeddings = embedding_model.embed_documents(texts)
            embedding_dims = len(embeddings[0])
            # existing_collections = drivers.get_existing_collection(
            #     client, db_type="qdrant"
            # )
            # if db_name not in existing_collections:
            if not client.collection_exists(db_name):
                client.create_collection(
                    collection_name=db_name,
                    vectors_config=VectorParams(size=embedding_dims, distance=Distance.DOT)
                ) # TODO - refactor configs into utils
            drivers.write_doc_to_qdrant(
                db_name=db_name,
                metadatas=metadatas,
                documents=data,
                embeddings=embeddings,
                ids=ids
                )