import chromadb
from chromadb.config import Settings
import os
from langchain_openai import AzureOpenAIEmbeddings
from typing import List, Dict, Tuple, Union, Any, Callable
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tqdm
from openai import RateLimitError
import time


class EmbeddingModel:
    default_embedding_model = AzureOpenAIEmbeddings(
        model=os.environ.get("DEFAULT_EMBEDDING_MODEL")
        )

class VectorDBClients:
    chroma_client = chromadb.HttpClient(
        host=os.environ.get("CHROMADB_ENDPOINT"), 
        port=os.environ.get("CHROMADB_PORT"),
        settings=Settings(
            chroma_client_auth_provider="token",
            chroma_client_auth_credentials=os.environ.get("CHROMADB_TOKEN"))
        )
    
def write_doc_to_db(
        db_name: str, 
        docs: List,
        embedding_model=EmbeddingModel.default_embedding_model,
        db_driver: Callable=VectorDBClients.chroma_client,
        id_prefix: str="",
        chunk_size: int=1000,
        additional_metadata: Dict[str, Any]={},
        verbose: bool=False,
    ) -> None:
    """write a list of documents to database"""
    # FIXME - add logic to wait if rate limite is reached
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size // 10)
    # TODO - implement logic for neo4j ingestion
    if isinstance(db_driver, chromadb.api.client.Client):
        existing_collection_names = [col.name for col in db_driver.list_collections()]
        
        def get_collection(db_name=db_name):
            if db_name in existing_collection_names:
                collection = db_driver.get_collection(db_name)
            else:
                collection = db_driver.create_collection(db_name)
            return collection
        
        def write_docs(document, id, counter):
            metadatas, embeddings, ids = [], [], []
            split_docs = text_splitter.split_text(document.page_content)
            metadata = document.metadata
            metadata.update(additional_metadata)
            metadatas = [metadata for _ in split_docs]
            try:
                embeddings = embedding_model.embed_documents(split_docs)
            except RateLimitError:
                time.sleep(1) # sleep for 1 second if rate limite error is encountered
            ids = [f"{id_prefix}_{id}_chunk{i}" for i in range(1, len(split_docs)+1)]
            collection.add(
                documents=split_docs,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            counter += len(split_docs)
            return counter
        collection = get_collection()
        counter = 0 # chromadb may crash when one collection reaches > 200k docs
        if verbose:
            for id, document in tqdm.tqdm(enumerate(docs)):
                if verbose: print(f"{counter} documents processed")
                if counter <= 200000:
                    counter = write_docs(document, id, counter)
                else:
                    collection = get_collection(f"{db_name}_{counter // 200000}")
                    counter = write_docs(document, id, counter)
        else:
            for id, document in enumerate(docs):
                if verbose: print(f"{counter} documents processed")
                if counter <= 200000:
                    counter = write_docs(document, id, counter)
                else:
                    collection = get_collection(f"{db_name}_{counter // 200000}")
                    counter = write_docs(document, id, counter)
            
    else:
        raise NotImplementedError