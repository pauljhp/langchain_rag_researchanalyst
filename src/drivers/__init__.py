import chromadb
from chromadb.config import Settings
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import base
from typing import (
    List, Dict, Tuple, Union, Any, Callable, Literal, Optional,
    Hashable)
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import tqdm
from openai import RateLimitError
import time
import utils
from collections import namedtuple
from abc import ABCMeta, abstractmethod
import itertools


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
            chroma_client_auth_credentials=os.environ.get("CHROMADB_TOKEN")
            )
        )
    
class BaseHierarchicalVectorDB(ABCMeta):
    """Base class for a hierarchical retriever
    The core idea is to build a tree-based index similar to a B+ tree
    The index collections will have empty documents but contain the embeddings,
    metadata, and children collection ids. 
    Upon retrieval a beam search will be conducted to return the most likely
    candidates
    """
    def __init__(
            self, 
            base_db_name: str, 
            max_levels: int=3, 
            branching_factor: int=10,
            min_chunk_size: Optional[int]=100,
            max_chunk_size: Optional[int]=10000,
            embedding_model=EmbeddingModel.default_embedding_model
            ):
        self.base_db_name = base_db_name
        self.max_levels = max_levels
        self.branching_factor = branching_factor
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_sizes = {
            f"level{i}_index" if i != self.max_levels - 1 else f"content": 
            max_chunk_size // branching_factor 
            for i in range(self.max_levels)
            }
        self.embedding_model = embedding_model

    @abstractmethod
    def write_data(self,
                   document: base.Document, 
                   embeddings: List[float], 
                   metadatas: List[Dict[Hashable, Any]]):
        raise NotImplementedError

    @abstractmethod
    def query(self, 
              query: str,
              window_size: int):
        raise NotImplementedError
    
class ChromaHierachicalVectorDB(BaseHierarchicalVectorDB):
    def __init__(self, 
            base_db_name: str, 
            max_levels: int=3, 
            branching_factor: int=10,
            min_chunk_size: Optional[int]=100,
            max_chunk_size: Optional[int]=10000,
            embedding_model=EmbeddingModel.default_embedding_model):
        super().__init__(base_db_name, max_levels, branching_factor, 
                         min_chunk_size, max_chunk_size, embedding_model)
        self.collections = {}
        for level_name, _ in self.chunk_sizes.items():
            self.collections[level_name] = VectorDBClients.chroma_client.create_collection(
                f"{self.base_db_name}_{level_name}"
            )

    def write_to_collection(
            self, 
            current_collection: Callable,
            splitted_documents: List[str],# splitted_documents
            metadatas: List[Dict[Hashable, Any]],
            ids: List[str],
            is_leaf: bool=False,
            ):
        """write current level to collection. If level is index, 
        only embeddings will be written"""
        raise NotImplementedError
        embeddings = self.embedding_model.embed_documents(splitted_documents)
        current_collection.add(
            metadatas=metadatas, 
            ids=ids, embeddings=embeddings)
        # FIXME - not finished
        

    def write_data(self,
                   documents: List[base.Document], 
                   metadatas: List[Dict[Hashable, Any]],
                   id_prefix: str):
        raise NotImplementedError
    # FIXME - implementation not finished
        current_documents = documents
        for level_name, chunk_size in self.chunk_sizes.items(): # implement a modified B+ tree
            collection = self.collections[level_name]
            parent_splitter = utils.Chunker(chunk_size=chunk_size)
            _splitted_documents = [
                parent_splitter.split_text(doc.page_content) 
                for doc in current_documents]
            _metadatas = [
                [metadata for _ in docs] 
                for docs, metadata in zip(_splitted_documents, metadatas)]
            splitted_documents = itertools.chain(*_splitted_documents)
            metadatas = itertools.chain(*_metadatas)
            embeddings = self.embedding_model.embed_documents(documents)
            ids = [f"{id_prefix}_{level_name}_{i}" for i in range(len(splitted_documents))]

        

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
    text_splitter = RecursiveCharacterTextSplitter(
        seperators=utils.seperators,
        chunk_size=chunk_size, 
        chunk_overlap=chunk_size // 10)
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