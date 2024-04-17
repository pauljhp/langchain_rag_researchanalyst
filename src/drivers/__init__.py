import chromadb
import qdrant_client
# from chromadb.config import Settings
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
from qdrant_client.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

VectorDBTypes = Literal["chromadb", "qdrant", "azuresearch", "auradb"]

class EmbeddingModel:
    default_embedding_model = AzureOpenAIEmbeddings(
        model=os.environ.get("DEFAULT_EMBEDDING_MODEL")
        )

class VectorDBClients:
    # chroma_client = chromadb.PersistentClient(path="../data/chromadata")
    
    # chromadb.HttpClient(
    #     host=os.environ.get("CHROMADB_ENDPOINT"), 
    #     port=os.environ.get("CHROMADB_PORT"),
    #     settings=Settings(
    #         chroma_client_auth_provider="token",
    #         chroma_client_auth_credentials=os.environ.get("CHROMADB_TOKEN")
    #         )
    #     )
    qdrant_client = qdrant_client.QdrantClient(
        url=os.environ.get("QDRANT_ENDPOINT"),
        port=None,
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=60,
        )
    azure_search_index_client = SearchIndexClient(
            endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
            credential=AzureKeyCredential(os.environ.get("AZURE_SEARCH_KEY"))
        )
    azure_search_clients = dict(
        rh_bbg=SearchClient(
            endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
            credential=AzureKeyCredential(os.environ.get("AZURE_SEARCH_KEY")),
            index_name="rh-bbg-vector-db-dev"
        ),
        rh_portal=SearchClient(
            endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
            credential=AzureKeyCredential(os.environ.get("AZURE_SEARCH_KEY")),
            index_name="rh-portal-vector-db-dev"
        ),
        rh_factset=SearchClient(
            endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
            credential=AzureKeyCredential(os.environ.get("AZURE_SEARCH_KEY")),
            index_name="rh-factset-vector-db-dev"
        )
    )
    azure_search_langchain_stores = dict(
        azure_search_client_rh_bbg = AzureSearch(
            index_name="rh-bbg-vector-db-dev", #"rh-teams-index"
            embedding_function=EmbeddingModel.default_embedding_model,
            azure_search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.environ.get("AZURE_SEARCH_KEY")
            ),
        azure_search_client_rh_portal = AzureSearch(
            index_name="rh-portal-vector-db-dev", #"rh-teams-index"
            embedding_function=EmbeddingModel.default_embedding_model,
            azure_search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.environ.get("AZURE_SEARCH_KEY")
            ),
        azure_search_client_rh_factset = AzureSearch(
            index_name="rh-factset-vector-db-dev", #"rh-teams-index"
            embedding_function=EmbeddingModel.default_embedding_model,
            azure_search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.environ.get("AZURE_SEARCH_KEY")
            ),
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
    
# class ChromaHierachicalVectorDB(BaseHierarchicalVectorDB):
#     def __init__(self, 
#             base_db_name: str, 
#             max_levels: int=3, 
#             branching_factor: int=10,
#             min_chunk_size: Optional[int]=100,
#             max_chunk_size: Optional[int]=10000,
#             embedding_model=EmbeddingModel.default_embedding_model):
#         super().__init__(base_db_name, max_levels, branching_factor, 
#                          min_chunk_size, max_chunk_size, embedding_model)
#         self.collections = {}
#         for level_name, _ in self.chunk_sizes.items():
#             self.collections[level_name] = VectorDBClients.chroma_client.create_collection(
#                 f"{self.base_db_name}_{level_name}"
#             )

#     def write_to_collection(
#             self, 
#             current_collection: Callable,
#             splitted_documents: List[str],# splitted_documents
#             metadatas: List[Dict[Hashable, Any]],
#             ids: List[str],
#             is_leaf: bool=False,
#             ):
#         """write current level to collection. If level is index, 
#         only embeddings will be written"""
#         raise NotImplementedError
#         embeddings = self.embedding_model.embed_documents(splitted_documents)
#         current_collection.add(
#             metadatas=metadatas, 
#             ids=ids, embeddings=embeddings)
#         # FIXME - not finished
        

#     def write_data(self,
#                    documents: List[base.Document], 
#                    metadatas: List[Dict[Hashable, Any]],
#                    id_prefix: str):
#         raise NotImplementedError
#     # FIXME - implementation not finished
#         current_documents = documents
#         for level_name, chunk_size in self.chunk_sizes.items(): # implement a modified B+ tree
#             collection = self.collections[level_name]
#             parent_splitter = utils.Chunker(chunk_size=chunk_size)
#             _splitted_documents = [
#                 parent_splitter.split_text(doc.page_content) 
#                 for doc in current_documents]
#             _metadatas = [
#                 [metadata for _ in docs] 
#                 for docs, metadata in zip(_splitted_documents, metadatas)]
#             splitted_documents = itertools.chain(*_splitted_documents)
#             metadatas = itertools.chain(*_metadatas)
#             embeddings = self.embedding_model.embed_documents(documents)
#             ids = [f"{id_prefix}_{level_name}_{i}" for i in range(len(splitted_documents))]

def get_existing_collection(
        client: Union[chromadb.Client, qdrant_client.QdrantClient],
        db_type: VectorDBTypes="qdrant",
    ) -> List[str]:
    match db_type:
        case "chromadb":
            collection_objs = client.list_collections()
            collection_names = [col.name for col in collection_objs]
        case "qdrant":
            collection_objs = client.get_collections()
            available_methods = dir(collection_objs)
            if "dict" in available_methods:
                collection_names = [d.get("name") for d in 
                    client.get_collections().dict()\
                        .get("collections")]
            elif "model_dump" in available_methods:
                collection_names = [d.get("name") for d in 
                    client.get_collections().model_dump()\
                        .get("collections")]
        case _:
            raise NotImplementedError("{db_type} is not implemented!")
    return collection_names


def write_doc_to_qdrant(
        db_name: str,
        metadatas: List[Dict],
        documents: List[base.Document], # dict of the actual data
        embeddings: List[List[int]],
        ids: Optional[Union[List[int], List[str]]]
    ) -> None:
    payloads = [{"metadata": metadata | d.metadata, "page_content": d.page_content} 
                for metadata, d in zip(metadatas, documents)]
    points = [PointStruct(id=id, vector=vector, payload=payload) 
        for id, vector, payload in zip(ids, embeddings, payloads)]
    VectorDBClients.qdrant_client.upsert(
        collection_name=db_name,
        points=points,
        wait=True
    )


# def write_doc_to_chromadb(
#         db_name: str, 
#         docs: List,
#         embedding_model=EmbeddingModel.default_embedding_model,
#         db_driver: Callable=VectorDBClients.chroma_client,
#         id_prefix: str="",
#         chunk_size: int=1000,
#         additional_metadata: Dict[str, Any]={},
#         verbose: bool=False,
#     ) -> None:
#     """write a list of documents to database"""
#     # FIXME - add logic to wait if rate limite is reached
#     text_splitter = RecursiveCharacterTextSplitter(
#         seperators=utils.seperators,
#         chunk_size=chunk_size, 
#         chunk_overlap=chunk_size // 10)
#     # TODO - implement logic for neo4j ingestion
#     client = db_driver
#     existing_collection_names = get_existing_collection(db_driver, db_type="chromadb")
    
#     def get_collection(db_name=db_name):
#         if db_name in existing_collection_names:
#             collection = db_driver.get_collection(db_name)
#         else:
#             collection = db_driver.create_collection(db_name)
#         return collection
    
#     def write_docs(document, id, counter):
#         metadatas, embeddings, ids = [], [], []
#         split_docs = text_splitter.split_text(document.page_content)
#         metadata = document.metadata
#         metadata.update(additional_metadata)
#         metadatas = [metadata for _ in split_docs]
#         try:
#             embeddings = embedding_model.embed_documents(split_docs)
#         except RateLimitError:
#             time.sleep(1) # sleep for 1 second if rate limite error is encountered
#         ids = [f"{id_prefix}_{id}_chunk{i}" for i in range(1, len(split_docs)+1)]
#         collection.add(
#             documents=split_docs,
#             metadatas=metadatas,
#             ids=ids,
#             embeddings=embeddings
#         )
#         counter += len(split_docs)
#         return counter
#     collection = get_collection()
#     counter = 0 # chromadb may crash when one collection reaches > 200k docs
#     if verbose:
#         for id, document in tqdm.tqdm(enumerate(docs)):
#             if verbose: print(f"{counter} documents processed")
#             if counter <= 200000:
#                 counter = write_docs(document, id, counter)
#             else:
#                 collection = get_collection(f"{db_name}_{counter // 200000}")
#                 counter = write_docs(document, id, counter)
#     else:
#         for id, document in enumerate(docs):
#             if verbose: print(f"{counter} documents processed")
#             if counter <= 200000:
#                 counter = write_docs(document, id, counter)
#             else:
#                 collection = get_collection(f"{db_name}_{counter // 200000}")
#                 counter = write_docs(document, id, counter)


