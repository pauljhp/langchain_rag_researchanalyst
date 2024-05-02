import chromadb
import qdrant_client
# from chromadb.config import Settings
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import base
from typing import (
    List, Dict, Tuple, Union, Any, Callable, Literal, Optional,
    Hashable)
# from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
# import tqdm
# from openai import RateLimitError
# import time
# import utils
# from collections import namedtuple
# from abc import ABCMeta, abstractmethod
# import itertools
from qdrant_client.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
import os


VectorDBTypes = Literal["chromadb", "qdrant", "azuresearch", "auradb"]
filepath = os.path.dirname(os.path.abspath(__file__))


class EmbeddingModel:
    default_embedding_model = AzureOpenAIEmbeddings(
        model=os.environ.get("DEFAULT_EMBEDDING_MODEL")
        )

class VectorDBClients:
    chroma_client = chromadb.PersistentClient(path=f"{filepath}/../db/chromadata")
    
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