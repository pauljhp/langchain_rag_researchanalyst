from llama_index.core import PromptTemplate
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import PromptTemplate
from llama_index.core.tools import RetrieverTool
from llama_index.core.retrievers import (
    RecursiveRetriever, BaseRetriever, RouterRetriever,
    VectorIndexRetriever)
from llama_index.readers.qdrant import QdrantReader
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
import os
from typing import Optional, Union, Any, Literal
from llama_index.core.query_engine import (
    MultiStepQueryEngine, RouterQueryEngine, RetrieverQueryEngine
)
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding 
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    get_response_synthesizer
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
import drivers
from qdrant_client.http.models import VectorParams, Distance

# Define vector parameters
vector_params = VectorParams(
    size=1536,  # Adjust this based on your embedding size
    distance=Distance.DOT
)

class VectorStoreAsIndex:
    """Vector store Index for Llama-Index"""
    qdrant_client = drivers.VectorDBClients.qdrant_client

    @classmethod
    def get_qdrant_store(cls, db_name: str):
        vector_store = QdrantVectorStore(
                db_name, client=cls().qdrant_client, enable_hybrid=True, batch_size=20
            )

        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return vector_store

reader = QdrantReader(url=os.environ.get("QDRANT_ENDPOINT"), api_key="QDRANT_API_KEY", port=None)

query_gen_str_with_num = """\
You are a helpful assistant that generates multiple search queries based on a \
single input query. Generate {num_queries} search queries, one on each line, \
related to the input query below. \
If the query asks for multiple entities, companies, periods or the likes, \
break them down into individial queries on each of the entities. \
If the query is best to be completed in multiple steps, break them down \
into individual steps. \
Only write the queries to the database for retrieving the raw data. \
Query: {query}
Queries:
"""
query_gen_str_wo_num = """\
You are a helpful assistant that generates multiple search queries based on a \
single input query. Generate search queries as many as you see fit, one on each line, \
related to the input query below. \
If the query asks for multiple entities, companies, periods or the likes, \
break them down into individial queries on each of the entities. \
If the query is best to be completed in multiple steps, break them down \
into individual steps. \
Only write the queries to the database for retrieving the raw data. \
Query: {query}
Queries:
"""
query_gen_prompt_with_num = PromptTemplate(query_gen_str_with_num)
query_gen_prompt_wo_num = PromptTemplate(query_gen_str_wo_num)

llm = AzureOpenAI(deployment_name="gpt-35-16k", 
        model="gpt-35-turbo-16k", 
        api_version="2023-07-01-preview")


class CustomQueryEngine:
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name=os.environ.get("DEFAULT_EMBEDDING_MODEL")
        )

    

    def get_index(self, vector_store):
        service_context = ServiceContext.from_defaults(embed_model=self.embed_model)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)
        return index
    
    @staticmethod
    def get_vector_store(
        collection_name: str, 
        vector_store_type: drivers.VectorDBTypes
        ):
        match vector_store_type:
            case "qdrant":
                vector_store = QdrantVectorStore(
                    parallel=4,
                    collection_name=collection_name, 
                    client=drivers.VectorDBClients.qdrant_client, 
                    enable_hybrid=True, 
                    batch_size=10,
                    client_kwargs=dict(vector_params=vector_params))
            case "chromadb":
                raise NotImplementedError
            case "azuresearch":
                client = drivers.VectorDBClients.azure_search_index_client
                vector_store = AzureAISearchVectorStore(
                    search_or_index_client=client,
                    filterable_metadata_field_keys=["NoteType", "Analyst", "Security"],
                    index_name=collection_name,
                    index_management="create_if_not_exists",
                    id_field_key="id",
                    chunk_field_key="content",
                    embedding_field_key="content_vector",
                    embedding_dimensionality=1536,
                    metadata_string_field_key="metadata",
                    doc_id_field_key="id",
                    language_analyzer="en.lucene",
                    # vector_algorithm_type="exhaustiveKnn",
                )
            case _:
                raise NotImplementedError
        return vector_store

    @staticmethod
    def generate_queries(
            query: str, 
            llm: AzureOpenAI=llm, 
            num_queries: Optional[int]=None):
        if num_queries is not None:
            response = llm.predict(
                query_gen_prompt_with_num, query=query, num_queries=num_queries
            )
        else:
            response = llm.predict(
                query_gen_prompt_wo_num, query=query
            )
        queries = response.split("\n")
        # queries_str = "\n".join(queries)
        return queries

    def get_retriever(
        self,
        collection_name: str, 
        vector_store_type: drivers.VectorDBTypes,
        retriever_type: Literal["simple", "router"]="simple"
    ):
        vector_store = self.get_vector_store(
            collection_name=collection_name, 
            vector_store_type=vector_store_type)
        index = self.get_index(vector_store)
        match retriever_type:
            case "simple":
                retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=5,
                    embed_model=self.embed_model,
                    vector_store_query_mode="hybrid"
                )
            case "router":
                retriever = self.get_router_retriever()
        return retriever

    def get_router_retriever(
        self,
        ):
        earnings_transcripts = RetrieverTool.from_defaults(
            retriever=self.get_retriever("earnings_transcripts_llamaindex", "qdrant"),
            description="retrieves information from company earnings transcripts. "
        )
        historical_internal_research = RetrieverTool.from_defaults(
            retriever=self.get_retriever("rh-bbg-vector-db-dev", "azuresearch"),
            description="Retrieves older Impax internal research documents. Don't use this for latest information. "
        )
        current_internal_research = RetrieverTool.from_defaults(
            retriever=self.get_retriever("rh-portal-vector-db-dev", "azuresearch"),
            description="Retrieves current Impax internal research documents. Use this for latest information. "
        )
        company_10k = RetrieverTool.from_defaults(
            retriever=self.get_retriever("reports", "qdrant"),
            description="retrieves information from company 10k and annual reports. Use this to get a detailed view of the companies. "
        )
        retriever = RouterRetriever(
            selector=PydanticSingleSelector.from_defaults(llm=llm),
            retriever_tools=[
                earnings_transcripts, historical_internal_research,
                current_internal_research, company_10k
            ]
        )
        return retriever

    def create_simple_query_engine(
            self, 
            collection_name: str,
            vector_store_type: drivers.VectorDBTypes,
            retriever_type: Literal["simple", "router"]="simple"
            ):
        retriever = self.get_retriever(collection_name, vector_store_type, retriever_type=retriever_type)

        response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode="tree_summarize",
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        return query_engine
    
    def create_multi_step_query_engine(
            self, 
            collection_name: str,
            vector_store_type: drivers.VectorDBTypes,
            retriever_type: Literal["simple", "router"]="simple"):
        step_decompose_transform = StepDecomposeQueryTransform(llm=llm, verbose=True)
        # selector = LLMSingleSelector.from_defaults(llm=llm)
        # base_index = VectorStoreIndex(base_nodes, embed_model=embed_model)
        index_summary = "used for retrieve information about companies mentioned in the query."
        query_engine = self.create_simple_query_engine(collection_name=collection_name, vector_store_type=vector_store_type, retriever_type=retriever_type)
        query_engine = MultiStepQueryEngine(
            query_engine=query_engine,
            query_transform=step_decompose_transform,
            index_summary=index_summary,
        )
        return query_engine