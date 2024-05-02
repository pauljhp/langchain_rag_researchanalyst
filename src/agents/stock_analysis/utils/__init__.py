from llama_index.core import PromptTemplate
from llama_index.llms.azure_openai import AzureOpenAI
# from langchain_openai import AzureChatOpenAI
from llama_index.core import PromptTemplate
from llama_index.core.tools import RetrieverTool, ToolMetadata, QueryPlanTool, QueryEngineTool
from llama_index.core.retrievers import (
    RecursiveRetriever, BaseRetriever, RouterRetriever,
    VectorIndexRetriever)
from llama_index.readers.qdrant import QdrantReader
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
import os
from typing import Optional, Union, Any, Literal, List, Dict
from llama_index.core.query_engine import (
    MultiStepQueryEngine, RouterQueryEngine, RetrieverQueryEngine,
    SubQuestionQueryEngine
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
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings, Response, QueryBundle
# from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
# from llama_index.core.selectors import (
#     PydanticMultiSelector,
#     PydanticSingleSelector,
# )
# from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from . import driver
from qdrant_client.http.models import VectorParams, Distance


filepath = os.path.dirname(os.path.abspath(__file__))    
# Define vector parameters
vector_params = VectorParams(
    size=1536,  # Adjust this based on your embedding size
    distance=Distance.DOT
)

class VectorStoreAsIndex:
    """Vector store Index for Llama-Index"""
    qdrant_client = driver.VectorDBClients.qdrant_client

    @classmethod
    def get_qdrant_store(cls, db_name: str):
        vector_store = QdrantVectorStore(
                db_name, client=cls().qdrant_client, enable_hybrid=True, batch_size=20
            )

        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return vector_store

reader = QdrantReader(url=os.environ.get("QDRANT_ENDPOINT"), api_key="QDRANT_API_KEY", port=None)

class GeneratedSubQueries(BaseModel):
    reply: List[str]

query_parser = PydanticOutputParser(output_cls=GeneratedSubQueries)

llm = AzureOpenAI(deployment_name="gpt-35-16k", 
        model="gpt-35-turbo-16k", 
        temperature=0,
        context_window=16384,
        api_version="2023-07-01-preview")


class CustomQueryEngine:
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name=os.environ.get("DEFAULT_EMBEDDING_MODEL")
        )
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        chunk_size_limit=1000,
        )

    def get_index(self, vector_store):
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, 
            service_context=self.service_context)
        return index
    
    @staticmethod
    def get_vector_store(
        collection_name: str, 
        vector_store_type: driver.VectorDBTypes
        ):
        match vector_store_type:
            case "qdrant":
                vector_store = QdrantVectorStore(
                    parallel=4,
                    collection_name=collection_name, 
                    client=driver.VectorDBClients.qdrant_client, 
                    enable_hybrid=True, 
                    batch_size=64,
                    client_kwargs=dict(vector_params=vector_params))
            case "chromadb":
                collection = driver.VectorDBClients.chroma_client\
                    .create_collection(
                        name=collection_name,
                        get_or_create=True
                )
                vector_store = ChromaVectorStore(
                    chroma_collection=collection
                    # persist_dir=f"{filepath}/../db/chromadata"
                )
            case _:
                raise NotImplementedError
        return vector_store

    def get_retriever(
        self,
        collection_name: str, 
        vector_store_type: driver.VectorDBTypes,
        vector_store_index_kwargs: Optional[Dict[str, Any]]={},
        retriever_type: Literal["simple", "router"]="simple"
    ) -> BaseRetriever:
        vector_store = self.get_vector_store(
            collection_name=collection_name, 
            vector_store_type=vector_store_type)
        index = self.get_index(vector_store)
        match retriever_type:
            case "simple":
                retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=14,
                    embed_model=self.embed_model,
                    vector_store_query_mode="hybrid",
                    **vector_store_index_kwargs
                )
            case "router":
                retriever = self.get_router_retriever()
        return retriever

    def get_retrieval_tools(self) -> List[RetrieverTool]:
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
        tools = [earnings_transcripts, historical_internal_research, current_internal_research, company_10k]
        return tools
 

    def create_simple_query_engine(
            self, 
            collection_name: str,
            vector_store_type: driver.VectorDBTypes,
            retriever_type: Literal["simple", "router"]="simple",
            synthesize: bool=True,
            vector_store_index_kwargs: Optional[Dict[str, Any]]={}
            ):
        retriever = self.get_retriever(
            collection_name, 
            vector_store_type, 
            retriever_type=retriever_type,
            vector_store_index_kwargs=vector_store_index_kwargs)
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode="tree_summarize",
            service_context=self.service_context
        )
        if synthesize:
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
            )
        else:
            query_engine = RetrieverQueryEngine(
                retriever=retriever)
        return query_engine
    
    def create_sub_questions_query_engine(
            self,
    ):
        engine_tools = self.get_query_engine_tools()
        # question_generator = LLMQuestionGenerator(
        #     llm=llm, 
        #     prompt=query_gen_prompt_wo_num.partial_format(query=lambda x: x),
        #     )
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode="tree_summarize",
            service_context=self.service_context
        )
        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=engine_tools,
            llm=llm,
            service_context=self.service_context,
            # question_gen=question_generator,
            response_synthesizer=response_synthesizer,
            use_async=False
        )
        return query_engine

    def create_multi_step_query_engine(
            self, 
            collection_name: str,
            vector_store_type: driver.VectorDBTypes,
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