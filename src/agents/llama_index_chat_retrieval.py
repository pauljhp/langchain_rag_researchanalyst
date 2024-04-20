from query_engines import CustomQueryEngine, llm, embed_model
from tools.llamaindex_tools import (
    duckduckgo_tools, arxiv_tools, google_search_tool,
    wikipedia_tools
    # neo4j_company_keyword_tools, neo4j_supply_chain_tools
    )
from llama_index.core.agent import (
    ReActAgent, AgentRunner, ParallelAgentRunner, 
    FunctionCallingAgentWorker, ReActAgentWorker)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLMMetadata
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.tools import ToolMetadata, QueryEngineTool
from typing import Literal, Optional


prompt_helper = PromptHelper.from_llm_metadata(
    llm_metadata=LLMMetadata(
        context_window=16384,
        num_output=3000,
        is_chat_model=True),
    chunk_overlap_ratio=0.1, 
    chunk_size_limit=1000)
engine = CustomQueryEngine()
query_tools = engine.get_query_engine_tools()
tools = query_tools + wikipedia_tools \
        + arxiv_tools + [google_search_tool] + duckduckgo_tools 
        #+ neo4j_supply_chain_tools + neo4j_company_keyword_tools
chat_store = SimpleChatStore()


class ChatAgent:
    
    def __init__(
            self, 
            session_id: str, 
            chat_agent_type: Literal["react", "parallel", "upload_file"]="react",
            reader_directory: Optional[str]=None):
        self.session_id = session_id
        self.memory = ChatMemoryBuffer(
            token_limit=12000,
            chat_store=chat_store,
            chat_store_key=session_id)
        self.tools = tools
        self.chat_agent_type = chat_agent_type
        match chat_agent_type:
            case "react":
                self.agent = self.get_react_agent()
            case "parallel":
                self.agent = self.get_parallel_chat_agent()
            case "upload_file":
                self.agent = self.get_upload_documents_agent(reader_directory)
            case _:
                raise NotImplementedError

    def get_react_agent(self):
        agent = ReActAgent.from_tools(
            self.tools,
            llm=llm,
            verbose=True,
            max_iterations=15,
            memory=self.memory,
            prompt_helper=prompt_helper
        )
        return agent

    def get_upload_documents_agent(self, reader_directory: str):
        reader = SimpleDirectoryReader(input_dir=reader_directory, recursive=False)
        docs = reader.load_data(num_workers=4)
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=100, chunk_overlap=0),
                TitleExtractor(llm=llm, num_workers=4),
                embed_model,
            ]
        )
        nodes = pipeline.run(documents=docs)
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        loaded_file_retrieval_tool = QueryEngineTool(
            query_engine = index.as_query_engine(llm=llm, prompt_helper=prompt_helper),
            metadata=ToolMetadata(
                name="load_file_retrieval_tool",
                description="This tool helps retrieve files user uploaded. "
                "Use this to refer to files user upload. "
            )
        )
        self.tools_ = self.tools + [loaded_file_retrieval_tool]
        agent = ReActAgent.from_tools(
            self.tools_,
            llm=llm,
            verbose=True,
            max_iterations=15,
            memory=self.memory,
            prompt_helper=prompt_helper
        )
        return agent


    def get_parallel_chat_agent(self):
        step_engine = ReActAgentWorker.from_tools(
            self.tools, llm=llm, verbose=True,
            max_iterations=20)
        agent = ParallelAgentRunner(
            llm=llm,
            agent_worker=step_engine,
            memory=self.memory
            )
        return agent
    
    def clear_user_upload(self):
        assert self.chat_agent_type == "upload_file", "can only clear user upload with upload file mode"
        self.tools_.pop()