from query_engines import CustomQueryEngine, llm
from tools.llamaindex_tools import duckduckgo_tools, arxiv_tools, google_search_tool
from llama_index.core.agent import (
    ReActAgent, AgentRunner, ParallelAgentRunner, 
    FunctionCallingAgentWorker, ReActAgentWorker)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.storage.chat_store import SimpleChatStore

engine = CustomQueryEngine()
query_tools = engine.get_query_engine_tools()
tools = query_tools + duckduckgo_tools + arxiv_tools + [google_search_tool]
chat_store = SimpleChatStore()

def get_new_agent():
    agent = ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True
    )
    return agent

def get_parallel_chat_agent(session_id: str):
    step_engine = ReActAgentWorker.from_tools(tools, llm=llm, verbose=True)
    memory = ChatMemoryBuffer(
        token_limit=12000,
        chat_store=chat_store,
        chat_store_key=session_id)
    agent = ParallelAgentRunner(
        llm=llm,
        agent_worker=step_engine,
        memory=memory
        )
    return agent