from query_engines import CustomQueryEngine, llm
from tools.llamaindex_tools import duckduckgo_tools, arxiv_tools, google_search_tool
from llama_index.core.agent import ReActAgent


engine = CustomQueryEngine()
query_tools = engine.get_query_engine_tools()
tools = query_tools + duckduckgo_tools + arxiv_tools + [google_search_tool]


def get_new_agent():
    agent = ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True
    )
    return agent