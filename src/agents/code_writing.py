# import os
# import uuid
from typing import Annotated, List, Tuple, Union, Dict, Optional

import functools
import operator
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from utils.create_agent import create_agent, create_team_supervisor, agent_node
from tools.code_execution import run_python_code, eval_python_code
from api.data_ingestion import load_data_from_urls
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.tools import Tool
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_experimental.agents.agent_toolkits import create_python_agent
import functools
import operator
from langchain.tools import Tool
from agents.rag_retriever import chroma_retrieve_documents, qdrant_retrieve_documents
from langgraph.graph import StateGraph, END
from utils import DBConfig
from qdrant_client.http.models import (
    Filter,
    FieldCondition, MatchValue, Range, DatetimeRange, ValuesCount
)
# from collections import namedtuple


# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"


llm_gpt35 = AzureChatOpenAI(deployment_name="gpt35-test", model_name="gpt-35-turbo", api_version="2023-07-01-preview")
llm_gpt35_16k = AzureChatOpenAI(deployment_name="gpt-35-16k", model_name="gpt-35-turbo-16k", api_version="2023-07-01-preview")

search = GoogleSearchAPIWrapper()

google_search_tool = Tool(
    name="google_search",
    description="Search Google for results. Use this only for quick overview of results, as it does not offer much details.",
    func=search.run,
)

google_search_tool_list = Tool(
    name="google_search_list",
    description="Use this if you are asked to get the url of a website. Search Google for recent results, but returns a list of dictionaries, with keys ('title', 'link'). the links can be further used to refine search. ",
    func=lambda x: search.results(x, num_results=5),
)

scraper_tool = Tool(
    name="scrape_urls",
    func=load_data_from_urls,
    description="load content of a list of urls. Takes a list of strings"
)

def get_qdrant_retrieval_tool(db_name: str):
    doc_retrieval_tool = Tool(
        name="doc_retrieval",
        func=lambda x: qdrant_retrieve_documents(
            db_name,
            query=x,
            n_results=4
        ),
        description=f"retrieve relevant information from the {db_name} collection, from the qdrant store. This tool takes only one argument which is a string."
    )
    return doc_retrieval_tool


# agents
retrieval_tool = get_qdrant_retrieval_tool(
                    db_name="documentation_qdrant")
search_agent = create_agent(
    llm_gpt35_16k,
    [google_search_tool, google_search_tool_list, retrieval_tool],
    "Your task is to search for the relevant documentation to write code"
    "You are provided with access to the documentation of qdrant, and you can search the internet as well"
    "Look up for the relevant documentation, and write the relevant code accroding to requirements",
)
search_node = functools.partial(agent_node, agent=search_agent, name="code_documentation_search")

code_writer = create_python_agent(
    llm=llm_gpt35_16k,
    tool=run_python_code,
    system_prompt="You are a developer experienced in working with python"
    "write the code as required, pass the code to the tools provided to you" 
    "and return an object",
    verbose=True,
    agent_executor_kwargs=dict(handle_parsing_errors=True)
)
code_writer_node = functools.partial(agent_node, agent=code_writer)

supervisor_agent = create_team_supervisor(
    llm_gpt35_16k,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  code_writer, search_agent. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["code_writer", "search_agent"],
)

class TeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    input: str
    output: str

def get_qdrant_filter_writer_graph():
    graph = StateGraph(TeamState)
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("code_writer", code_writer)
    graph.add_node("search_node", search_node)

    graph.add_edge("code_writer", "supervisor")
    graph.add_edge("supervisor", "search_node")
    graph.add_edge("search_node", "code_writer")
    graph.add_conditional_edges(
        "supervisor", 
        lambda x: x["next"],
        {"supervisor": "code_writer",
         "code_writer": "search_node",
         "FINISH": END})
    graph.set_entry_point("supervisor")
    chain = graph.compile()
    def enter_chain(message: str):
        results = {
            "messages": [HumanMessage(content=message)],
        }
        return results
    chain = enter_chain | chain
    return chain

