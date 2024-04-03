# import os
# import uuid
from typing import Annotated, List, Tuple, Union, Dict, Optional
# import matplotlib.pyplot as plt
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.tools import tool
# from langsmith import trace
import functools
import operator
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from utils.create_agent import create_agent, create_team_supervisor, agent_node
from tools.code_execution import python_repl
from api.data_ingestion import load_data_from_urls
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.tools import Tool
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
import functools
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

class ResearchTeamState(TypedDict):

    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str

# tools
def get_chroma_retrieval_tool(db_name: str, filter: Dict):
    doc_retrieval_tool = Tool(
        name="doc_retrieval",
        func=lambda x: chroma_retrieve_documents(
            db_name,
            query=x,
            filter=filter,
            n_results=5
        ),
        description=f"retrieve relevant information from the {db_name} collection in chromadb. This tool takes only one argument which is a string."
    )
    return doc_retrieval_tool

def get_qdrant_retrieval_tool(db_name: str, filter: Filter):
    doc_retrieval_tool = Tool(
        name="doc_retrieval",
        func=lambda x: qdrant_retrieve_documents(
            db_name,
            query=x,
            filter=filter,
            n_results=5
        ),
        description=f"retrieve relevant information from the {db_name} collection, from the qdrant store. This tool takes only one argument which is a string."
    )
    return doc_retrieval_tool


# agents
def get_search_node(db_configs: List[DBConfig]):
    retrieval_tools = []
    for db_config in db_configs:
        retrieval_tool = get_chroma_retrieval_tool(**db_config._asdict())
        retrieval_tools.append(retrieval_tool)
    
    search_agent = create_agent(
        llm_gpt35_16k,
        [google_search_tool, google_search_tool_list] + retrieval_tools,
        "You are a research assistant who can search for up-to-date info using the tavily search engine.",
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")
    return search_node

research_agent = create_agent(
    llm_gpt35_16k,
    [scraper_tool],
    "You are a research assistant who can scrape specified urls for more detailed information using the scrape_webpages function.",
)

research_node = functools.partial(agent_node, agent=research_agent, name="Web Scraper")

supervisor_agent = create_team_supervisor(
    llm_gpt35_16k,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  Search, Web Scraper. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["Search", "Web Scraper"],
)

# TODO - seperate out doc retrieval agent from the prompt

def get_research_graph(dbconfigs: List[DBConfig]):
    retrievers = []
    for dbconfig in dbconfigs:
        retriever = get_chroma_retrieval_tool(*dbconfig)
        retrievers.append(retriever)
        
    search_node = get_search_node(dbconfigs)
    research_graph = StateGraph(ResearchTeamState)
    research_graph.add_node("Search", search_node)
    research_graph.add_node("Web Scraper", research_node)
    research_graph.add_node("supervisor", supervisor_agent)

    # Define the control flow
    research_graph.add_edge("Search", "supervisor")
    research_graph.add_edge("Web Scraper", "supervisor")
    research_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
        "Search": "Search", 
        "Web Scraper": "Web Scraper", 
        "FINISH": END},
    )
    research_graph.team_members = ["supervisor", "Search", "Web Scraper"]
    research_graph.set_entry_point("supervisor")
    chain = research_graph.compile()


    # The following functions interoperate between the top level graph state
    # and the state of the research sub-graph
    # this makes it so that the states of each graph don't get intermixed
    def enter_chain(message: str):
        results = {
            "messages": [HumanMessage(content=message)],
        }
        return results


    research_chain = enter_chain | chain
    return research_chain

class Impax10StepWriter():
    def __init__(
            self, 
            company_name: str, 
            dbconfigs: DBConfig,
            recursion_limit: int=10):
        """Graph with Impax's 10 step template"""
        self.company_name = company_name
        self.research_chain = get_research_graph(dbconfigs=dbconfigs)
        self.recursion_limit = recursion_limit

    def get_answer(self, question: str):
        answer = self.research_chain.invoke(
            question, {"recursion_limit": self.recursion_limit}
        )
        return answer

    def market_overview(self):
        question = f"""2.	Market
            How is the market that {self.company_name} operates in defined with respect to size, regulation, and growth? Describe the competitive landscape and the company’s position in the addressable market, together with customers and customer concentration, and suppliers? 
            Focus on the following:
            - Market size;
            - Competitive landscape;
            - Growth;
            - Regulation.
            Give detailed articulation, reasoning and evidence based on the sources provided. Include your sources"""
        return self.get_answer(question)
    
    def cit_tse(self):
        question = f"What does {self.company_name} do? Give a brief introduction of the company's history, main products and services, and business model.\n"
        "What are the company’s credentials that establish its role in the transition to a more sustainable economy?\n" 
        "Why is an investment in the company an attractive opportunity?"
        return self.get_answer(question)
    
    def competitive_advantage(self):
        question = f"Competitive Advantage of {self.company_name}"
        "What unique technologies, brand strength, embedded intellectual property, scale and distribution capabilities exist that give the business a competitive edge?\n"
        "focus on the following:\n"
        "- Network effect;\n"
        "- Customer stickiness/loyalty/switching cost;\n"
        "- Brand;\n"
        "- Economies of scale;"
        "- Unique technologies.\n"
        "Try to use all the tools, team members and information provided to you.\n"
        "Give detailed articulation, reasoning and evidence based on the sources provided. Include your sources"
        return self.get_answer(question)
    
    def business_model(self):
        question = f"Business Model and Strategy Analysis for {self.company_name}\n"
        "Does the company have a sustainable competitive advantage? Are the company’s plans credible? Are the financial returns satisfactory or is there a plan to improve these?"
        "Focus on:\n"
        "- Growth strategies - what segment will drive growth? How likely will these be realized?\n"
        "- Any M&A plans? \n"
        "- Any additional financing plans?\n"
        "- Plans to improve margin and return profile.\n"
        "Give detailed articulation, reasoning and evidence based on the sources provided.\n"
        return self.get_answer(question)

    def risks(self):
        question = f"Risks for an investment in {self.company_name}"
        return self.get_answer(question)
    