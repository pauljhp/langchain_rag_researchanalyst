from tools.web_browsing import SeleniumWebBrowser
from typing import Annotated, List, Tuple, Union, Dict, Optional
from langchain_community.document_loaders import WebBaseLoader
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
from agents.rag_retriever import chroma_retrieve_documents
from langgraph.graph import StateGraph, END
from utils import DBConfig