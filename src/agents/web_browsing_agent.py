from tools.web_browsing import SeleniumWebBrowser, get_driver
from typing import Annotated, List, Tuple, Union, Dict, Optional
from langchain_community.document_loaders import WebBaseLoader
import functools
import operator
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from utils.create_agent import create_agent, create_team_supervisor, agent_node
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.tools import Tool, tool
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
import functools
from agents.rag_retriever import Retriever
from langgraph.graph import StateGraph, END
from utils import DBConfig

driver = get_driver()
web_browser = SeleniumWebBrowser(driver=driver)
link_and_buttons = {"links": {}, "buttons": {}}

# tools for browsing
@tool("list_outgoing_links")
def list_outgoing_links(url: str) -> Dict[int, str]:
    """lists all the outbound links from a source url
    Use this to get an overview of the page and decide which outgoing
    link to follow next
    
    :param url: str. The source url
    """
    outgoing_links = web_browser._find_links(url)
    link_and_buttons["links"] = outgoing_links
    links = {container.id: container.text for container in outgoing_links}
    return links

@tool("click_link")
def click_link(link_id: int) -> None:
    """click on the subsequent link you decide to follow.
    This doesn't return anything, but instead modifies the `web_browser` object
    by clicking the link

    The available links are stored in `link_and_button`, and you can access
    the actual link by calling `link_and_button["links"][link_id]

    :param link_id: id of the link
    """
    link = link_and_buttons["links"][link_id]
    web_browser.driver.get(link)

@tool("get_current_page_content")
def get_current_page_content() -> str:
    """Use this to check the html content of the current page
    This will help you decide if you have reached your destination page
    
    This tool takes no arguments
    """
    return web_browser._get_current_page_content()

@tool("get_link_content")
def get_link_content(link: str) -> str:
    """Use this to check a page's content by visting it
    :param link: str. The url of the page you want to visit
    """
    return web_browser._get_link_content(link)

@tool("list_buttons")
def list_buttons(url: str) -> Dict[int, str]:
    """lists all the buttons that doesn't have an outgoing url
    Use this to get a list of the buttons available for clicking before deciding
    what to click next

    :param url: str. The source url
    """
    buttons = web_browser._find_buttons(url)
    link_and_buttons["buttons"] = buttons
    buttons_to_return = {container.id: container.text for container in buttons}
    return buttons_to_return

llm_16k = AzureChatOpenAI(
        deployment_name="gpt-35-16k", 
        model_name="gpt-35-turbo-16k", 
        api_version="2023-07-01-preview"
        )