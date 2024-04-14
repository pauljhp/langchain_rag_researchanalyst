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
def click_link(link_id: int) -> Union[bool, Tuple[bool, str]]:
    """click on the subsequent link you decide to follow.
    This doesn't return anything, but instead modifies the `web_browser` object
    by clicking the link

    The available links are stored in `link_and_button`, and you can access
    the actual link by calling `link_and_button["links"][link_id]

    :param link_id: id of the link
    """
    link = link_and_buttons["links"][link_id]
    try:
        web_browser.driver.get(link)
        return True
    except Exception as e:
        return False, e
    

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

@tool("click_button")
def click_button(button_id: int) -> Union[bool, Tuple[bool, str]]:
    """click on the bottons you decide to follow.
    This doesn't return anything, but instead modifies the `web_browser` object
    by clicking the link

    The available links are stored in `link_and_button`, and you can access
    the actual link by calling `link_and_button["links"][link_id]

    :param link_id: id of the link
    """
    button = link_and_buttons["buttons"][button_id]
    try:
        button.click()
        return True
    except Exception as e:
        return False, e


llm_16k = AzureChatOpenAI(
        deployment_name="gpt-35-16k", 
        model_name="gpt-35-turbo-16k", 
        api_version="2023-07-01-preview"
        )


# from langchain.agents import AgentType, initialize_agent, create_openai_tools_agent
# from langchain_openai.chat_models import AzureChatOpenAI
# from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
# from langchain_community.tools.playwright.utils import (
#     create_async_playwright_browser, create_sync_playwright_browser
# )
# import os
# from langchain_openai.chat_models import AzureChatOpenAI
# from utils.create_agent import create_team_supervisor, agent_node
# from langgraph.graph import StateGraph, END
# from typing_extensions import TypedDict
# from typing import Annotated, List, Tuple, Union, Dict, Optional, Any
# from langchain_core.messages import AIMessage, BaseMessage
# import functools
# import operator
# from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
# from langchain_core.prompts import PromptTemplate
# from langfuse.callback import CallbackHandler

# langfuse_handler = CallbackHandler(secret_key=os.environ.get("LANGFUSE_SECRET_KEY"), public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"), host=os.environ.get("LANGFUSE_HOST"))

# async_browser = create_async_playwright_browser()
# sync_browser = create_sync_playwright_browser()
# async_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
# sync_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
# async_tools = async_toolkit.get_tools()
# sync_tools = sync_toolkit.get_tools()

# class TeamState(TypedDict):
#       messages: Annotated[List[BaseMessage], operator.add]
#       intermediate_steps: Annotated[List[BaseMessage], operator.add]
#       team_members: List[str]
#       next: str

# llm = AzureChatOpenAI(
#         deployment_name="gpt-35-16k", 
#         model_name="gpt-35-turbo-16k", 
#         api_version="2023-07-01-preview"
#         )

# llm = AzureChatOpenAI(
#         deployment_name="gpt-35-16k", 
#         model_name="gpt-35-turbo-16k", 
#         api_version="2023-07-01-preview"
#         )

# agent_chain = initialize_agent(
#     sync_tools,
#     llm,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
# )

# from langchain.agents import AgentExecutor, create_openai_tools_agent
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant"),
#         MessagesPlaceholder("chat_history", optional=True),
#         ("human", "{input}"),
#         MessagesPlaceholder("agent_scratchpad"),
#     ]
# )

# sync_agent = create_openai_tools_agent(llm, sync_tools, prompt)
# sync_agent_executor = AgentExecutor(agent=sync_agent, tools=sync_tools, verbose=True, return_intermediate_steps=True)

# async_agent = create_openai_tools_agent(llm, sync_tools, prompt)
# async_agent_executor = AgentExecutor(agent=async_agent, tools=async_tools, verbose=True, return_intermediate_steps=True)


# def invoke_browser(question: str):
#     input = {"input": question}
#     return sync_agent_executor.invoke(input, config={"callbacks": [langfuse_handler]})

# async def invoke_browser(question: str):
#     input = {"input": question}
#     result = await async_agent_executor.ainvoke(input, config={"callbacks": [langfuse_handler]})
#     return result

# # # Using with chat history
# # from langchain_core.messages import AIMessage, HumanMessage
# # agent_executor.invoke(
# #     {
# #         "input": "what's my name?",
# #         "chat_history": [
# #             HumanMessage(content="hi! my name is bob"),
# #             AIMessage(content="Hello Bob! How can I assist you today?"),
# #         ],
# #     }
# # )
