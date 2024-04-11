from typing import Any, Callable, List, Optional, TypedDict, Union
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages)
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory, SimpleMemory
from pydantic import BaseModel


document_reducer = ReduceDocumentsChain(
    name="document_reducer",
    memory=ConversationBufferMemory(
        memory_key="message_history", # default is "chat_history"
        return_messages=True
    )
)

def create_agent(
        system_prompt: str,
        tools: List[Tool],
        llm: Union[AzureChatOpenAI, ChatOpenAI],
        response_format: BaseModel, 
        response_parser: Callable
    ):
    input = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"])
        }
    llm_with_tools = llm.bind(tools=tools)
    prompt = ChatPromptTemplate.from_messages(
        SystemMessage(content=system_prompt),
    )
    agent = (
        input |
        prompt |
        llm_with_tools |
        response_parser
    )
    return agent

def create_agent_executor(
    llm: Union[AzureChatOpenAI, ChatOpenAI],
    tools: list,
    system_prompt: str,
    team_members: List[str]
) -> str:
    """Create a function-calling agent and add it to the graph."""
    team_member_str = ','.join(team_members)
    system_prompt += "\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    f" You are chosen for a reason! You are one of the following team members: {{{team_member_str}}}."
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="intermediate_steps"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        handle_parsing_errors=True)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )