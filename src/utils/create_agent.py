from typing import Any, Callable, List, Optional, TypedDict, Union, Literal
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages)
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import Runnable
from langchain.output_parsers import (
    ResponseSchema, StructuredOutputParser, PydanticOutputParser,
    RetryOutputParser)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory, SimpleMemory
from pydantic import BaseModel
from langchain.globals import set_debug
set_debug(True)


llm = AzureChatOpenAI(
        deployment_name="gpt-35-16k", 
        model_name="gpt-35-turbo-16k", 
        api_version="2023-07-01-preview"
        )
# document_reducer = ReduceDocumentsChain(
#     name="document_reducer",
#     memory=ConversationBufferMemory(
#         memory_key="message_history", # default is "chat_history"
#         return_messages=True
#     )
# )

ResponseParserTypes = Literal["structured", "pydantic", "json"]

class Response:
    response_schemas = list()
    def add_to_response_schema(self, name: str, description: str) -> None:
        new_reponse = ResponseSchema(name=name, description=description)
        self.response_schemas.append(new_reponse)
    
    def create_response_parser(
            self,
            parser_type: ResponseParserTypes,
            retry: bool=False
            ) -> StructuredOutputParser:
        match parser_type:
            case "structured":
                parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
            case "pydantic":
                raise NotImplementedError
                parser = PydanticOutputParser(pydantic_object=) 
            case "json":
                raise NotImplementedError
                parser = JsonOutputParser(pydantic_object=)
            case _:
                raise NotImplementedError
        if retry:
            retry_parser = RetryOutputParser.from_llm(parser=parser, llm=llm)
            return retry_parser
        return parser
    
    def __init__(
            self, 
            names: List[str], 
            descriptions: List[str],
            parser_type: ResponseParserTypes="json"):
        for name, description in zip(names, descriptions):
            self.add_to_response_schema(name, description)
        self.parser = self.create_response_parser(parser_type)


def create_agent(
        # tools: List[Tool],
        llm: Union[AzureChatOpenAI, ChatOpenAI],
        response: Response,
        system_prompt: str="",
    ) -> Runnable:
    input = {
        "input": lambda x: x["input"],
        "history": lambda x: x["history"] if "history" in x.keys() else [],
        "intermediate_steps": lambda x: x["intermediate_steps"] if "intermediate_steps" in x.keys() else [],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"])
        }
    # llm_with_tools = llm.bind(tools=tools)
    system_prompt += "Follow the instruction below and produce output in the as instructed.\n" + \
        f"<system-instruction>{system_prompt}</system-instruction>\n" if bool(system_prompt) else "" + \
        f"<format-instruction>{response.parser.get_format_instructions()}</format-instruction>\n" +\
        "<question>{input}</question>"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="history"),
            MessagesPlaceholder(variable_name="intermediate_steps"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = (
        input |
        prompt |
        llm |
        response.parser
    )
    return agent

def create_agent_executor(
    llm: Union[AzureChatOpenAI, ChatOpenAI],
    tools: list,
    system_prompt: str,
    team_members: List[str],
    response: Response
) -> str:
    """Create a function-calling agent and add it to the graph."""
    team_member_str = ','.join(team_members)
    system_prompt += "\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    f" You are chosen for a reason! You are one of the following team members: {{{team_member_str}}}."
    agent = create_agent(
        system_prompt=system_prompt, 
        # tools=tools, 
        llm=llm, 
        response=response)

    # agent = create_openai_functions_agent(llm, tools, prompt)
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