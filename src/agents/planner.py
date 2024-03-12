from langchain.agents import (
    AgentExecutor, 
    AgentOutputParser, 
    Agent,
    tool,
    tools
    )

from langchain.chains import LLMMathChain
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from typing import List, Any, Callable, Optional, TypedDict, Union
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class Planner:
    system_prompt = """Your task is to understand the question and break it down into parts.
    Let's first understand the problem and devise a plan to solve the problem. Think step by step. 
    For example, if the question asks for information regarding multiple companies, break then down into seperate companies.
    If the question asks for a list of questions, break it down into small steps, iterate these questions, then put the answers together. 
    If the question asks for information in non-English speaking regions, consider searching for the information in the local language, then translate the information back.
    Please output the plan starting with the header 'Plan:' and then followed by a numbered list of steps. 
    Please make the plan the minimum number of steps required to accurately complete the task. 
    If the task is a question, the final step should almost always be 'Given the above steps taken, please respond to the users original question'. 
    In your last step, check the original question to see if you have indeeded finished the task, then output the final answer.
    At the end of your plan, say '<END_OF_PLAN>'
    """
    llm = AzureChatOpenAI(deployment_name="gpt-35-16k", model_name="gpt-35-turbo-16k", api_version="2023-07-01-preview")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    planner = load_chat_planner(llm, system_prompt=system_prompt)

    def _plan(self, query: str):
        return self.planner.plan({"input": query})
    
    def _get_executor(self, llm, tools: List):
        return load_agent_executor(llm, tools, verbose=True)
    
    def __init__(self, tools: List):
        self.tools = tools
        self.executor = self._get_executor(self.llm, tools)
        self.agent = PlanAndExecute(planner=self.planner, executor=self.executor)
    
    def run(self, query: str):
        return self.agent.run(query)
    
    def agent_node(state, agent, name):
        result = agent.invoke(state)
        return {"messages": [HumanMessage(content=result["output"], name=name)]}

    def create_team_supervisor(llm: AzureChatOpenAI, system_prompt, members) -> str:
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

    def __call__(self, query: str):
        return self.analyst_execution_agent.run({"input": query})
