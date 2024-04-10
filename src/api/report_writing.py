from agents.researcher import Impax10StepWriter, get_research_graph
from agents.doc_writing import authoring_chain
from langchain_openai.chat_models import AzureChatOpenAI
from utils.create_agent import create_agent, create_team_supervisor
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator
# import inspect
from utils import DBConfig


llm = AzureChatOpenAI(deployment_name="gpt-35-16k", model_name="gpt-35-turbo-16k", api_version="2023-07-01-preview")

supervisor_node = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following teams: {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["Research team", "Paper writing team"],
)

# Top-level graph state
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def get_last_message(state: State) -> str:
    return state["messages"][-1].content


def join_graph(response: dict):
    return {"messages": [response["messages"][-1]]}


def get_research_chain(dbconfigs: List[DBConfig]):
    return get_research_graph(dbconfigs)


def get_report_writer_graph(dbconfigs: List[DBConfig]):
    super_graph = StateGraph(State)
    super_graph.add_node(
        f"Research team", get_last_message | get_research_graph(dbconfigs) | join_graph)
    super_graph.add_node(
        "Paper writing team", get_last_message | authoring_chain | join_graph
    )
    super_graph.add_node("supervisor", supervisor_node)

    # Define the graph connections, which controls how the logic
    # propagates through the program
    super_graph.add_edge("Research team", "supervisor")
    super_graph.add_edge("Paper writing team", "supervisor")
    super_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "Paper writing team": "Paper writing team",
            "Research team": "Research team",
            "FINISH": END,
        },
    )
    super_graph.set_entry_point("supervisor")
    super_graph = super_graph.compile()
    return super_graph