from agents.rag_retriever import Retriever


class Impax10StepWriter:
    annual_report_retriever = Retriever(
        vector_store="qdrant",
        database_name="company_reports",
        top_k=10
    )

    @classmethod
    def get_cit(cls, company_name: str) -> str:
        template = f"Write a snap shot of the company: {company_name}, including an introduction and investment thesis. "
        "consider the following: \n"
        "What are the company’s credentials that establish its role in the transition to a more sustainable economy? \n"
        "Why is an investment in the company an attractive opportunity?"
        response = cls().annual_report_retriever.get_combined_answer(template)
        return response.content

    @classmethod
    def get_market_overview(cls, company_name: str):
        template = f"Write about the market {company_name} operates in. Structure you answers in bullet points. "
        "consider the following: \n"
        "How is the market that the company operates in defined with respect to size, level of competition, regulation, and growth?  \n"
        "Describe the competitive landscape and the company’s position in the addressable market, "
        "together with customers and customer concentration, and suppliers? "
        response = cls().annual_report_retriever.get_combined_answer(template)
        return response.content
    
    @classmethod
    def get_business_model(cls, company_name: str):
        template = f"Write about the business model of {company_name}. Structure you answers in bullet points. "
        "consider the following: \n"
        "What are the segments of the company, what is their route to market, pricing strategy, and sales structure? \n"
        "What expansion plans does the company talk about? "
        "Are the company’s plans credible? Are the financial returns satisfactory or is there a plan to improve these? "
        response = cls().annual_report_retriever.get_combined_answer(template)
        return response.content


    @classmethod
    def get_competitive_advantage(cls, company_name: str):
        template = f"Write about the competitive advantage of {company_name}. Structure you answers in bullet points. "
        "consider the following: \n"
        "What unique technologies, brand strength, embedded intellectual property, "
        "scale and distribution capabilities, network effects, or unique access to resources "
        "exist that give the business a competitive edge?"
        response = cls().annual_report_retriever.get_combined_answer(template)
        return response.content
    
    @classmethod
    def get_risks(cls, company_name: str):
        template = f"Write about the potential risks of investing in {company_name}. Structure you answers in bullet points. "
        "consider the following: \n"
        "What are the perceived risks of investing in the context of the wider landscape "
        "(industry dynamics, policy, global macro factors and societal forces), "
        "from the perspective of different stakeholders " 
        "and from the perspective of the company’s supply chain and distribution capability?"
        response = cls().annual_report_retriever.get_combined_answer(template)
        return response.content

# from agents.researcher import Impax10StepWriter, get_research_graph
# from agents.doc_writing import authoring_chain
# from langchain_openai.chat_models import AzureChatOpenAI
# from utils.create_agent import create_agent, create_team_supervisor
# from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
# from typing import TypedDict, Annotated, List
# from langgraph.graph import StateGraph, END
# import operator
# # import inspect
# from utils import DBConfig


# llm = AzureChatOpenAI(deployment_name="gpt-35-16k", model_name="gpt-35-turbo-16k", api_version="2023-07-01-preview")

# supervisor_node = create_team_supervisor(
#     llm,
#     "You are a supervisor tasked with managing a conversation between the"
#     " following teams: {team_members}. Given the following user request,"
#     " respond with the worker to act next. Each worker will perform a"
#     " task and respond with their results and status. When finished,"
#     " respond with FINISH.",
#     ["Research team", "Paper writing team"],
# )

# # Top-level graph state
# class State(TypedDict):
#     messages: Annotated[List[BaseMessage], operator.add]
#     next: str


# def get_last_message(state: State) -> str:
#     return state["messages"][-1].content


# def join_graph(response: dict):
#     return {"messages": [response["messages"][-1]]}


# def get_research_chain(dbconfigs: List[DBConfig]):
#     return get_research_graph(dbconfigs)


# def get_report_writer_graph(dbconfigs: List[DBConfig]):
#     super_graph = StateGraph(State)
#     super_graph.add_node(
#         f"Research team", get_last_message | get_research_graph(dbconfigs) | join_graph)
#     super_graph.add_node(
#         "Paper writing team", get_last_message | authoring_chain | join_graph
#     )
#     super_graph.add_node("supervisor", supervisor_node)

#     # Define the graph connections, which controls how the logic
#     # propagates through the program
#     super_graph.add_edge("Research team", "supervisor")
#     super_graph.add_edge("Paper writing team", "supervisor")
#     super_graph.add_conditional_edges(
#         "supervisor",
#         lambda x: x["next"],
#         {
#             "Paper writing team": "Paper writing team",
#             "Research team": "Research team",
#             "FINISH": END,
#         },
#     )
#     super_graph.set_entry_point("supervisor")
#     super_graph = super_graph.compile()
#     return super_graph