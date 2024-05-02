from crewai import Crew, Process
from textwrap import dedent

from stock_analysis_agents import StockAnalysisAgents
from stock_analysis_tasks import StockAnalysisTasks, AuxiliaryTasks
from langchain_openai.chat_models import AzureChatOpenAI
from tools.sec_loader_tools import SECFilingsLoader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
# from langfuse.callback import CallbackHandler
from dotenv import load_dotenv
import os
load_dotenv()


llm = AzureChatOpenAI(
    deployment_name="gpt-35-16k", 
    model_name="gpt-35-turbo-16k", 
    api_version="2023-07-01-preview")
# langfuse_handler = CallbackHandler(
#     secret_key=os.environ.get("")
#     public_key="pk-lf-...",
#     host="https://cloud.langfuse.com"
#     )


class FinancialCrew:
  def __init__(self, company):
    self.company = company

  def run(self):
    agents = StockAnalysisAgents()
    tasks = StockAnalysisTasks()

    research_analyst_agent = agents.research_analyst()
    financial_analyst_agent = agents.financial_analyst()
    editor = agents.editor()

    research_task = tasks.research(research_analyst_agent, self.company)
    financial_task = tasks.financial_analysis(financial_analyst_agent, self.company)
    # filings_task = tasks.filings_analysis(research_analyst_agent, self.company)
    esg_task = tasks.esg_analysis(research_analyst_agent, self.company)
    editing_task = tasks.edit(editor, self.company)

    crew = Crew(
      agents=[
        research_analyst_agent,
        financial_analyst_agent,
        editor
      ],
      tasks=[
        # filings_task,
        financial_task,
        research_task,
        esg_task,
        editing_task
      ],
      verbose=False,
      manager_llm=llm,
      llm=llm,
      function_calling_llm=llm,
      process=Process.hierarchical,
      # memory=True,
    )

    result = crew.kickoff()
    return result

if __name__ == "__main__":
  print("## Welcome to Financial Analysis Crew")
  print('-------------------------------')
  
  company = input(
    dedent("""
      What is the company you want to analyze?
    """))
  get_ticker = Crew(
    agents=[StockAnalysisAgents().research_assistant()],
    tasks=[AuxiliaryTasks.get_ticker(company=company, agent=StockAnalysisAgents().research_assistant())]
  )
  ticker = get_ticker.kickoff()
  sec_filing_index = SECFilingsLoader(ticker=ticker).load_data()
  # TODO - get ticker recursively until it's a valid ticker.
  # TODO - add get ticker tools
  sec_engine = sec_filing_index.as_query_engine()
  additional_tool = QueryEngineTool(
    query_engine=sec_engine,
    metadata=ToolMetadata(
      description=f"Use this tool to retrieve information from filings of {company} (ticker: {ticker}).",
      name="sec_query_tool"
    )
  )
  financial_crew = FinancialCrew(company)
  result = financial_crew.run()
  print("\n\n########################")
  print("## Here is the Report")
  print("########################\n")
  print(result)
