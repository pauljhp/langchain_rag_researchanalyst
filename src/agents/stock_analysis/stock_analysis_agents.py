from crewai import Agent

# from tools.browser_tools import BrowserTools
from tools.fmp_tools import FMPTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from tools.sec_tools import SECTools

from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    WebsiteSearchTool,
    ScrapeWebsiteTool,
    SeleniumScrapingTool)
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.tools import Tool
# from langchain_community.utilities.google_search import GoogleSearchAPIWrapper # to be deprecated
from langchain_google_community import GoogleSearchAPIWrapper



docs_tool = DirectoryReadTool(directory='../data/outputs/blog-posts/')
file_tool = FileReadTool()
search = GoogleSearchAPIWrapper()
google_search_tool = Tool(
    name="google_search",
    description="Search Google for results. " 
    "Use this to get up to date information. " 
    "Pass in your query as a string\n"
    ":param query: str",
    func=search.run,
)

selenium_scraper = SeleniumScrapingTool()
webscraper = ScrapeWebsiteTool()
llm = AzureChatOpenAI(
    deployment_name="gpt-35-16k", 
    model_name="gpt-35-turbo-16k", 
    api_version="2023-07-01-preview")
llm_gpt4 = AzureChatOpenAI(
  api_key="19594a377a8f4b9a8a853e9dfc01433e",
  azure_endpoint="https://impax-gpt-4.openai.azure.com/",
  deployment_name="ipx-gpt-4",
  model_name="gpt-4",
  api_version="2023-07-01-preview"
)
# web_rag_tool = WebsiteSearchTool(
#   config=dict(llm=dict(
#     provider="azureopenai",
#     config=dict(model="gpt-35-16k")))
# )

class StockAnalysisAgents():
  def financial_analyst(self):
    return Agent(
      llm=llm,
      role="Senior Investment Analyst",
      goal="""Conduct thorough, sound financial analysis that's 
      impactful, complete, and accurante. Write even
      your immediate outputs to the location specified by the task whenever 
      possible.""",
      backstory="""A seasoned financial analyst, well-versed in valuation,"
      " financial statement analysis, and financial modeling.""",
      verbose=True,
      tools=[
        # BrowserTools.scrape_and_summarize_website,
        FMPTools().income_statements,
        YahooFinanceNewsTool(),
        webscraper,
        selenium_scraper,
        SearchTools.search_internet,
        CalculatorTools.calculate,
        # SECTools.search_filings,
        google_search_tool,
        file_tool
      ],
      memory=True
    )

  def research_analyst(self):
    return Agent(
      llm=llm,
      role='Senior Research Analyst',
      goal="""Conduct fundamental research and ESG analysis. Write even
      your immediate outputs to the location specified by the task whenever 
      possible.""",
      backstory="""Known as the BEST research analyst, you're
      skilled in sifting through news, company announcements, 
      and market sentiments. Now you're working on a super 
      important customer""",
      verbose=True,
      tools=[
        # BrowserTools.scrape_and_summarize_website,
        # web_rag_tool,
        webscraper,
        selenium_scraper,
        SearchTools.search_internet,
        SearchTools.search_news,
        YahooFinanceNewsTool(),
        SECTools.search_filings,
        google_search_tool,
        file_tool
      ],
      memory=True,
      allow_delegation=True,
      cache=True,
  )

  def research_assistant(self):
    return Agent(
      llm=llm,
      role='Research Assistant',
      goal="""Your goal is to help the research analyst search for relevant
      information, and organize them into digestable, well-organized formats.""",
      backstory="""The most diligent research assistant. """,
      verbose=True,
      tools=[
        SECTools.search_filings,
        SearchTools.search_internet,
        SearchTools.search_news,
        YahooFinanceNewsTool(),
        webscraper,
        selenium_scraper,
        google_search_tool,
        file_tool
      ],
      memory=True
  )

  def editor(self):
    return Agent(
      llm=llm,
      role='Editor',
      goal="""Complete well-formatted, truthful, impactful and 
      easy to read research.""",
      backstory="""You're the most experienced editor for financial
      reports. Your task is to help your teammates format their
      raw reports into final report, following a template format.""",
      verbose=True,
      tools=[
        # BrowserTools.scrape_and_summarize_website,
        # web_rag_tool,
        webscraper,
        selenium_scraper,
        SearchTools.search_internet,
        SearchTools.search_news,
        CalculatorTools.calculate,
        YahooFinanceNewsTool(),
        file_tool
      ],
      memory=True
    )