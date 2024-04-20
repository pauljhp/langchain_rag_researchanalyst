# from llama_index.tools.google.search.base import GoogleSearchToolSpec
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.tools.duckduckgo.base import DuckDuckGoSearchToolSpec
from llama_index.tools.arxiv.base import ArxivToolSpec
from llama_index.core.tools.function_tool import FunctionTool, ToolMetadata
# from llama_index.tools.finance import FinanceAgentToolSpec
from llama_index.tools.neo4j import Neo4jQueryToolSpec
from llama_index.tools.wikipedia import WikipediaToolSpec
from googleapiclient.discovery import build
from pydantic import BaseModel
import os


llm = AzureOpenAI(deployment_name="gpt-35-16k", 
        model="gpt-35-turbo-16k", 
        temperature=0,
        context_window=16384,
        api_version="2023-07-01-preview")

def google_search(search_term):
    service = build("customsearch", "v1", developerKey=os.environ.get("GOOGLE_API_KEY"))
    res = service.cse().list(q=search_term, cx=os.environ.get("GOOGLE_CSE_ID")).execute()
    return res['items']

class GoogleSearchParams(BaseModel):
    search_term: str


duckduckgo_tools = DuckDuckGoSearchToolSpec().to_tool_list()
arxiv_tools = ArxivToolSpec(max_results=10).to_tool_list()
wikipedia_tools = WikipediaToolSpec().to_tool_list()

google_search_tool = FunctionTool(
    fn=google_search,
    metadata=ToolMetadata(
        name="google_search_tool",
        description="Search for the latest information on google. Takes a string as input",
        fn_schema=GoogleSearchParams
    )
)

neo4j_supply_chain_tools = Neo4jQueryToolSpec(
    url=os.environ.get("NEO4J_URI"),
    user=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
    database="supplychain",
    llm=llm,
    validate_cypher=True
).to_tool_list(
    func_to_metadata_mapping={
        "construct_cypher_query": ToolMetadata(
            name="cosntruct_cypher_query",
            description="used for constructing a cypher query. "
            "Node labels in this database include Entity, Source, and Target. "
            "Relationship types include CUSTOMER and SUPPLIER. "
        )
    }
)

neo4j_company_keyword_tools = Neo4jQueryToolSpec(
    url=os.environ.get("NEO4J_URI"),
    user=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
    database="fmptest",
    llm=llm,
    validate_cypher=True
).to_tool_list(
    func_to_metadata_mapping={
        "construct_cypher_query": ToolMetadata(
        name="construct_cypher_query",
        description="Used for constructing a cypher query. "
        "This database contains node labels: Company, Entity, Keyword, environmentalIndustry, "
        "environmentalSector, environmentalTheme, socialPillar, socialSector, and socialTheme. "
        "The relations contained are descSimilar, isPeer, mention, and partOf."
    )}
)