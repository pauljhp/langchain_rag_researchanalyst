from llama_index.tools.google.search.base import GoogleSearchToolSpec
from llama_index.tools.duckduckgo.base import DuckDuckGoSearchToolSpec
from llama_index.tools.arxiv.base import ArxivToolSpec
from llama_index.core.tools.function_tool import FunctionTool, ToolMetadata
from googleapiclient.discovery import build
from pydantic import BaseModel
import os


def google_search(search_term):
    service = build("customsearch", "v1", developerKey=os.environ.get("GOOGLE_API_KEY"))
    res = service.cse().list(q=search_term, cx=os.environ.get("GOOGLE_CSE_ID")).execute()
    return res['items']

class GoogleSearchParams(BaseModel):
    search_term: str


duckduckgo_tools = DuckDuckGoSearchToolSpec().to_tool_list()
arxiv_tools = ArxivToolSpec(max_results=10).to_tool_list()
google_search_tool = FunctionTool(
    fn=google_search,
    metadata=ToolMetadata(
        name="google_search_tool",
        description="Search for the latest information on google. Takes a string as input",
        fn_schema=GoogleSearchParams
    )
)