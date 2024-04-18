from llama_index.tools.google.search.base import GoogleSearchToolSpec
from llama_index.tools.duckduckgo.base import DuckDuckGoSearchToolSpec
from llama_index.tools.arxiv.base import ArxivToolSpec


duckduckgo_tools = DuckDuckGoSearchToolSpec().to_tool_list()
arxiv_tools = ArxivToolSpec().to_tool_list()