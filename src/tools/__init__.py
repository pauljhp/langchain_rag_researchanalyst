from langchain.tools import Tool
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain.tools import Tool


search = GoogleSearchAPIWrapper()

google_search_tool = Tool(
    name="google_search",
    description="Search Google for results. Use this only for quick overview of results, as it does not offer much details.",
    func=search.run,
)

google_search_tool_list = Tool(
    name="google_search_list",
    description="Use this if you are asked to get the url of a website. Search Google for recent results, but returns a list of dictionaries, with keys ('title', 'link'). the links can be further used to refine search. ",
    func=lambda x: search.results(x, num_results=5),
)