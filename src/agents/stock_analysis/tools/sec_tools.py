import os

import requests

from langchain.tools import tool
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS

# from sec_api import QueryApi
from unstructured.partition.html import partition_html
from langchain_openai import AzureOpenAIEmbeddings
from .sec_loader_tools import SECFilingsLoader, service_context


embedding = AzureOpenAIEmbeddings(model=os.environ.get("DEFAULT_EMBEDDING_MODEL"))
in_memory_indexes = {}


class SECTools():
  @tool("Search SEC Filings")
  def search_filings(inputs: str):
    """
    Useful to search information from the latet filing forms for a
    given stock. These typically contains the most reliable, but not the latest 
    information about the company.
    :param inputs: str. The input to this tool should be a pipe (|) separated text of
    length two, representing the 1) stock ticker you are interested in, and 2) what
    question you have from it. 
		For example, `AAPL|what was last quarter's revenue`.
    
    Make sure the first part is a valid ticker and matches with the company
    you are looking for! A google search to verify is always a good idea.
    **important**
    Try to make sure that the query is concrete. Here are some *GOOD query 
    examples*: "what is xxx's business model"; "what are xxx's product segments". 
    *BAD examples*: "latest 10-Q", "latest information" 
    (this is meaningless as it's too general and will not return anything).
    Try to avoid the BAD ones and follow the GOOD examples in your queries.
    :returns: str
    """
    # TODO - accept list of tickers as input
    
    # queryApi = QueryApi(api_key=os.environ['SEC_API_API_KEY'])
    # query = {
    #   "query": {
    #     "query_string": {
    #       "query": f"ticker:{stock} AND formType:\"10-Q\""
    #     }
    #   },
    #   "from": "0",
    #   "size": "1",
    #   "sort": [{ "filedAt": { "order": "desc" }}]
    # }

    # fillings = queryApi.get_filings(query)['filings']
    ticker, question = inputs.split("|")
    key = f"{ticker}"
    if key not in in_memory_indexes.keys():
      filings_index = SECFilingsLoader(
          tickers=[ticker], 
          filing_type="10-K", # TODO - add logic to get 20-F/10-Q
          amount=3, 
          num_workers=4,
        ).load_data()
      in_memory_indexes[key] = filings_index
    else:
      filings_index = in_memory_indexes[key]
    engine = filings_index.as_query_engine(service_context=service_context)
    if question:
      context = engine.query(question)
      return context
    else:
      return "The query engine has been prepared. However, since you did not "\
    "pass in a query, I will not return anything. Please pass a query at the "\
    "end, like this: `AAPL|what was last quarter's revenue`."

  # @tool("Search 10-K form")
  # def search_10k(data):
  #   """
  #   Useful to search information from the latest 10-K form for a
  #   given stock.
  #   The input to this tool should be a pipe (|) separated text of
  #   length two, representing the stock ticker you are interested, what
  #   question you have from it.
  #   For example, `AAPL|what was last year's revenue`.
  #   """
  #   stock, ask = data.split("|")
  #   queryApi = QueryApi(api_key=os.environ['SEC_API_API_KEY'])
  #   query = {
  #     "query": {
  #       "query_string": {
  #         "query": f"ticker:{stock} AND formType:\"10-K\""
  #       }
  #     },
  #     "from": "0",
  #     "size": "1",
  #     "sort": [{ "filedAt": { "order": "desc" }}]
  #   }

  #   fillings = queryApi.get_filings(query)['filings']
  #   if len(fillings) == 0:
  #     return "Sorry, I couldn't find any filling for this stock, check if the ticker is correct."
  #   link = fillings[0]['linkToFilingDetails']
  #   answer = SECTools.__embedding_search(link, ask)
  # #   return answer

  # def __embedding_search(url, ask):
  #   text = SECTools.__download_form_html(url)
  #   elements = partition_html(text=text)
  #   content = "\n".join([str(el) for el in elements])
  #   text_splitter = CharacterTextSplitter(
  #       separator = "\n",
  #       chunk_size = 1000,
  #       chunk_overlap  = 150,
  #       length_function = len,
  #       is_separator_regex = False,
  #   )
  #   docs = text_splitter.create_documents([content])
  #   retriever = FAISS.from_documents(
  #     docs, embedding=embedding
  #   ).as_retriever()
  #   answers = retriever.get_relevant_documents(ask, top_k=4)
  #   answers = "\n\n".join([a.page_content for a in answers])
  #   return answers

  # def __download_form_html(url):
  #   headers = {
  #     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
  #     'Accept-Encoding': 'gzip, deflate, br',
  #     'Accept-Language': 'en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7',
  #     'Cache-Control': 'max-age=0',
  #     'Dnt': '1',
  #     'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
  #     'Sec-Ch-Ua-Mobile': '?0',
  #     'Sec-Ch-Ua-Platform': '"macOS"',
  #     'Sec-Fetch-Dest': 'document',
  #     'Sec-Fetch-Mode': 'navigate',
  #     'Sec-Fetch-Site': 'none',
  #     'Sec-Fetch-User': '?1',
  #     'Upgrade-Insecure-Requests': '1',
  #     'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
  #   }

  #   response = requests.get(url, headers=headers)
  #   return response.text
