from typing import Dict, List, Collection

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from unstructured.partition.html import partition_html
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_core.documents.base import Document
from unstructured.documents.elements import Element
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, AzureOpenAI
from langchain_community.vectorstores import Chroma
from urllib.parse import urlparse
from pathlib import Path
from langchain.tools import Tool
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper

from langchain.agents import (
    AgentExecutor, 
    AgentOutputParser, 
    Agent,
    tool,
    tools
    )

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

class Config:
    strategy = "hi_res" # Strategy for analyzing PDFs and extracting table structure
    model_name = "yolox"


class CustomDocumentLoaders:

    # def load_excel

    # def load_powerpoint

    # def load_word:
    
    @staticmethod
    def load_pdf(filename: str) -> List[Element]:
        """load pdf files stored locally"""
        elements = partition_pdf(
            filename=filename, 
            strategy=Config.strategy, 
            infer_table_structure=True, 
            model_name=Config.model_name
        )
        return elements

    @staticmethod
    def load_sitemap(url: str) -> List[Document]:
        """useful for loading the sitemap of a url"""
        sitemap_loader = SitemapLoader(url)
        # sitemap = sitemap_loader.load()
        return sitemap_loader
    
    @staticmethod
    def load_web_pdf(url: str) -> List[Document]:
        """Useful for loading pdf files hosted on the web"""
        pdf_loader = PyMuPDFLoader(url)
        return pdf_loader
    
    @staticmethod
    def load_url(urls: List[str]) -> List[Document]:
        """load urls into a Document object. 
        Args: Collection of urls
        Returns: List[Document]
        """
        loader = SeleniumURLLoader(urls=urls)
        # data = loader.load()
        return loader