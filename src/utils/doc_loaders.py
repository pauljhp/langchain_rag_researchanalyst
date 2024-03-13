from typing import Dict, List, Collection

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from unstructured.partition.html import partition_html
from langchain_community.document_loaders import PyMuPDFLoader, OnlinePDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.document_loaders import SeleniumURLLoader, UnstructuredURLLoader
from langchain_core.documents.base import Document
from unstructured.documents.elements import Element
from urllib.parse import urlparse, urlunsplit
from pathlib import Path
import validators
import requests
from typing import Optional, List

class Config:
    strategy = "hi_res" # Strategy for analyzing PDFs and extracting table structure
    model_name = "yolox"

def load_pdf_after_download(pdf_url: str, referer: Optional[str]=None):
    parsed_url = urlparse(pdf_url)
    if referer is None:
        referer = urlunsplit((parsed_url.scheme, parsed_url.netloc, "", "", ""))
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0',
        'Referer': referer
    }
    # Try to get the PDF file using the requests library
    response = requests.get(pdf_url, headers=headers, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        with open('local_copy.pdf', 'wb') as f:
            f.write(response.content)
            print("downloaded file to local directory")
        loaded_pdf = PyMuPDFLoader('local_copy.pdf').load()
        if Path("local_copy.pdf").exists():
            Path("local_copy.pdf").unlink() # remove local copy
            print("removed local copy")
    return loaded_pdf


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
        status_code = requests.head(url).status_code
        if status_code >= 400: # need to download pdf
            print("downloading pdf")
            parsed_url = urlparse(url)
            if "viewer" in parsed_url.path and "file" in parsed_url.query:
                referer = urlunsplit((parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", ""))
                pdf_path = parsed_url.query.replace("file=", "")
                reconstructed_url = urlunsplit((parsed_url.scheme, parsed_url.netloc, pdf_path, "", ""))
                print(reconstructed_url)
                loaded_pdf = load_pdf_after_download(reconstructed_url)
            else:
                loaded_pdf = load_pdf_after_download(url)
            return loaded_pdf
        else:
            try:
                pdf_loader = PyMuPDFLoader(url)
                return pdf_loader.load()
            except:
                pdf_loader = OnlinePDFLoader(url)
                return pdf_loader.load()
    
    @staticmethod
    def load_urls(urls: List[str]) -> List[Document]:
        """load urls into a Document object. 
        Args: Collection of urls
        Returns: List[Document]
        """
        try:
            loader = SeleniumURLLoader(urls=urls)
            return loader
        except:
            loader = UnstructuredURLLoader(urls=urls)
            return loader
        

