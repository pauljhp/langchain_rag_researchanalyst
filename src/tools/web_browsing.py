from typing import (
    Tuple, List, Dict, Any, Optional, Literal, Union, NamedTuple, Callable
    )
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import SeleniumURLLoader
import numpy as np
from abc import ABC, abstractmethod, abstractclassmethod
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urlsplit, urljoin
from collections import namedtuple
from utils import PriorityQueueItem, PriorityQueue



UrlContainer = namedtuple(
    typename="UrlList",
    field_names=["url", "text", "metadata", "level"],
    rename=True
)


def get_driver(
        driver_type: Literal["Chrome", "Firefox"]="Chrome"
        ) -> Union[webdriver.chrome.webdriver.WebDriver, 
                   webdriver.firefox.webdriver.WebDriver]:
    """get selenium webdriver"""
    match driver_type:
        case "Chrome":
            options = ChromeOptions()
        case "Firefox":
            options = FirefoxOptions()
        case _:
            raise ValueError(f"driver_type {driver_type} not supported!")
    
    # options.binary_location = "/usr/bin/chromedriver"
    options.add_argument("--verbose") 
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--headless")
    options.add_argument("--port=9515")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    
    options.headless = True

    match driver_type:
        case "Chrome":
            options.add_argument("--user-agent='Chrome/122.0.6261.94'")
            driver = webdriver.Chrome(options=options)
        case "Firefox":
            options.add_argument("--user-agent='Firefox'")
            driver = webdriver.Firefox(options=options)
        case _:
            raise ValueError(f"driver_type {driver_type} not supported!")
    
    driver.set_window_size(1920 * 2, 1080 * 2)
    return driver

def get_embedding_model(model_name: str="text-embedding"):
    embedding_model = AzureOpenAIEmbeddings(model=model_name)
    return embedding_model

class BaseWebBrowser(ABC):

    @abstractmethod
    def recursive_get_text(self):
        raise NotImplementedError
    
    @abstractclassmethod
    def find_links(self):
        raise NotImplementedError

    @abstractclassmethod
    def find_input_forms(self):
        raise NotImplementedError


class SeleniumWebBrowser:
    """Webbrowser with selenium"""

    def __init__(self, driver: Optional[webdriver.chrome.webdriver.WebDriver]=None):
        if driver is None:
            self.driver = get_driver()
        else:
            self.driver = driver

    def recursive_get_text(self, element: webdriver.remote.webelement.WebElement):
        text = element.text.strip() if element.text.strip() \
             else element.get_attribute("innerHTML")
        for child in element.find_elements(by=By.XPATH, value="./*"):
            text += " " + self.recursive_get_text(child)
        return text.strip()

    def _find_links(self, url: str) -> List[Dict[str, Any]]:
        """find the links and buttons on a webpage
        
        :return: List of dictionary"""
        self.driver.get(url)
        if self.driver.current_url != url: # redirected
            WebDriverWait(self.driver, 5)
        links = self.driver.find_elements(by=By.XPATH, value="//a")
        buttons = self.driver.find_elements(by=By.XPATH, value="//button")
        res = links + buttons 
        res = [
            {
                "id": i, 
                "text": self.recursive_get_text(link), 
                "obj": link, 
                "url": link.get_attribute("href")
                } 
            for i, link in enumerate(res)]
        return res

    @classmethod
    def find_links(cls, url: str, driver: Optional[webdriver.chrome.webdriver.WebDriver]=None) -> List[Dict[str, Any]]:
        return cls(driver)._find_links(url)
    
    def _beam_search_relevant_links(self, query: str, url: str, depth_limit: int=5):
        """beam search to get relevant links given an url"""
        raise NotImplementedError

    def _find_input_forms(self, url: str):
        """find the input areas on a webpage"""
        self.driver.get(url)
        input_areas = self.driver.find_elements(by=By.XPATH, value="//input")
        res = [
            {
                "id": i, 
                "placeholder_text": area.get_property("placeholder"), 
                "obj": area
                } 
            for i, area in enumerate(input_areas)]
        return res
    
    @classmethod
    def find_input_forms(cls, url: str):
        return cls()._find_input_forms(url)

    @staticmethod
    def fill_form(obj, input: str) -> None:
        obj.send_keys(input, Keys.ENTER)

    def _get_outbound_links(
            self,
            url: str
            ) -> Tuple[List[Dict[str, Any]], webdriver.chrome.webdriver.WebDriver]:
        """does the same as _find_links, but also returns the driver object"""
        links = self._find_links(url)
        return links, self.driver
    
    @classmethod
    def get_outbound_links(
        cls, 
        url: str,
        driver: Optional[webdriver.chrome.webdriver.WebDriver]=None
        ) -> Tuple[List[Dict[str, Any]], webdriver.chrome.webdriver.WebDriver]:
        """classmethod version of _get_outbound_links. takes an url and returns
        a list of its outbound urls
        
        :param url: str. The url of the website of interest

        :return: Tuple of (List[Dict], webdriver.chrome.webdriver.WebDriver)
        """
        if driver is not None:
            return cls(driver)._get_outbound_links(url)
        else:
            return cls()._get_outbound_links(url)

    def _get_link_content(
            self,
            url:str
        ) -> str:
        # TODO - add handler for web documents like *.pdf
        self.driver.get(url)
        text = self.driver.find_elements(by=By.XPATH, value="//html")[0].text
        return text
    
    @classmethod
    def get_link_content(cls, url: str) -> str:
        return cls()._get_link_content(url)
    
###############################################################
########### web-browsing specific helper functions ############
###############################################################

def get_l2_dist(
    query_embedding: np.array, 
    candidate_embeddings: List[np.array], 
    dist_thres: float=0.6,
) -> List[Tuple[int, int]]:
    # query_embedding = np.array(embedding_model.embed_query(query))
    # candidate_embeddings = embedding_model.embed_documents([candidate["text"] for candidate in candidates])
    res = []
    for candidate_id, emb in enumerate(candidate_embeddings):
        dist = np.linalg.norm(query_embedding - emb)
        res.append((dist, candidate_id))
    res = sorted(res, key=lambda x: x[0])
    res = [(dist, id) for dist, id in res if dist <= dist_thres]
    return res


class URLCrawl:
    """Crawler with search functions"""
    def __init__(
            self,
            query: str,
            start_urls: List[UrlContainer],
            browser: Literal["selenium", "requests"]="selenium",
            embedding_model_name: str="text-embedding",
            distance_threshold: float=0.6
            ):
        """
        :param query: question of interest. takes string
        :param start_url: convert each start url into an instance of the 
            UrlContainer class
        :param browser: which WebBrowser class to use
        :param embedding_model_name: which embedding model to use. This will
        be passed to AzureOpenAI
        """
        self.browser = browser
        self.query = query
        self.embedding_model = get_embedding_model(embedding_model_name)
        self.query_embedding = self.embedding_model.embed_query(query)
        self.start_urls = start_urls
        self.distance_thres = distance_threshold

    def _beam_search_target(self, stopping_criteria: Callable):
        raise NotImplementedError
        url_text_embeddings = self.embedding_model.embed_documents([item.text for item in self.start_urls])
        distances = get_l2_dist(self.query_embedding, url_text_embeddings, self.distance_thres)
        open_queue = [PriorityQueueItem(priority=dist, item=item) for dist, item in zip(distances, self.start_urls)]


    @classmethod
    def beam_search_target(cls):
        raise NotImplementedError

    def _greedy_get_all_links(self):
        """depth-limited search for all outbound links from the starting url"""
        raise NotImplementedError