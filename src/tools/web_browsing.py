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
from langchain_community.document_loaders import SeleniumURLLoader, UnstructuredURLLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
import numpy as np
from abc import ABC, abstractmethod, abstractclassmethod
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urlsplit, urljoin
from collections import namedtuple
from utils import PriorityQueueItem, PriorityQueue, Queue, Stack
import validators



UrlContainer = namedtuple(
    typename="UrlContainer",
    field_names=["url", "text", "metadata", "level", "id", "object"],
    rename=True
)

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

def is_valid_url(url: str) -> bool:
    return validators.url(url)

def load_urls(urls: List[str]):
    urls = [url for url in urls if is_valid_url(url)]
    try:
        loader = SeleniumURLLoader(urls)
        return loader.load()
    except:
        loader = UnstructuredURLLoader(urls)
        return loader.load()

def html2text(html: str) -> str:
    soup = BeautifulSoup(html, parser="lxml")
    text = " ".join([element.stripped_strings for element in soup.find_all("*")])
    return text


def url_content_type_is_text(url: str) -> bool:
    """check if the content type of a url is text"""
    try:
        response = requests.head(url, allow_redirects=True)

        if response.status_code >= 400:
            response = requests.get(url, stream=True)
        
        content_type = response.headers.get('Content-Type', '').lower()
        return content_type.startswith('text/')
    
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False





###############################################################
#################### web-browser classes ######################
###############################################################


class BaseWebBrowser(ABC):
    """Base webbrowser class"""
    @abstractmethod
    def recursive_get_text(self):
        raise NotImplementedError
    
    @abstractclassmethod
    def find_links(self):
        raise NotImplementedError

    @abstractclassmethod
    def find_input_forms(self):
        raise NotImplementedError


class SeleniumWebBrowser(BaseWebBrowser):
    """Webbrowser with selenium"""

    def __init__(self, driver: Optional[webdriver.chrome.webdriver.WebDriver]=None):
        if driver is None:
            self.driver = get_driver()
        else:
            self.driver = driver

    def recursive_get_text(self, element: webdriver.remote.webelement.WebElement):
        match element.tag_name:
            case "a":
                html = element.get_attribute("innerHTML").strip()
                text = html2text(html)
            case _:
                text = element.text.strip()
        for child in element.find_elements(by=By.XPATH, value="./*"):
            inner_text = self.recursive_get_text(child).strip()
            if inner_text != text:
                text += " " + inner_text
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
            UrlContainer(
                id=i,
                text=self.recursive_get_text(link),
                object=link,
                url=link.get_attribute("href"),
                level=1,
                metadata=None,
                # visited=False
            )

            for i, link in enumerate(res)]
        return res

    @classmethod
    def find_links(cls, url: str, driver: Optional[webdriver.chrome.webdriver.WebDriver]=None) -> List[Dict[str, Any]]:
        return cls(driver)._find_links(url)

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
    
class BeautifulSoupWebBrowser(BaseWebBrowser):
    """Web browser with requests and BeautifulSoup
    Use this for general webbrowser (where no js rendering is required)"""

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session if session else requests.Session()

    def recursive_get_text(self, element):
        return " ".join(element.stripped_strings)

    def _find_links(self, url: str) -> List[Dict[str, Any]]:
        response = self.session.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        links = soup.find_all('a')
        buttons = soup.find_all('button')
        res = links + buttons
        res = [
            UrlContainer(
                url=urljoin(url, link.get('href')) if link.name == 'a' else None,
                text=self.recursive_get_text(link),
                metadata=None,
                level=1,
                object=None,
                id=i,
                # visited=False
                )
            for i, link in enumerate(res)]
        return res

    @classmethod
    def find_links(cls, url: str, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
        return cls(session)._find_links(url)
    
    def _find_input_forms(self, url: str) -> List[Dict[str, Any]]:
        response = self.session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        input_areas = soup.find_all('input')
        res = [
            {
                "id": i,
                "placeholder_text": area.get('placeholder'),
                "obj": area
            }
            for i, area in enumerate(input_areas)]
        return res
    
    @classmethod
    def find_input_forms(cls, url: str, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
        return cls(session)._find_input_forms(url)

    # Note: Static methods like fill_form are not applicable for BeautifulSoupWebBrowser as it does not support interaction with the browser.

    def _get_outbound_links(self, url: str) -> List[Dict[str, Any]]:
        links = self._find_links(url)
        return links
    
    @classmethod
    def get_outbound_links(cls, url: str, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
        return cls(session)._get_outbound_links(url)

    def _get_link_content(self, url: str) -> str:
        response = self.session.get(url)
        # Assuming all content is text-based, not binary (like a PDF)
        return response.text
    
    @classmethod
    def get_link_content(cls, url: str, session: Optional[requests.Session] = None) -> str:
        return cls(session)._get_link_content(url)

###############################################################
######## crawler implemented with the browser classes #########
###############################################################


class URLCrawl:
    """Crawler with search functions"""
    def __init__(
            self,
            query: Optional[str]=None,
            start_urls: List[UrlContainer]=[],
            browser: Literal["selenium", "requests"]="requests",
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

    def _greedy_get_all_links(self, depth_limit: int=1):
        """depth-limited breadth-first search for all outbound links from the 
        starting urls
        TODO - add concurrent scrape
        """
        queue = Queue(self.start_urls)
        data = []
        data += load_urls([i.url for i in queue])
        max_level = max([i.level for i in queue])
        expanded_urls = []
        scraped_urls = [url_obj.url for url_obj in queue]
        while len(queue) > 0:
            if max_level < depth_limit: # keep expanding until reaching depth limit
                url_container = queue.pop()
                if is_valid_url(url_container.url):
                    if url_container.url not in expanded_urls: # expand url to get children
                        match self.browser:
                            case "selenium": new_urls = SeleniumWebBrowser.find_links(url_container.url)
                            case "requests": new_urls = BeautifulSoupWebBrowser.find_links(url_container.url)
                        expanded_urls.append(url_container.url)
                        new_urls = [url_obj for url_obj in new_urls if url_obj.url not in expanded_urls]
                        for url_obj in new_urls:
                            if is_valid_url(url_obj.url):
                                queue.push(url_obj)
                        new_data_urls = [url_obj for url_obj in new_urls if url_obj.url not in scraped_urls]
                        data += load_urls([url_obj.url for url_obj in new_data_urls])
                        scraped_urls += [url_obj.url for url_obj in new_data_urls]
                        
                max_level = max([i.level for i in queue]) if len(queue) else 0
            else: # up reaching depth limit, stop expanding children urls
                urls = []
                while len(queue) > 0:
                    url_obj = queue.pop()
                    urls.append(url_obj.url)
                print(urls)
                data += load_urls(urls)
        return data

    
    @classmethod
    def greedy_get_all_links(
        cls,
        start_urls: List[UrlContainer]=[],
        browser: Literal["selenium", "requests"]="requests",
        depth_limit: int=1,
        ):
        return cls(start_urls=start_urls, browser=browser)\
            ._greedy_get_all_links(depth_limit=depth_limit)