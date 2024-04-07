from typing import NamedTuple, List, Tuple, Optional, Any, Union, Literal
from collections import namedtuple, OrderedDict, deque
import heapq
import tiktoken
import validators
import requests
from urllib.parse import urlparse
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import uuid
import datetime as dt
import numpy as np
import base64
import hashlib
import pdfplumber
from tempfile import TemporaryDirectory


#####################################################
############# shared constructs ####################
#####################################################    

PriorityQueueItem = namedtuple(
    typename="PriorityQueueItem",
    field_names=["priority", "item"]
)

Numeric = Union[int, float, complex]

DBConfig = namedtuple(
    typename="DBConfig",
    field_names=["db_name", "filter"]
)

seperators = ["\t", "\n", "\n\n", " ", "\r"]

ChunkingStrategy = Literal["semantic", "tiktoken", "recursive", "character"]

tmp_dir = TemporaryDirectory()

#####################################################
################## shared classes ###################
#####################################################    
class PriorityQueue():
    """Priority queue"""
    def __init__(
            self, 
            reversed: bool=False,
            values: Optional[List[PriorityQueueItem]]=None):
        if values is not None:
            if reversed:
                self._queue = [
                    PriorityQueueItem(priority=-item.priority, item=item.item)
                    for item in values]
            else:
                self._queue = values
        else:
            self._queue = []
        heapq.heapify(self._queue)
        self.reversed = reversed
        
    def push(self, item: PriorityQueueItem) -> None:
        if self.reversed:
            heapq.heappush(self._queue, PriorityQueueItem(-item.priority, item.item))
        else:
            heapq.heappush(self._queue, PriorityQueueItem(item.priority, item.item))

    def pop(self):
        return heapq.heappop(self._queue).item
    
    def is_empty(self) -> bool:
        return len(self._queue) == 0

class Queue():
    """First in first out data structure"""
    def __init__(self, values: Optional[List[Any]]):
        if values:
            self._queue = deque(values)
        else:
            self._queue = deque()

    def push(self, newval: Any):
        self._queue.append(newval)

    def pop(self):
        item = self._queue.popleft()
        return item
    
    def __len__(self):
        return len(self._queue)
    
    def __iter__(self):
        return list(self._queue).__iter__()

class Stack():
    """Last in first out data structure"""
    def __init__(self, values: Optional[List[Any]]):
        if values:
            self._stack = deque(values)
        else:
            self._stack = deque()

    def push(self, newval: Any):
        self._stack.append(newval)

    def pop(self):
        item = self._stack.pop()
        return item
    
    def __len__(self):
        return len(self._stack)
    
    def __iter__(self):
        return list(self._stack).__iter__()

class Chunker:
    chunking_strategy_mapper = {
        "semantic": SemanticChunker,
        "tiktoken": CharacterTextSplitter,
        "character": CharacterTextSplitter,
        "recursive": RecursiveCharacterTextSplitter
    }
    def __init__(
            self, 
            chunking_strategy: ChunkingStrategy="tiktoken", 
            chunk_size: int=1000
            ):
        self.chunking_strategy = chunking_strategy
        match chunking_strategy:
            case "character" | "recursive": 
                self.chunker = self.chunking_strategy_mapper[chunking_strategy](
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_size // 10
                    )
            case "tiktoken":
                self.chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    "cl100k_base", # this is the encoder tha works with GPT 3.5 and GPT 4
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_size // 10
                )
            case _:
                raise NotImplementedError
            # TODO - implement semantic chunker

    def split_text(self, text: str):
        match self.chunking_strategy:
            case "tiktoken" | "character" | "recursive": 
                return self.chunker.split_text(text)
            case _:
                raise NotImplementedError


#####################################################
########## shared utility functions #################
#####################################################    
    
def num_tokens_from_string(
        string: str, encoding_name: str="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def is_valid_url(url: str) -> bool:
    return bool(validators.url(url))

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

def detect_url_type(url: str) -> Literal["webpdf", "localpdf", "url", "others"]:
    url_parsed = urlparse(url)
    suffix_1 = Path(url_parsed.path).suffix
    suffix_2 = Path(url_parsed.query).suffix
    if url_parsed.scheme.lower() in ("http", "https"):
        if suffix_1 in ("", ".html", ".htm") and suffix_2 in ("", ".html", ".htm"):
            return "url"
        elif suffix_1.lower() == ".pdf" or suffix_2.lower() == ".pdf":
            return "webpdf"
        else: return "others"
    elif url_parsed.scheme == "":
        if suffix_1.lower() == '.pdf':
            return "localpdf"
        else: return "others"
    else: return "others"

def encode_string(input_str, length: int=16) -> bytes:
    input_bytes = input_str.encode("utf-8")
    hash_bytes = hashlib.sha256(input_bytes).digest()
    encoded_str = base64.b64encode(hash_bytes)
    final_output = encoded_str[:length]
    return final_output

def figi_to_uuid(
        figi: str, 
        numerical_id: Optional[int]=None, 
        timestamp: Optional[dt.datetime]=None
        ) -> uuid.UUID:
    if numerical_id is not None:
        if numerical_id <= 99999999:
            numerical_id_str = str(numerical_id).zfill(8)
        else: numerical_id_str = "99999999" # overflown id
    else: numerical_id_str = "00000000"
    if timestamp is not None:
        timestamp_str = timestamp.strftime("%Y%m%d")
    else: timestamp_str = "00000000"
    uuid_str = f"{timestamp_str}{numerical_id_str}{figi}"
    uuid_bytes = encode_string(uuid_str)
    id = uuid.UUID(bytes=uuid_bytes)
    return id

def get_random_uuid():
    """get random uuid"""
    id = uuid.uuid4()
    return id

def pdf_page_has_table(page) -> bool:
    lines = page.lines
    text = page.extract_text()
    has_table = bool(lines) and bool(text)
    if has_table: return True
    else:
        table_settings = {
                "vertical_strategy": "text", # or 'lines'
                "horizontal_strategy": "text", # or 'lines'
            }
        table = page.extract_table(table_settings)
        if table:
            return True
    return False

def pdf_has_table(filepath: str) -> bool:
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            has_table = pdf_page_has_table(page)
            if has_table:
                return True
    return False