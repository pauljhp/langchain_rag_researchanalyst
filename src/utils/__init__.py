from typing import NamedTuple, List, Tuple, Optional, Any, Union
from collections import namedtuple, OrderedDict, deque
import heapq
import tiktoken


PriorityQueueItem = namedtuple(
    typename="PriorityQueueItem",
    field_names=["priority", "item"]
)

Numeric = Union[int, float, complex]

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


#####################################################
########## shared utility functions #################
#####################################################    
    
def num_tokens_from_string(
        string: str, encoding_name: str="gpt2") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
