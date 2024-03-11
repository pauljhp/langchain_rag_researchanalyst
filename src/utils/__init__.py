from typing import NamedTuple, List, Tuple, Optional, Any, Union
from collections import namedtuple, OrderedDict
import heapq


PriorityQueueItem = namedtuple(
    typename="PriorityQueueItem",
    field_names=["priority", "item"]
)

Numeric = Union[int, float, complex]

class PriorityQueue():
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

