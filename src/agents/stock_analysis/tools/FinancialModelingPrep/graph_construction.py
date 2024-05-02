"""constructing a networkx.Graph object from the API"""
from typing import Union, Any, Optional, List, Dict, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import numpy as np
import pandas as pd
import json
from .tickers import Ticker
from .indices import Index
from .utils.utils import iter_by_chunk


class ConstructGraph:

    def __init__(self, ticker_list: Optional[Union[str, List[str]]]=None):
        if isinstance(ticker_list, str):
            assert ticker_list in ["SPX", "IXIC", "DJI"], "only SPX, IXIC and DJI supported at the moment if you pass index names"
            self.ticker_list = Index().get_members(ticker_list).index.to_list()
        elif isinstance(ticker_list, list):
            self.ticker_list = ticker_list
        else: raise NotImplementedError

    def peers_graph(self, return_type: str='graph', 
        directed: bool=True, max_workers: int=8, 
        batch_request_max: int=30) -> Union[pd.DataFrame, nx.Graph, nx.DiGraph]:
        """returns a graph where the relations are 'is_peer_with'
        :param return_type: takes 'graph', 'adj_list', 'adj_mt', 'json'
        :param direct: directed graph or not
        :param max-workers: max_workers for multithreading
        :param batch-request_max: max number of tickers passed to the API at 
            once
        """
        res = []
        if max_workers > 1:
            with ThreadPoolExecutor() as executor:
                for chunk in iter_by_chunk(self.ticker_list, 
                    batch_request_max * max_workers):
                    chunk = list(chunk)
                    futures = [executor.submit(Ticker.get_peers, list(subchunk)) for 
                        subchunk in iter_by_chunk(chunk, batch_request_max)
                        ]
                    for future in as_completed(futures):
                        res += future.result()
        else:
            for chunk in iter_by_chunk(self.ticker_list, batch_request_max):
                chunk = list(chunk)
                res += Ticker.get_peers(chunk)
        if return_type in ['json']:
            return res
        else:
            adj_list = []
            for entry in res:
                triplets = [(entry.get('symbol'), 'is_peer_with', tail) 
                    for tail in entry.get('peersList')]
                adj_list += triplets
            pd_adj_ls = pd.DataFrame(adj_list, 
                columns=['head', 'relation_name', 'tail'])
            if return_type in ['adj_list']:
                return pd_adj_ls
            elif return_type in ['graph', 'adj_mt']:
                G = nx.from_pandas_edgelist(pd_adj_ls, source='head', 
                    target='tail', edge_attr='relation_name')
                if return_type == 'graph':
                    return G
                elif return_type == 'adj_mt':
                    return nx.to_pandas_adjacency(G)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    @classmethod
    def get_peers_graph(cls, ticker_list: Optional[Union[str, List[str]]]=None,
        return_type: str='graph', 
        directed: bool=True, max_workers: int=8, 
        batch_request_max: int=30):
        return cls(ticker_list=ticker_list).peers_graph(return_type=return_type, 
            directed=directed, batch_request_max=batch_request_max)

    # def ownership_graph(self, return_type: str='graph',)