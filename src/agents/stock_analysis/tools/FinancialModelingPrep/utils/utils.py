import datetime as dt 
import pandas as pd
import numpy as np
from typing import Union, Optional, Any, Dict, List
from copy import deepcopy
import networkx as nx
from pathlib import Path
from .._abstract import AbstractAPI
from .config import Config
import itertools

DEFAULT_CONFIG = "./FinancialModelingPrep/.config/config.json"


def pandas_strptime(df: pd.DataFrame, 
    index_name: Optional[Union[str, List[str]]]=None,
    index_iloc: Optional[Union[int, List[str]]]=None,
    axis: Union[str, int]=0,
    datetime_format: str ="%Y-%m-%d",
    inplace: bool=False):
    """converts str datetime to np.datetime64
    :param index_name: index or column name to be processed
    :param index_iloc: positional index of the row/column to be processed
    :param axis: takes either 0/1, or 'index'/'columns'
    :param datetime_format: datetime.strptime format
    :param inplace: False by default, will create a deepcopy of the original 
        frame. Otherwise will changed the original frame inplace
    """
    assert index_name or index_iloc, 'index_name and index_iloc cannot be both unspecified'
    axes = {'index': 0, 'columns': 1}
    if isinstance(axis, str):
        axis = axes.get(axis)
    if inplace:
        if index_name:
            if isinstance(index_name, str):
                if axis:
                    df.loc[:, index_name] = df.loc[:, index_name]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    df.loc[index_name, :] = df.loc[index_name, :]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
            elif isinstance(index_name, list):
                if axis:
                    for ind, s in df.loc[:, index_name].iteritems():
                        df.loc[:, ind] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    for ind, s in df.loc[index_name, :].iterrows():
                        df.loc[ind, :] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))

        else:   
            if isinstance(index_iloc, int):
                if axis:
                    df.iloc[:, index_iloc] = df.iloc[:, index_iloc]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    df.iloc[index_iloc, :] = df.iloc[index_iloc, :]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
            elif isinstance(index_iloc, list):
                if axis:
                    for ind, s in df.iloc[:, index_iloc].iteritems():
                        df.loc[:, ind] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    for ind, s in df.iloc[index_iloc, :].iterrows():
                        df.loc[ind, :] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))
        return df

    else:
        newdf = deepcopy(df)
        if index_name:
            if isinstance(index_name, str):
                if axis:
                    newdf.loc[:, index_name] = newdf.loc[:, index_name]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    newdf.loc[index_name, :] = newdf.loc[index_name, :]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
            elif isinstance(index_name, list):
                if axis:
                    for ind, s in newdf.loc[:, index_name].iteritems():
                        newdf.loc[:, ind] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    for ind, s in df.loc[index_name, :].iterrows():
                        newdf.loc[ind, :] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))

        else:   
            if isinstance(index_iloc, int):
                if axis:
                    newdf.iloc[:, index_iloc] = newdf.iloc[:, index_iloc]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    newdf.iloc[index_iloc, :] = newdf.iloc[index_iloc, :]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
            elif isinstance(index_iloc, list):
                if axis:
                    for ind, s in df.iloc[:, index_iloc].iteritems():
                        newdf.loc[:, ind] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    for ind, s in df.iloc[index_iloc, :].iterrows():
                        newdf.loc[ind, :] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))
      
        return newdf


def iter_by_chunk(iterable: Any, chunk_size: int):
    """iterate by chunk size"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

# class TickerSearch(AbstractAPI):
#     """class for searching for a ticker via keywords"""
#     def __init__(self):
#         super(TickerSearch, self).__init(config=DEFAULT_CONFIG)
    
#     def _search(self, keyword: str, limit: int=10,
#         exchange: str='NASDAQ'):
#         url = "search"
#         return self._get_data(url=url, quary=keyword,
#             limit=limit, exchange=exchange)

#     @classmethod
#     def search(cls, keyword: str, limit: int=10,
#         exchange: str='NASDAQ'):
#         return cls()._search(keyword, limit, exchange)

#     def _search_ticker(self, keyword: str, limit: int=10,
#         exchange: str='NASDAQ'):
#         url = "search-ticker"
#         return self._get_data(url=url, quary=keyword,
#             limit=limit, exchange=exchange)

#     @classmethod
#     def search_ticker(cls, keyword: str, limit: int=10,
#         exchange: str='NASDAQ'):
#         return cls()._search_ticker(keyword, limit, exchange)

#     def _search_name(self, keyword: str, limit: int=10,
#         exchange: str='NASDAQ'):
#         """
#         :param exchange: takes the following:
#             'ETF' 'MUTUAL_FUND' 'COMMODITY' 'INDEX' 'CRYPTO' 'FOREX' 'TSX' 
#             'AMEX' 'NASDAQ' 'NYSE' 'EURONEXT' 'XETRA' 'NSE' 'LSE', and 'ALL'
#         """
#         available_exchanges = ['ETF', 'MUTUAL_FUND', 'COMMODITY', 'INDEX', 
#             'CRYPTO', 'FOREX', 'TSX', 'AMEX', 'NASDAQ', 'NYSE', 'EURONEXT', 
#             'XETRA' 'NSE' 'LSE', 'ALL']
#         assert exchange in availalbe_exchanges, "the exchange you specified is not available"
#         if exchange == 'ALL':
#             raise NotImplementedError 
#         else:
#             url = "search-name"
#             return self._get_data(url=url, quary=keyword,
#                 limit=limit, exchange=exchange)
    
#     @classmethod
#     def search_name(cls, keyword: str, limit: int=10,
#         exchange: str='NASDAQ'):
#         """classmethod version of _search_name"""
#         return cls()._search_name(keyword, limit, exchange)

    

# def equity_screener(mcap_ub: Union[int, float]):
#     # TODO
#     raise NotImplementedError

# ########################
# ### cyclical imports ###
# ########################
# from ..tickers import Ticker