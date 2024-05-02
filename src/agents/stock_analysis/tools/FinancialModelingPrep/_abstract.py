import requests
from requests.adapters import HTTPAdapter
from urllib.parse import urljoin, urlparse
import pandas as pd
import numpy as np
import re
import json
import time
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Optional, Callable
from .utils.config import Config
import datetime as dt
import logging
import sqlite3
from pathlib import Path
from copy import deepcopy


SOURCE_PATH = os.path.dirname(os.path.abspath(__file__))
LOGPATH = f'{SOURCE_PATH}/.log/'
LOGFILE = os.path.join(LOGPATH, 'log.log')

if not os.path.exists(LOGPATH):
    os.makedirs(LOGPATH)

logging.basicConfig(filename=LOGFILE, 
    # encoding='utf-8', 
    level=logging.DEBUG)


class AbstractAPI(ABC):
    
    def _get_config(self, 
                    api_key: str=None
                    # config: Union[str, Config, dict]
                    ):
        if api_key is None:
            api_key = os.environ.get("FMP_API_KEY")
        config = Config(dict(apikey=api_key))
        assert isinstance(config, Config)
        return config
        # if isinstance(config, Path):
        #     config = config.as_posix()
        # if isinstance(config, str):
        #     try:
        #         d = json.load(open(config))
        #         config = Config(d)
        #     except FileNotFoundError as e:
        #         logging.error(e)
        #         apikey = input("the config file you specified does not exist. Please enter APIKEY (case sensitive): ")
        #         config = Config(dict(apikey=apikey))
        #     except Exception as e:
        #         logging.error(e)
        #         raise e
        # elif isinstance(config, Config):
        #     pass
        # elif isinstance(config, dict): 
        #     config = Config(config)
        # else:
        #     raise NotImplementedError
        # assert isinstance(config, Config)
        # return config

    def _get_available_tickers(self, mode='statements'
        ) -> Union[List, pd.DataFrame]:
        """
        Get the list of all available tickers.
        """
        available_ticker_urls = {
        'statements': 'financial-statement-symbol-lists/',
        'market_data': 'available-trade/list/',
        'indices': 'symbol/available-indexes'
        }
        url = available_ticker_urls.get(mode)
        endpoint = urljoin(self._endpoint, url)
        ls = self._get_data(endpoint)
        # print(ls)
        assert isinstance(ls, list)
        if isinstance(ls[0], str):
            return ls
        else:
            new_ls = [pd.Series(i).to_frame().T for i in ls]
            df = pd.concat(new_ls).T
            return df

    def __init__(self, 
        # config: Optional[Union[Config, str]]=None, 
        version: str='v3',
        sql_path: Optional[str]=None,
        ):
        """
        :param config: The config file to use. Takes str or Config object.
            wrap apikey in the config file with "apikey": "your_apikey"
        :param version: The version of the API to use. 
            v3 is free, v4 is premium.
        :param mode: The mode of the API to use. Takes 'statements', 'market_data'
            'index', and 'funds'
        """
        self.config = self._get_config()
        self.__endpoint = 'https://financialmodelingprep.com/api/'
        self.__version = version
        self._endpoint = urljoin(self.__endpoint, f"{self.__version}/")
        self.apikey = self.config.get('apikey', returntype='str')
        if not self.apikey:
            self.apikey = input("enter your apikey: ")
        self.session = requests.Session()
        self._default_headers = dict()
        self._default_params = dict(apikey=self.apikey)
        self.session.mount('https://', HTTPAdapter(max_retries=3))
        self.sql_conn = sqlite3.connect(sql_path) if sql_path else None
        self._cur = self.sql_conn.cursor() if self.sql_conn else None
    
    def _get_data(self, url: str, ticker: Optional[str]=None, 
        additional_params: Optional[Dict]=None,
        ignore_error: bool=True, 
        **kwargs, ) -> Union[List, Dict]:
        """
        :param url: The url to get data from. Do not include the root endpoint
        """
        params = deepcopy(self._default_params)
        if ticker: params.update(dict(symbol=ticker))
        if additional_params: params.update(additional_params)
        params.update(kwargs)
        url = urljoin(self._endpoint, url)
        res = self.session.get(url, params=params)
        if res.status_code in [200, 202]:
            try:
                return res.json()
            except Exception as e:
                logging.error(e)
                if ignore_error:
                    pass
                else:
                    raise e

    def pandas_to_sql(self, pandas_obj: Union[pd.Series, pd.DataFrame],
        table_name: str, **kwargs):
        """
        :param pandas_obj: The pandas object to save to sql.
        :param table_name: The name of the table to save to.
        """
        pandas_obj.to_sql(table_name, self.sql_conn, **kwargs)
        self.sql_conn.commit()

    @classmethod
    def batch_download(cls, tickers: List[str], func: Callable, **kwargs):
        """
        :param func: The function to call for each ticker.
        :param kwargs: The keyword arguments to pass to the function.
        """
        res = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(func, ticker, **kwargs) 
                for ticker in tickers]
            res = [future.result() for future in as_completed(futures)]
        return res

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, newendpoint: str):
        self._endpoint = newendpoint

    @property
    def default_headers(self):
        return self._default_headers

    @default_headers.setter
    def default_headers(self, newheaders: Dict):
        self._default_headers = newheaders

    @property
    def default_params(self):
        return self._default_params

    @default_params.setter
    def default_params(self, newparams: Dict):
        self._default_params = newparams 