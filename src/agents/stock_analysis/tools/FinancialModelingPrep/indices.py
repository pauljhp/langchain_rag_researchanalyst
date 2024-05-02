from __future__ import annotations
from doctest import DocFileCase
from urllib.parse import urljoin
from copy import deepcopy
import pandas as pd
from typing import Optional, Union, List, Dict, Callable
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
from ._abstract import AbstractAPI
from .utils.config import Config
from .utils.utils import pandas_strptime


DEFAULT_CONFIG = "./FinancialModelingPrep/.config/config.json"
QUARTER_END = {
    1: (3, 31), 
    2: (6, 30),
    3: (9, 30),
    4: (12, 31)
    }
TODAY = dt.datetime.today()
NOW = dt.datetime.now()
CUR_YEAR = TODAY.year
LAST_Q = (TODAY - dt.timedelta(days=90)).month // 3

config_p = Path(DEFAULT_CONFIG)
if not config_p.exists():
    config_p.parent.mkdir(parents=True, exist_ok=True)
    config_p.write_text(r"""{{"apikey": "{a}"}}""".format(
    a=input("the config file wasn't found - enter your apikey: "))
    )


class Index(AbstractAPI):

    def __init__(self, 
        config: Union[str, Config]=DEFAULT_CONFIG, 
        **kwargs):
        super(Index, self).__init__(config=config,
            **kwargs)
        self.available_indices = self._get_available_tickers(mode='indices')

    def get_index_members(self, index: str='SPX') -> Optional[pd.DataFrame]:
        """get constituents and details of the indices
        :param index: takes 'SPX', 'IXIC', and 'DJI'
        """
        if index == 'SPX':
            url = 'sp500_constituent'
        elif index == 'IXIC':
            url = 'nasdaq_constituent'
        elif index == 'DJI':
            url = 'dowjones_constituent'
        else:
            raise NotImplementedError
        
        res = self._get_data(url=url)
        if isinstance(res, list):
            df = pd.concat([pd.Series(d).to_frame().T for d in res])
            df = df.set_index("symbol")
            return df
    
    @classmethod
    def get_members(cls, index: str='SPX'):
        return cls(config=DEFAULT_CONFIG
            ).get_index_members(index=index)
