from __future__ import annotations
from urllib.parse import urljoin
from copy import deepcopy
import pandas as pd
from typing import (
    Optional, Union, List, Dict, Callable, Sequence
    )
from collections import deque
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
from argparse import ArgumentParser
from pathlib import Path
import math
from ._abstract import AbstractAPI
from .utils.config import Config
from .utils.utils import pandas_strptime, iter_by_chunk


# DEFAULT_CONFIG = "./FinancialModelingPrep/.config/config.json"
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
DEFAULT_START_DATE = dt.date(2020, 1, 1)

# config_p = Path(DEFAULT_CONFIG)
# if not config_p.exists():
#     config_p.parent.mkdir(parents=True, exist_ok=True)
#     config_p.write_text(r"""{{"apikey": "{a}"}}""".format(
#     a=input("the config file wasn't found - enter your apikey: "))
#     )

class Ticker(AbstractAPI):

    def __init__(self, ticker: Optional[Union[str, List[str]]]=None, 
        # config: Union[str, Callable, Config]=DEFAULT_CONFIG, 
        mode: str='statements',
        ignore_unavailable_tickers: bool=False,
        **kwargs):
        super(Ticker, self).__init__(**kwargs)
        self.available_tickers = self._get_available_tickers(mode=mode)
        if isinstance(ticker, str):
            if "," in ticker: 
                tickers = ticker.upper().split(",")
                tickers = [t.strip() for t in tickers]
                if not ignore_unavailable_tickers:
                    assert len(tickers) == sum([t in self.available_tickers 
                        for t in tickers]), \
                        f"All tickers must be available! These are not valid tickers: {' '.join([t for t in tickers if t not in self.available_tickers])}"
                self.tickers = tickers
            else:
                if not ignore_unavailable_tickers:
                    assert ticker.upper().strip() in [t.upper() for t in self.available_tickers], "Not a valid ticker!"
                self.tickers = [ticker.upper()]
        elif isinstance(ticker, list):
            if not ignore_unavailable_tickers:
                assert len(ticker) == sum([str(t).upper().strip() in self.available_tickers 
                        for t in ticker]), \
                        f"All tickers must be available! These are not valid tickers: {' '.join([t for t in ticker if t not in self.available_tickers])}"
            self.tickers = [str(t).upper() for t in ticker]
        elif ticker is None: # special instantiation without ticker only allowed for using the all_company_profiles class method
            self.classmethod_mode = True
            self.tickers = ""
            warnings.warn("Ticker unspecified. Only get_all_company_profiles method will work in this mode")
        else:
            raise TypeError("ticker must be a string or a list of strings")
        self.tickers_str = ",".join(self.tickers)

    def __get_statements(self, statement: str='income', 
        freq: str="A", 
        save_to_sql: bool=False, 
        limit: int=100) -> Optional[pd.DataFrame]:
        """interface for getting income/balance sheet/cash flow statements
        :param statement: takes 'income', 'balance_sheet', 'cashflow'
        :param freq: takes 'A' or 'Q'
        """
        params = deepcopy(self._default_params)
        params.update(dict(limit=limit))
        endpoints = dict(income='income-statement/', 
            balance_sheet='balance-sheet-statement/',
            cashflow='cash-flow-statement/')
        if freq == "A":
            res = self._get_data(url=urljoin(endpoints.get(statement),
                f"{','.join(self.tickers)}/"),
                limit=limit)
        elif freq == 'Q':
            res = self._get_data(url=urljoin(endpoints.get(statement),
                f"{','.join(self.tickers)}/"),
                period='quarter', limit=limit)
        else:
            raise NotImplementedError
        if isinstance(res, list):
            ls = []
            for entry in res:
                s = pd.Series(entry)
                ls.append(s.to_frame().T)
            df = pd.concat(ls).T
            df = df.T.set_index(['date', 'symbol', 'reportedCurrency', 'cik', 
                'fillingDate', 'acceptedDate', 'calendarYear', 'period']).T
            df = df.stack(['symbol', 'cik']).swaplevel(0, 2)
            if save_to_sql:
                assert self.sql_conn is not None, "sql_path must be specified if you want to use 'save_to_sql'"
                start = f"{df.columns.get_level_values('calendarYear')[0]}\
                    {df.columns.get_level_values('period')[0]}"
                end = f"{df.columns.get_level_values('calendarYear')[-1]}\
                    {df.columns.get_level_values('period')[-1]}"
                tablename = f"{'_'.join(self.ticker)}_{statement}_{freq}_{start}_{end}"
                df.to_sql(tablename, 
                    self._cur, if_exists="replace")
            return df
        else:
            raise TypeError("value returned from API is not a list")

    def income_statements(self, 
        freq: int="A", 
        save_to_sql: bool=False, 
        limit: int=100) -> Optional[pd.DataFrame]:
        """get income statement
        :param freq: takes 'A' or 'Q'
        """
        return self.__get_statements(statement='income', 
            freq=freq, save_to_sql=save_to_sql, limit=limit)

    @classmethod
    def get_income_statements(cls, 
        tickers: Union[str, Sequence[str]], 
        # config: Union[str, Callable, Config]=DEFAULT_CONFIG, 
        freq: int='A',
        limit: int=100):
        """classmethod version of `income_statement`. 
        Doesn't allow save_to_sql"""
        return cls(ticker=tickers, 
                #    # config=config
                   )\
            .income_statements(freq=freq, limit=limit)

    def balance_sheet(self, 
        freq: str="A", 
        save_to_sql: bool=False, 
        limit: int=100) -> Optional[pd.DataFrame]:
        """get balance sheet statement
        :param freq: takes 'A' or 'Q'
        """
        return self.__get_statements(statement='balance_sheet', 
            freq=freq, save_to_sql=save_to_sql, limit=limit)

    @classmethod
    def get_balance_sheet(cls,
        tickers: Union[str, Sequence[str]],
        # config: Union[str, Callable, Config]=DEFAULT_CONFIG,
        freq="A", 
        limit: int=100
        ):
        """classmethod version of `balance_sheet`"""
        return cls(ticker=tickers, # config=config
                   )\
            .income_statements(freq=freq, limit=limit)

    def cashflow(self, 
        freq="A", 
        save_to_sql: bool=False,
        limit: int=100) -> Optional[pd.DataFrame]:
        """get cash flow statement
        :param freq: takes 'A' or 'Q'
        """
        return self.__get_statements(statement='cashflow', 
            freq=freq, save_to_sql=save_to_sql, limit=limit)
    
    @classmethod
    def get_cashflow(cls,
        tickers: Union[str, Sequence[str]],
        # config: Union[str, Callable, Config]=DEFAULT_CONFIG,
        freq="A", 
        limit: int=100
        ):
        """classmethod version of `cashflow`"""
        return cls(ticker=tickers, # config=config
                   )\
            .cashflow(freq=freq, limit=limit)
    
    def product_segments(self, freq='A') -> Union[Dict, List]:
        """get the product segments for the ticker
        :param freq: takes 'A' or 'Q'

        Note: This temporarily resets the endpoint to v4 and sets it back to 
        v3 after the function call ends. This may cause issues when using other 
        functions with this concurrently.
        """
        # TODO - fix concurrency issue cause by the temporary endpoint reset
        endpoint = self.endpoint
        self.endpoint = endpoint.replace("v3", "v4")
        url = "revenue-product-segmentation/"
        if freq == 'A':
            d = self._get_data(url=url, ticker=",".join(self.tickers))
        elif freq == 'Q':
            d = self._get_data(url=url, ticker=",".join(self.tickers), 
                period='quarter')
        else:
            self.endpoint = endpoint
            raise NotImplementedError
        self.endpoint = endpoint # set endpoint back to v3
        return d

    @classmethod
    def get_product_segments(cls, ticker: Union[str, List[str]], 
        # config: Union[str, Config]=DEFAULT_CONFIG,
        freq: str='A'):
        """classmethod version of self.product_segments()"""
        return cls(ticker=ticker, # config=config
                   ).product_segments(freq=freq)

    def geo_segments(self, freq='A', **kwargs) -> Union[Dict, List]:
        """get the geographical segments for the ticker
        :param freq: takes 'A' or 'Q'

        Note: This temporarily resets the endpoint to v4 and sets it back to 
        v3 after the function call ends. This may cause issues when using other 
        functions with this concurrently.
        """
        # TODO - fix concurrency issue cause by the temporary endpoint reset
        endpoint = self.endpoint
        self.endpoint = endpoint.replace("v3", "v4")
        url = "revenue-geographic-segmentation/"
        if freq == 'A':
            d = self._get_data(url=url, ticker=",".join(self.tickers), **kwargs)
        elif freq == 'Q':
            d = self._get_data(url=url, ticker=",".join(self.tickers), 
                period='quarter', **kwargs)
        else:
            self.endpoint = endpoint
            raise NotImplementedError
        self.endpoint = endpoint # set endpoint back to v3
        return d

    @classmethod
    def get_geo_segments(cls, ticker: Union[str, List[str]], 
        # config: Union[str, Config]=DEFAULT_CONFIG,
        freq: str='A', structure='flat', **kwargs
        ):
        """classmethod version of geo_segments"""
        return cls(ticker=ticker, # config=config).geo_segments(freq=freq, 
            structure='flat', **kwargs)

    def transcripts(self, year: int, quarter: Optional[int]=None):
        """get earnings call transcript
        :param year: year of the earnings call
        :param quarter: takes 1, 2, 3, 4
        """
        if quarter:
            url = urljoin("earning_call_transcript/", ",".join(self.tickers))
            assert quarter in range(1, 5), "quarter must be between 1 and 4"
            return self._get_data(url=url, year=year, quarter=quarter)
        endpoint = self.endpoint
        self.endpoint = endpoint.replace("v3", "v4")
        url = urljoin("batch_earning_call_transcript/", ",".join(self.tickers))
        res = self._get_data(url=url, year=year)
        self.endpoint = endpoint
        return res

    @classmethod
    def get_transcripts(cls, ticker: str, 
        year: int, quarter: Optional[int]=None):
        """classmethod version of get_transcripts. Takes the same arguments
        To use this, you must make sure the apikey is saved through:
            ./FinancialModelingPrep/.config/config.json
        Otherwise the api will not return
        """
        return cls(ticker, 
                #    DEFAULT_CONFIG
                   ).transcripts(year=year, quarter=quarter)

    def transcript_dates(self):
        """Only accepts one ticker"""
        url = "https://financialmodelingprep.com/api/v4/earning_call_transcript"
        res = self._get_data(url=url, ticker=self.tickers[0])
        df = pd.DataFrame(res, columns=["quarter", "year", "date"])
        df["date"] = df["date"].astype("datetime64[ns]")
        return df
    
    @classmethod
    def get_transcript_dates(cls, ticker: str):
        return cls(ticker, 
                #    DEFAULT_CONFIG
                   ).transcript_dates()


    def inst_ownership(self, incl_cur_q: bool=True, 
        save_to_sql: bool=False,):
        """get number of shares held by institutional shareholders disclosed 
        through 13F
        :param incl_cur_q: Include current Q or not
        """
        # TODO - solve concurrency issue caused by the temporary endpoint reset
        endpoint = self.endpoint
        self.endpoint = endpoint.replace("v3", "v4")
        url = "institutional-ownership/symbol-ownership"
        res = self._get_data(url=url, ticker=",".join(self.tickers),
            includeCurrentQuarter=incl_cur_q)
        if isinstance(res, list):
            ls = []
            for entry in res:
                s = pd.Series(entry)
                ls.append(s.to_frame().T)
            df = pd.concat(ls).T
            df = df.T.set_index(['date', 'symbol', 'cik',]).T
            df = df.stack(['symbol', 'cik']).swaplevel(0, 2)
            if save_to_sql:
                start = f"{df.columns.get_level_values('date')[0]}"
                end = f"{df.columns.get_level_values('date')[-1]}"
                tablename = f"{'_'.join(self.ticker)}_ownership_{start}_{end}"
                df.to_sql(tablename, 
                    self._cur, if_exists="replace")
            return df
        else:
            raise TypeError("value returned from API is not a list")

    @classmethod
    def get_inst_ownership(cls, ticker: str, incl_cur_q: bool=True):
        """classmethod version of get_ownership"""
        return cls(ticker=ticker, 
            # config=DEFAULT_CONFIG
            ).inst_ownership(incl_cur_q=incl_cur_q)

    def inst_owners(self, year: int=CUR_YEAR,
        quarter: int=LAST_Q,
        save_to_sql: bool=False,
        max_workers: int=8) -> Optional[pd.DataFrame]:
        """get number of shares held by institutional shareholders disclosed 
        through 13F
        :param incl_cur_q: Include current Q or not
        """
        # TODO - solve concurrency issue caused by the temporary endpoint reset
        endpoint = self.endpoint
        self.endpoint = endpoint.replace("v3", "v4")
        url = "institutional-ownership/institutional-holders/symbol-ownership-percent"
        def get_page(page: int=0):
            month, day = QUARTER_END.get(quarter)
            date = dt.date(year, month, day).strftime("%Y-%m-%d")
            res = self._get_data(url=url, ticker=",".join(self.tickers),
                page=page,
                date=date)
            if res: return res
        page, i = 1, 0
        res = []
        
        if max_workers > 1:
            with ThreadPoolExecutor() as executor:
                while page:
                    futures = [executor.submit(get_page, p) 
                        for p in range(i, i + max_workers)]
                    for future in as_completed(futures):
                        page = future.result()
                        if isinstance(page, list):
                            res += page
                    i += max_workers
        else:
            while page:
                page = get_page(i)
                if isinstance(page, list): res += page
                i += 1

        if isinstance(res, list):
            ls = []
            for entry in res:
                s = pd.Series(entry)
                ls.append(s.to_frame().T)
            df = pd.concat(ls).T
            df = df.T.set_index(['date', 'symbol', 'cik',]).T
            # df = df.stack(['symbol', 'cik']).swaplevel(0, 2)
            return df
        else:
            return res

        if save_to_sql:
            start = f"{df.columns.get_level_values('date')[0]}"
            end = f"{df.columns.get_level_values('date')[-1]}"
            tablename = f"{'_'.join(self.ticker)}_ownership_{start}_{end}"
            df.to_sql(tablename, 
                self._cur, if_exists="replace")
            return df
        else:
            raise TypeError("value returned from API is not a list")

    @classmethod
    def get_inst_owners(cls, ticker: str, year: int=CUR_YEAR, 
        quarter: int=LAST_Q, max_workers: int=8):
        """classmethod version of self.inst_owners()"""
        return cls(ticker=ticker, 
            # config=DEFAULT_CONFIG
            ).inst_owners(year=year, 
                quarter=quarter, max_workers=max_workers)

    def __get_v4_info(self, url: str) -> Dict:
        """template function for getting v4 info"""
        endpoint = self.endpoint
        self.endpoint = endpoint.replace("v3", "v4")
        res = self._get_data(url=url, ticker=",".join(self.tickers))
        self.endpoint = endpoint
        return res

    def peers(self) -> List[str]:
        """get the stock's peers"""
        url = "stock_peers"
        res = self.__get_v4_info(url=url)
        return res

    @classmethod
    def get_peers(cls, ticker: Union[str, List[str]]) -> List[str]:
        """classmethod version of get_peers"""
        res = cls(ticker=ticker, 
            # config=DEFAULT_CONFIG
            ).peers()
        return res

    def core_info(self) -> Dict:
        """get the stock's core information such as cik, exchange, industry"""
        url = "company-core-information"
        res = self.__get_v4_info(url=url)
        return res

    @classmethod
    def get_core_info(cls, ticker: Union[str, List[str]], 
        # config: Union[str, Config]=DEFAULT_CONFI
        ):
        """classmethod version of self.core_info()"""
        return cls(ticker=ticker, # config=config
                   ).core_info()

    def company_profile(self):
        """get company's profile information"""
        url = urljoin("profile/", ",".join(self.tickers))
        res = self._get_data(url=url)
        if isinstance(res, list):
            df = pd.concat([pd.Series(s).to_frame().T for s in res])
            df = df.set_index(["symbol"])
            return df
        else: return res

    @classmethod
    def get_company_profile(cls, ticker: Union[str, List[str]]) -> Dict[str, float]:
        """classmethod version of get_profile"""
        res = cls(ticker=ticker, 
            # config=DEFAULT_CONFIG
            ).company_profile()
        return res

    @classmethod
    def get_all_company_profiles(cls, tickers=None,
        limit: int=100000, 
        max_workers=16,
        sql_path: Optional[str]=None) -> pd.DataFrame:
        """returns dataframe containing all the companies up to a limit
        :param limit: max number of tickers to be sent at once
        :param max_workers: max workers for concurrent operation if limmit is 
            set to higher than 1000
        """
        max_len = 1000 # FinancialModelingPrep takes max 1000 tickers at a time
        available_tickers = cls().available_tickers
        if limit <= max_len:
            tickers = ",".join(available_tickers[:limit])
            res = cls(ticker=tickers, 
                # config=DEFAULT_CONFIG, 
                ignore_unavailable_tickers=True).company_profile()
            
        else:
            fn = lambda ls: cls(ticker=",".join(ls), 
                # config=DEFAULT_CONFIG,
                ignore_unavailable_tickers=True).company_profile()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(fn, chunk) 
                for chunk in iter_by_chunk(available_tickers[:limit], max_len)]
                res = pd.concat([future.result() for future in futures])
        
        if sql_path:
            cls(ticker=tickers, 
                # config=DEFAULT_CONFIG, 
                ignore_unavailable_tickers=True,
                sql_path=sql_path).pandas_to_sql(res, table_name='all_company_profiles')
        return res

    def list_execs(self) -> Union[pd.DataFrame, Dict]:
        """get list of key executives, their positions and bios"""
        url = urljoin("key-executives/", ",".join(self.tickers))
        res = self._get_data(url=url)
        if isinstance(res, list):
            df = pd.concat(pd.Series(d).to_frame().T for d in res)
            df.index = range(df.shape[0])
            return df
        else: return res
    
    @classmethod
    def get_list_execs(cls, ticker: Union[str, List[str]]):
        """classmethod version of `get_execs`"""
        return cls(ticker=ticker, 
            # config=DEFAULT_CONFIG
            ).list_execs()

    def financial_ratios(self, limit: int=10, 
        freq: str="A") -> Union[pd.DataFrame, Dict]:
        """get financial ratios in the statements
        :param limit: number of period going back
        :param freq: takes 'A' or 'Q'
        """
        url = urljoin("ratios/", self.tickers_str)
        if freq == 'A':
            res = self._get_data(url=url)
        elif freq == 'Q':
            res = self._get_data(url=url, period='quarter')
        else:
            raise NotImplementedError
        if isinstance(res, list):
            df = pd.concat([pd.Series(d).to_frame().T for d in res])
            df = pandas_strptime(df, index_name='date', axis=1)
            df = df.set_index(["symbol", "date", "period"])
            return df
        else:
            return res

    @classmethod
    def get_financial_ratios(cls, ticker: str, 
        limit: int=10, freq: str='A') -> Union[pd.DataFrame, list]:
        """classmethod version of get_financial_ratios"""
        return cls(ticker=ticker, 
            # config=DEFAULT_CONFIG
            ).financial_ratios(limit=limit, freq=freq)

    def key_metrics(self, limit: int=10, 
        freq: int='A') -> Union[pd.DataFrame, Dict]:
        """get the key metrics such as key financial ratios and valuation
        :param limit: going back how many period 
        :param freq: takes 'Q' or 'A'
        """
        url = urljoin("key-metrics/", self.tickers_str)
        if freq == 'A':
            res = self._get_data(url, limimt=limit)
        elif freq == 'Q':
            res = self._get_data(url, period='quarter', limit=limit)
        else:
            raise NotImplementedError
        if isinstance(res, list):
            df = pd.concat([pd.Series(d).to_frame().T for d in res])
            df = df.set_index(["symbol", "date", "period"])
            return df
        else:
            return res
    
    @classmethod
    def get_key_metrics(cls, ticker:str, limit: int=10,
        freq: int='A') -> Union[pd.DataFrame, Dict]:
        """classmethod version of get_key_metrics"""
        return cls(ticker=ticker, 
            # config=DEFAULT_CONFIG
            ).key_metrics(limit=limit, freq=freq)
    
    def financial_growth(self, limit: int=10, 
        freq: int='A') -> Union[pd.DataFrame, Dict]:
        """get the growth of key fundamental measures
        :param limit: going back how many period 
        :param freq: takes 'Q' or 'A'
        """
        url = urljoin("financial-growth/", self.tickers_str)
        if freq == 'A':
            res = self._get_data(url, limimt=limit)
        elif freq == 'Q':
            res = self._get_data(url, period='quarter', limit=limit)
        else:
            raise NotImplementedError
        if isinstance(res, list):
            df = pd.concat([pd.Series(d).to_frame().T for d in res])
            df = pandas_strptime(df, index_name='date', axis=1)
            df = df.set_index(["symbol", "date", "period"])
            return df
        else:
            return res

    @classmethod
    def get_financial_growth(cls, ticker:str, limit: int=10,
        freq: int='A') -> Union[pd.DataFrame, Dict]:
        """classmethod version of get_financial_growth"""
        return cls(ticker=ticker, 
            # config=DEFAULT_CONFIG
            ).financial_growth(limit=limit, freq=freq)
    
    def current_price(self):
        """get current quote price"""
        url = urljoin("quote/", self.tickers_str)
        res = self._get_data(url)
        if isinstance(res, list):
            df = pd.concat([pd.Series(d).to_frame().T for d in res])
            return df

    @classmethod
    def get_current_price(cls, ticker: Union[str, List[str]]):
        """classmethod version of current_price"""
        return cls(ticker=ticker, 
            # config=DEFAULT_CONFIG
            ).current_price()
    
    def _historical_price(self,
        start_date: Union[str, dt.date],
        end_date: Union[str, dt.date], 
        ticker: Optional[str]=None,
        freq='d'):
        """get the historical price of the tickers. 
        number of tickers need be be <= 5
        :param start_date: either "%Y-%m-%d" or dt.date format
        :param end_date: either "%Y-%m-%d" or dt.date format
        :param ticker: optional, if specicified, will overwrite self.ticker
        :param freq: takes 'd', '1hour', '30min', '15min', '5min', '1min'
        """
        if isinstance(start_date, dt.date): start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, dt.date): end_date = end_date.strftime("%Y-%m-%d")
        assert isinstance(start_date, str) and isinstance(end_date, str), f"only str and dt.date accepted for start_date and end_date, you entered {type(start_date)} and {type(end_date)}"
        ticker = ticker if ticker else self.tickers_str
        if freq == 'd':
            url = urljoin(f"historical-price-full/", ticker)
            res = self._get_data(url, 
            additional_params={'from': start_date, 
                "to": end_date})
            if len(self.tickers) <= 1: # single ticker mode
                if isinstance(res, dict):
                    # tickers = res.get('symbol')
                    if "historical" in res.keys():
                        df = pd.concat([pd.Series(d).to_frame().T for d 
                            in res.get('historical') if len(d) > 0])
                        df = df.set_index('date') # FIXME - this will not work with multiple ticker queries
                        return df
                else: return res
            else:
                if isinstance(res, dict):
                    if "historicalStockList" in res.keys():
                        res = res.get("historicalStockList")
                        ls = []
                        for r in res:
                            if "historical" in r.keys():
                                symbol = r.get("symbol")
                                df = pd.concat([pd.Series(d).to_frame().T for d 
                                    in r.get('historical') if len(d) > 0])
                                df = df.set_index('date') # FIXME - this will not work with multiple ticker queries
                                df.columns = pd.MultiIndex.from_tuples([(symbol, c) for c in df.columns])
                                ls.append(df.T)
                        res = pd.concat(ls).T
                    elif "historical" in res.keys():
                        symbol = res.get("symbol")
                        res = pd.concat([pd.Series(d).to_frame().T for d 
                                    in res.get('historical') if len(d) > 0])
                        res = res.set_index("date")
                        res.columns = pd.MultiIndex.from_tuples([(symbol, c) for c in res.columns])
                return res

            
        elif freq in ['1hour', '30min', '15min', '5min', '1min']:
            url = urljoin(f"historical-chart/{freq}/", ticker)
            res = self._get_data(url, 
            additional_params={'from': start_date, 
                "to": end_date})
            if isinstance(res, dict):
                # tickers = res.get('symbol')
                df = pd.concat([pd.Series(d).to_frame().T 
                    for d in res.get('historical')])
                df = df.set_index('date') # FIXME - this will not work with multiple ticker queries
                return df
            else: return res
        else:
            raise NotImplementedError

    def historical_price(self, 
        start_date: Union[str, dt.date],
        end_date: Union[str, dt.date], 
        freq='d',
        max_workers: int=8) -> pd.DataFrame:
        """extension of _historical_price to allow > 5 tickers"""
        res = []
        if len(self.tickers) >= 5:
            batch_size = 5 * max_workers
            for batch in iter_by_chunk(self.tickers, batch_size):
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(
                        self._historical_price, start_date=start_date, 
                        end_date=end_date, ticker=",".join(chunk), freq=freq)
                        for chunk in iter_by_chunk(batch, 5)] # FIXME - change to classmethod
                    for future in as_completed(futures):
                        res.append(future.result())
            res = pd.concat([d.T for d in res]).T
        else:
            res = self._historical_price(start_date, end_date, self.tickers_str, freq)
        return res
                

    @classmethod
    def get_historical_price(cls, ticker: Union[str, List[str]],
        start_date: Union[str, dt.date],
        end_date: Union[str, dt.date], 
        freq='d',
        **kwargs):
        """classmethod version of historical_price"""
        return cls(ticker=ticker, 
            # config=DEFAULT_CONFIG,
            **kwargs)\
                .historical_price(
                    start_date=start_date, 
                    end_date=end_date, 
                    freq=freq)

    def stock_news(self, limit: Optional[int]=None, 
        start_date: Union[dt.date, str]=DEFAULT_START_DATE):
        """get list of news"""
        max_entries = 50000
        url = 'stock_news'
        if isinstance(start_date, str): start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
        if not limit:
            res = self._get_data(url=url, tickers=self.tickers, limit=1000) # note the API weirdly takes a tickers parameter, not our usual symbol paramter. This will wrap it into kwargs
            earliest_date = dt.datetime.strptime(res[-1].get('publishedDate'),
                "%Y-%m-%d %H:%M:%S")
            if earliest_date > start_date:
                ratio = (TODAY - start_date).days \
                    / (TODAY - earliest_date).days 
                new_lim = min(max_entries, round(ratio * 1000))
                res = self._get_data(url=url, tickers=self.tickers, limit=new_lim)
        else:
            res = self._get_data(url=url, tickers=self.tickers, limit=limit)
        if isinstance(res, list):
            df = pd.concat([pd.Series(d).to_frame().T for d in res])
            df = pandas_strptime(df, index_name="publishedDate", axis=1, 
                datetime_format="%Y-%m-%d %H:%M:%S") 
            df = df.set_index(['symbol', 'publishedDate'])
            return df

    @classmethod
    def get_stock_news(cls, ticker: Union[str, List[str]],
        limit: Optional[int]=None, 
        start_date: Union[dt.date, str]=DEFAULT_START_DATE):
        """classmethod version of stock_news
        :param ticker: takes single ticker or list of tickers. Strongly suggest 
            using single ticker if querying for long range data
        """
        return cls(ticker=ticker, 
            # config=DEFAULT_CONFIG
            ).stock_news(start_date=start_date, 
                limit=limit)

    def earnings_surprises(self) -> Optional[pd.DataFrame]:
        """get earnings surprise data"""
        res = self._get_data(url=f'earnings-surprises/{self.tickers_str}')
        if isinstance(res, list):
            df = pd.concat([pd.Series(d).to_frame().T for d in res])
            df = pandas_strptime(df, axis=1, index_name="date")
            df = df.set_index("date")
            df.loc[:, "delta"] = df.actualEarningResult / df.estimatedEarning - 1
            # 
            return df
    
    @classmethod
    def get_earnings_surprises(
        cls, tickers: Union[str, List[str]]
        ) -> Optional[pd.DataFrame]:
        return cls(ticker=tickers, 
            # config=DEFAULT_CONFIG
            ).earnings_surprises()

def main(ticker: Optional[Union[str, List[str]]]=None, 
        action: str='get_all_company_profiles',
        sql_path: Optional[str]=None):
    exec(f"Ticker.{action}(tickers={ticker}, sql_path='{sql_path}')")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-a", "--action", type=str,
        default="get_all_company_profiles")
    parser.add_argument("-t", "--ticker", type=str, default=None)
    parser.add_argument("-s", "--sql_path", type=str)
    parser.add_argument("-k", "--apikey", type=str)
    args = parser.parse_args()
    # config_p = Path(DEFAULT_CONFIG)
    # if not config_p.exists():
    #     config_p.parent.mkdir(parents=True, exist_ok=True)
    #     config_p.write_text(r"""\"\{"apikey": "{a}"\}\"""".format(a=args.apikey))

    main(args.ticker, args.action, args.sql_path)