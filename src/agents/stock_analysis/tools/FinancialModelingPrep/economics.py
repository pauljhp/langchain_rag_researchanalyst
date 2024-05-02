from __future__ import annotations
from urllib.parse import urljoin
from copy import deepcopy
import pandas as pd
from typing import Optional, Union, List, Dict, Callable
from collections import deque
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
from argparse import ArgumentParser
from pathlib import Path
from ._abstract import AbstractAPI
from .utils.config import Config
from .utils.utils import pandas_strptime, iter_by_chunk
import numpy as np
import logging


LOGPATH = './FinancialModelingPrep/.log/'
LOGFILE = os.path.join(LOGPATH, 'log.log')

if not os.path.exists(LOGPATH):
    os.makedirs(LOGPATH)

logging.basicConfig(filename=LOGFILE, 
    # encoding='utf-8', 
    level=logging.DEBUG)

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
DEFAULT_START_DATE = dt.date(2020, 1, 1)
DEFAULT_SQL_PATH = './FinancialModelingPrep/data/economics.db'

config_p = Path(DEFAULT_CONFIG)
if not config_p.exists():
    config_p.parent.mkdir(parents=True, exist_ok=True)
    config_p.write_text(r"""{{"apikey": "{a}"}}""".format(a=input("the config file wasn't found - enter your apikey: ")))

sql_p = Path(DEFAULT_SQL_PATH)
if not sql_p.exists():
    print("creating sql file")
    sql_p.parent.mkdir(parents=True, exist_ok=True)
    sql_p.touch()

class Economics(AbstractAPI):

    def __init__(self, 
        config: Union[str, Config]=DEFAULT_CONFIG,
        sql_path: Optional[str]=DEFAULT_SQL_PATH):
        super(Economics, self).__init__(
            config=config,
            version='v4',
            sql_path=sql_path)
        self.all_fields = [
            'GDP', 
            'realGDP', 
            'nominalPotentialGDP',
            'realGDPPerCapita',
            'federalFunds',
            'CPI',
            'inflationRate', 
            'inflation',
            'retailSales',
            'consumerSentiment', 
            'durableGoods', 
            'unemploymentRate',
            'totalNonfarmPayroll', 
            'initialClaims',
            'industrialProductionTotalIndex',
            'newPrivatelyOwnedHousingUnitsStartedTotalUnits',
            'totalVehicleSales',
            'retailMoneyFunds', 'smoothedUSRecessionProbabilities', 
            '3MonthOr90DayRatesAndYieldsCertificatesOfDeposit', 
            'commercialBankInterestRateOnCreditCardPlansAllAccounts',
            '30YearFixedRateMortgageAverage',
            '15YearFixedRateMortgageAverage'
        ]

    def __get_data(self, field: str,
        start_date: Union[str, dt.date],
        end_date: Union[str, dt.date]):
        if isinstance(start_date, dt.date): start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, dt.date): end_date = end_date.strftime("%Y-%m-%d")
        return self._get_data(url="economic",
            additional_params={'from': start_date,
                'end': end_date},
            name=field)

    def get_data(self, field: str,
        start_date: Union[str, dt.date]=DEFAULT_START_DATE,
        end_date: Union[str, dt.date]=TODAY,
        freq: Optional[str]=None,
        aggfunc: Callable=np.mean):
        data = self.__get_data(field, start_date, end_date)
        data = pd.DataFrame(data, columns=['date', 'value'])
        data.columns = ['date', field]
        data = pandas_strptime(data, 
            index_name='date', axis=1)
        if freq:
            try:
                data.date = data.date.dt.to_period(freq)
                data = data.groupby(by='date').apply(aggfunc, axis=0, dtype='numeric')
            except Exception as e:
                logging.error(e)
        data = data.set_index("date")
        return data

    @classmethod
    def get_all_data(cls, 
        start_date: Union[str, dt.date]=DEFAULT_START_DATE, 
        end_date: Union[str, dt.date]=TODAY,
        max_workers: int=8,
        freq: str='M',
        to_sql: bool=False):
        res = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(cls(DEFAULT_CONFIG).get_data, 
                    field, start_date, end_date, freq=freq)
                for field in cls(DEFAULT_CONFIG).all_fields]
        for future in as_completed(futures):
            data = future.result()
            res.append(data)
        df = pd.concat(res)
        df = df.groupby(by='date').mean()
        if to_sql:
            export_df = df.copy(deep=True)
            export_df.index = export_df.index.astype(str)
            cls(config=DEFAULT_CONFIG, sql_path=DEFAULT_SQL_PATH).\
                pandas_to_sql(
                    export_df, 
                    table_name=f'economic_indicators_{start_date}-{end_date}_{freq}',
                    if_exists='append')
        return df


def main(start_date: Union[str, dt.date]=DEFAULT_START_DATE, 
        end_date: Union[str, dt.date]=TODAY,
        max_workers: int=8,
        freq: str='M',
        to_sql: bool=False):
        return Economics.get_all_data(start_date,
            end_date,
            max_workers,
            freq,
            to_sql=to_sql)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", "--start_date", type=str,
        help="start date")
    parser.add_argument("-e", "--end_date", type=str,
        help="end date")
    parser.add_argument("-m", "--max_workers", type=int,
        default=8,
        help="frequency")
    parser.add_argument("-f", "--freq", type=str,
        help="frequency")
    parser.add_argument("-sq", "--to_sql", action="store_true")
    
    args = parser.parse_args()

    main(args.start_date, args.end_date, args.max_workers, args.freq, args.sq)