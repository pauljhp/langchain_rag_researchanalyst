from .FinancialModelingPrep.tickers import Ticker
from langchain.tools import tool
import pandas as pd


class FMPTools:

    @tool("FMP Income Statements")
    def income_statements(ticker: str) -> pd.DataFrame:
        """helpful for getting the historical income statement of a company
        :param ticker: str
        :returns: pd.DataFrame
        """
        try:
            statement = Ticker.get_income_statements(tickers=ticker, freq="A")
            return statement
        except Exception as e:
            return f"An error has happened - this is the error message: {e}"