# Wrapper for the Financial Modeling Prep API

version 0.0.2

## Change log

- Added multithreaded pagination for downloading shareholder lists

- Added subpackage for constructing graph from API

## Setup
- Use conda to manage enviroment.

- Clone this repository with:

```$ git clone https://github.com/pauljhp/FinancialModelingPrep```
    
```$ cd ./FinancialModelingPrep```
    
- Create a new environment with:

```$ conda create --name <env> --file <this file>```
    
or your can use the .yml file:

```$ conda env create -f environment.yml```

## Usage

- All sub modules are designed to work both as modules and as scripts
- FMP API requires an API key, you can either wrap it in a json file like this: 
  {"apikey": <`YOURAPIKEY`>}
  And save it in ./.config/config.json.  If you save it somewhere else, you'll have to specify the path when initiating the class.
- Without specifying the config.json file, the API will prompt you to enter the api key.

### 1. Ticker Class

- You can instantiate a Ticker class:
```
>>> from FinancialModelingPrep.tickers import Ticker
>>> t = Ticker(`<YOUR TICKER>`) # you either input a single ticker (case insentitive), wrap multiple tickers seperated by ",", or wrap multiple tickers in a List[str]
```

- You can also use the class methods without instantiating:

```>>> Ticker.get_stock_news(<TICKER>, <start_date>)```

- If you have specified a sqlite path, when setting save_to_sql=True, apart from returning a pd.DataFrame, the dataframe will also be written into the sql database
- You can also call classmethod Ticker().get_income_statements(ticker=`<YOUR TICKER>`) with our instantiating the class

## Structure

- 3 main classes are implemented:
  - ticker.Ticker for getting stock level information;
  - funds.Funds for getting information about funds;
  - indices.Index for getting information about indices.
- All classes inherit from the base class _abstract.AbstractAPI
- You will also find utility classes, such as utils.utils.TickerSearch, which helps with searching for companies by keywords
- There is also a graph_construction.ConstructGraph class for constructing a graph from the API

## Style guide

- Use type hints for type checks and readability
- CamelCase classes, all file and method names should be lower cases, connected by \_;
- Global variables shoudl be in UPPER CASE;
- Seperate classes with 2 empty lines; seperate method with 1 empty line;
