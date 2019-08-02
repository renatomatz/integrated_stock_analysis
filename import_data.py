"""
TODO:
- implement <Quandl_Interface> class
- implement <get_etf> function
"""

import quandl
import pandas as pd
import json
from datetime import datetime, timedelta

class Quandl_Interface:
    """Basic interface for interacting with one dataset using the Quandl API
    """

    dataset: str
    filters: dict
    data: object

    _templates: dict
    _mem_file: str
    _API_KEY: int

    def __init__(self):
        pass

    def add_template(self, name, cols):
        pass

    def remove_template(self, name):
        pass

    def get_table_template(self, template):
        pass

    def get_table_custom(self, *args, **kwargs):
        pass

    def update_filters(self, update):
        pass

    def update_dataset(self, new):
        pass

    def update_data(self):
        pass

def get_own_daily_metrics(config, key_file="key.txt"):

    return get_daily_metrics(config["ticker"], config, key_file=key_file)

def get_daily_metrics(ticker,config, key_file="key.txt"):

    _set_quandl_api_key(key_file)

    evebitda = quandl.get_table("SHARADAR/DAILY",
                                paginate = True,
                                ticker=ticker,
                                qopts={"columns": ["ticker", "date", "evebitda"]})
    
    prices = quandl.get_table("SHARADAR/SEP",
                              paginate = True,
                              ticker=ticker,
                              qopts={"columns": ["ticker", "date", "close"]})

    # evebitda = _read_and_standardize("test_files/evebitda.csv")
    # prices = _read_and_standardize("test_files/prices.csv")

    data = pd.merge(left=evebitda, right=prices, on=["ticker", "date"], how="inner").set_index("date")

    try:
        data.loc[config["date"]]
    except KeyError:
        config["end_date"] = max(data.index)
        config["start_date"] = config["end_date"] - timedelta(days=700)

    # TODO: remove the comments
    # data = data[(data.index > config["start_date"]) & (data.index <= config["end_date"])]

    return data

def get_own_fundamentals(config, key_file="key.txt"):

    _set_quandl_api_key(key_file)

    data = quandl.get_table("SHARADAR/SF1", ticker=[config["ticker"]],
                            qopts = {"columns": ["calendardate", "price", "ev", "ebitda",
                                                 "grossmargin", "revenue", "fcf", "workingcapital"]})

    # data = _read_and_standardize("test_files/own_funds.csv", col="calendardate")

    data["ev/ebitda"] = data["ev"] / data["ebitda"]
    data["ev/sales"] = data["ev"] / data["revenue"]
    data["ev/fcf"] = data["ev"] / data["fcf"]
    data["nwc_percent_sales"] = (data["workingcapital"] / data["revenue"])*100

    data = data.set_index("calendardate")

    return data

def get_own_all(config, key_file="key.txt"):
    
    _set_quandl_api_key(key_file)

    return quandl.get_table("SHARADAR/SF1", ticker[config["ticker"]])

def get_own_events(config, key_file="key.txt"):
    
    _set_quandl_api_key(key_file)

    events = quandl.get_table("SHARADAR/EVENTS", ticker=[config["ticker"]], 
                              qopts={"columns":["date", "eventcodes"]})

    # events = _read_and_standardize("test_files/own_events.csv")

    events["eventcodes"] = events["eventcodes"].map(lambda x: x.split("|"))

    return events


def get_comps(config, key_file="key.txt", config_file="config.txt", end=2019, years=5, time="close"):

    _set_quandl_api_key(key_file)

    data = quandl.get_table("SHARADAR/TICKERS",
                                paginate=True,
                                qopts={"columns": ["ticker", "name", "category",
                                                   "siccode", "scalemarketcap", "lastupdated"]})

    # data = _read_and_standardize("test_files/comps.csv", col="lastupdated")

    data["scalemarketcap"] = data["scalemarketcap"].apply(lambda x: int(x[0]) if x else None)
    # keep only scale category number
    data = data.groupby("ticker").apply(lambda x: x[x.index == max(x.index)])
    # remove name duplicates, selecting most recent
    data.index = data.droplevel(level=1)
    data = data.drop_duplicates()
    # drop second index level created from aggregation and keep uniques

    ticker_data = data[data["ticker"] == config["ticker"]]
    if len(ticker_data) == 0:
        raise ValueError("Ticker does not exist")

    ticker_cap = int(ticker_data["scalemarketcap"])

    data = data[ticker_cap-1 <= data["scalemarketcap"]]
    data = data[data["scalemarketcap"] <= ticker_cap+1]
    # keep only data of companies with similar market cap
    
    i = 1
    # such that one digit is revealed at a time
    comps = data

    while (len(comps) > 6) and (i < 5):
        # make sic code become broader until there are at least three comps or first sic code digit
        comps = data[(int(ticker_data["siccode"]) % (10**i)) == data["siccode"].apply(lambda x: x % (10**i) if x else x)]
        i += 1

    if len(comps) < 3:
        comps = data[(int(ticker_data["siccode"]) % (10**i-1)) == data["siccode"].apply(lambda x: x % (10**i-1) if x else x)]
        # if there are less than three comps, select comps of one "level" above
     
    if len(comps) > 6:
        comps = comps.iloc[0:6]
        # guarantees there will be between 3 and 6 comps

    comps = _standardize_index(comps)

    return comps

def get_comp_fundamentals(comps, key_file="key.txt"):
    
    _set_quandl_api_key(key_file)

    data = quandl.get_table("SHARADAR/SF1",
                                    paginate = True,
                                    ticker=comps,
                                    qopts={"columns": ["ticker", "calendardate", "price", "ev", 
                                                       "marketcap", "ebitda", "revenue", "fcf"]})
                                                       
    # data = _read_and_standardize("test_files/comp_metrics.csv", col=["calendardate", "ticker"])

    data["ev/ebitda"] = data["ev"] / data["ebitda"]
    data["ev/sales"] = data["ev"] / data["revenue"]
    data["ev/fcf"] = data["ev"] / data["fcf"]
    # create new metrics 

    return data
    
def get_daily_comp_metrics(comps, key_file="key.txt"):

    _set_quandl_api_key(key_file)

    comp_metrics = quandl.get_table("SHARADAR/DAILY", ticker=comps,
                                    qopts={"columns": ["ticker", "date", "evebitda"]})

    return comp_metrics

def get_comps_all(comps, key_file="key.txt"):

    _set_quandl_api_key(key_file)

    return quandl.get_table("SHARADAR/SF1", ticker=comps)

def get_etf(config, key_file="key.txt"):
    
    _set_quandl_api_key(key_file)

    # TODO implement this lookup and possibly dig deeper into the sic code thing

    data = quandl.get_table("SHARADAR/SFP", ticker=[config["etf"]],
                            qopts={"columns":["date", "close"]})

    return data

def get_sp500(key_file="key.txt"):

    _set_quandl_api_key(key_file)

    return quandl.get_table("SHARADAR/SFP", ticker="VOO",
                            qopts={"columns":["date", "close"]})

def get_config(file):

    with open(file) as f:
        return json.load(f)

def round_col(df, cols, to="M"):
    new_cols = {}
    div = 10**(9 if to == "B" else 6 if to == "M" else 3 if to == "k" else 0)

    for c in cols:
        new_cols[c] = "{} (${})".format(c, to)
        df[c] = df[c] / div

    return df.rename(columns=new_cols, inplace=False)

def _standardize_index(df):
    return df.set_index(pd.Series(range(len(df))))

def _read_and_standardize(file, col=["date"]):

    date_col = col[0] if isinstance(col, list) else col

    result = pd.read_csv(file, index_col=col,
                         converters={date_col: lambda x: datetime.strptime(x, "%Y-%m-%d")})

    if "None" in result.columns:
        del result["None"]

    return result.where(pd.notnull(result), None)

def _set_quandl_api_key(file):

    with open(file, "r") as key:
        quandl.ApiConfig.api_key = key.readline().rstrip("\n")

