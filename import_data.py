import quandl
import pandas as pd
from datetime import datetime, timedelta

def get_processed(key_file="key.txt", config_file="config.txt"):

    _set_quandl_api_key(key_file)

    config = _get_config(config_file)

    evebitda = quandl.get_table("SHARADAR/DAILY",
                                paginate = True,
                                ticker=config["ticker"],
                                qopts={"columns": ["date", "evebitda"]})

    prices = quandl.get_table("SHARADAR/SEP",
                              paginate = True,
                              ticker=config["ticker"],
                              qopts={"columns": ["date", "close"]})

    prices = prices.set_index("date")
    evebitda = evebitda.set_index("date")

    try:
        prices.loc[config["end_date"]]
        evebitda.loc[config["end_date"]]
    except KeyError:
        end_date = evebitda.index[0]
        start_date = config["end_date"] - timedelta(days=700)

    prices_mask = (prices.index > config["start_date"]) & (prices.index <= config["end_date"])
    evebitda_mask = (evebitda.index > config["start_date"]) & (evebitda.index <= config["end_date"])
    prices = prices[prices_mask]
    evebitda = evebitda[evebitda_mask]

    result = pd.DataFrame(prices, evebitda)

    return result


def get_comps(key_file="key.txt", config_file="config.txt", end=2019, years=5, time="close"):

    _set_quandl_api_key(key_file)

    config = _get_config(config_file)

    metadata = quandl.get_table("SHRADAR/TICKERS",
                                paginate=True,
                                qopts={"columns": ["ticker", "name", "category",
                                                   "siccode", "sicsector", "sicindustry",
                                                   "scalemarketcap", "lastupdated"]})

    metadata["scalemarketcap"] = metadata["scalemarketcap"].map(lambda x: int(x[0]))
    # keep only scale category number

    ticker_data = metadata[metadata["ticker"] == config["ticker"]]
    ticker_cap = ticker_data["scalemarketcap"]

    comps = metadata[ticker_cap-1 <= metadata["scalemarketcap"] <= ticker_cap+1]
    # keep only data of companies with similar market cap
    i = 1
    # such that one digit is revealed at a time

    while (len(comps) > 6) and (i < len(ticker_data["siccode"])):
        # make sic code become broader until there are at least three comps or first sic code digit
        comps = comps[ticker_data["siccode"][:i] in comps["siccode"]]
        i += 1

    if len(comps) < 3:
        comps = metadata[ticker_data["siccode"][:i-1] in metadata["siccode"]]
        # if there are less than three comps, select comps of one "level" above
     
    if len(comps) > 6:
        comps = comps.iloc[0:6]
        # guarantees there will be between 3 and 6 comps


    comps_metrics = quandl.get_table("SHARADAR/DAILY",
                                     paginate = True,
                                     ticker=[*comps["ticker"]],
                                     qopts={"columns": ["ticker", "date", "ev", "marketcap", "evebit", "evebitda"]})

    comps_merged = pd.merge(left=comps, right=comps_metrics, 
                            left_on=["ticker", "lastupdated"], right_on=["ticker", "date"], how="left")

    comps_merged = comps_merged[["name", "ticker", "ev", "marketcap", "evebit", "evebitda"]]

    return comps_merged
    

def get_press_releases():
    pass

def _set_quandl_api_key(file):

    with open(file, "r") as key:
        quandl.ApiConfig.api_key = key.readline()

def _get_config(file):

    config = {}

    with open(file, "r") as config:
        config["company"] = config.readline()[8:-1]
        config["ticker"] = config.readline()[7:-1]
        config["end_date"] = datetime.strptime(config.readline()[12:-1], "%Y-%m-%d")
        config["start_date"] = config["end_date"] - timedelta(days=700)

    return config


