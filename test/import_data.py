import quandl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import pdb

def get_processeda():

    with open("key.txt", "r") as key:
        quandl.ApiConfig.api_key = key.readline()

    with open("config.txt", "r") as config:
        company = config.readline()[8:-1]
        ticker = config.readline()[7:-1]
        end_date = datetime.strptime(config.readline()[12:-1], "%Y-%m-%d")
        start_date = end_date - timedelta(days=700)

    evebitda = quandl.get_table("SHARADAR/DAILY",
                                ticker=ticker,
                                qopts={"columns": ["date", "evebitda"]})

    prices = quandl.get_table("SHARADAR/SEP",
                              ticker=ticker,
                              qopts={"columns": ["date", "close"]})

    prices = prices.set_index("date")
    evebitda = evebitda.set_index("date")

    try:
        prices.loc[end_date]
        evebitda.loc[end_date]
    except KeyError:
        end_date = evebitda.index[0]
        start_date = end_date - timedelta(days=700)

    prices_mask = (prices.index > start_date) & (prices.index <= end_date)
    evebitda_mask = (evebitda.index > start_date) & (evebitda.index <= end_date)
    prices = prices[prices_mask]
    evebitda = evebitda[evebitda_mask]

    result = pd.DataFrame(prices, evebitda)

    return result


def get_all_prices():
    pass
    

def get_press_releases():
    pass
