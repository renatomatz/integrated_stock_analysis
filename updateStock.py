import quandl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import pdb

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

with open("price.txt", "w") as f:
    f.write(str(float(prices.iloc[0])))

# plt.style.use("ggplot")

ax1 = plt.subplot(211)
plt.plot(prices, color="green", linewidth=0.7, alpha=0.7)
plt.annotate(str(float(prices.loc[end_date])),
             (end_date, prices.loc[end_date]))
plt.ylabel("Stock Price ($)")
plt.setp(ax1.get_xticklabels(), visible=False)
plt.grid()

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(evebitda, color="blue", linewidth=0.7, alpha=0.7)
plt.annotate(str(float(evebitda.loc[end_date])),
             (end_date, evebitda.loc[end_date]))
plt.ylabel("EV/EBITDA")
plt.xlabel("Date")
plt.grid()

plt.savefig("graph.jpg")
