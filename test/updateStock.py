import quandl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import pdb
from import_data import get_data

data = get_processed()

with open("price.txt", "w") as f:
    f.write(str(float(data.prices.iloc[0])))

# plt.style.use("ggplot")

ax1 = plt.subplot(211)
plt.plot(data.prices, color="green", linewidth=0.7, alpha=0.7)
plt.annotate(str(float(data.prices.iloc[-1])),
             (data.index.iloc[-1], data.prices.iloc[-1]))
plt.ylabel("Stock Price ($)")
plt.setp(ax1.get_xticklabels(), visible=False)
plt.grid()

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(data.evebitda, color="green", linewidth=0.7, alpha=0.7)
plt.annotate(str(float(data.evebitda.iloc[-1])),
             (data.index.iloc[-1], data.evebitda.iloc[-1]))
plt.ylabel("EV/EBITDA")
plt.xlabel("Date")
plt.grid()

plt.savefig("graph.jpg")
