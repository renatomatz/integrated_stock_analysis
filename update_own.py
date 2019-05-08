import pandas as pd
import matplotlib.pyplot as plt
import import_data

config = import_data.get_config("test_files/config.txt")
data = import_data.get_own_daily_metrics(config)
funds = import_data.get_own_fundamentals(config)

with open("charts/price.txt", "w") as f:
    f.write(str(float(data["close"].iloc[0])))

## FUNDAMENTALS


## PRICE AND EV/EBITDA
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

plt.savefig("charts/graph.jpg")
