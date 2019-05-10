import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import datetime

import import_data

LINE_WIDTH = 0.7
ALPHA = 0.7 

register_matplotlib_converters()

config = import_data.get_config("test_files/config.txt")
data = import_data.get_own_daily_metrics(config)
funds = import_data.get_own_fundamentals(config)
events = import_data.get_own_events(config)

# data = import_data._read_and_standardize("test_files/own_daily_processed.csv")
# funds = import_data._read_and_standardize("test_files/own_funds_processed.csv", col="calendardate")
# events = import_data._read_and_standardize("test_files/own_events_processed.csv")

with open("charts/price.txt", "w") as f:
    f.write(str(float(data["close"].iloc[0])))

## FUNDAMENTALS

funds_pretty = funds.iloc[0:9]

funds_pretty.columns = ["Stock Price", "Ent. Value", "EBITDA", "Gross Margin",
                        "Sales", "Free Cash Flow", "Working Cap.", "EV/EBITDA", 
                        "EV/Sales", "EV/FCF", "Working Cap. % of Sales"]

funds_pretty.index.names = ["Date"]

funds_pretty = import_data.round_col(funds_pretty, ["Ent. Value", "EBITDA", "Sales", "Free Cash Flow", "Working Cap."], to="B")
funds_pretty.index = funds_pretty.index.map(lambda x: x.strftime("%b %y"))

funds_pretty = funds_pretty.apply(lambda x: round(x, 2)).transpose()

funds_pretty.to_csv("charts/own_fundamentals.csv")

## SALES GROWTH

sales_growth = funds[["revenue"]].iloc[::-1].pct_change(fill_method="ffill").iloc[1:] * 100
sales_growth["positive"] = sales_growth["revenue"] > 0
sales_growth.index = sales_growth.index.map(lambda x: x.strftime("%Y"))

fig = plt.figure()

ax = fig.add_subplot(111)

sales_growth.plot(kind="bar", ax=ax,
        color=sales_growth.positive.map({True: 'g', False: 'r'}),
        linewidth=LINE_WIDTH, alpha=ALPHA, legend=False)

ax.set_xlabel(None)
ax.set_ylabel("Growth (%)")

ax.spines["right"].set_color("none")
ax.spines["top"].set_position("zero")
ax.spines["bottom"].set_position("zero")

if not any(sales_growth["positive"]):
    ax.set_ylim(top=(-sales_growth["revenue"].min())/5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
elif all(sales_growth["positive"]):
    ax.set_ylim(bottom=(-sales_growth["revenue"].max())/5)


fig.savefig("charts/sales_growth.jpg")
plt.clf()

## PRICE AND EV/EBITDA
fig = plt.figure()

box_props = {"boxstyle": "round", "fc":"white"}

ax1 = plt.subplot(211)

data.close.plot(ax=ax1, color="blue", linewidth=LINE_WIDTH, alpha=ALPHA, legend=False)
plt.annotate(str(float(data.close.iloc[-1])), (1, data.close.iloc[0]), 
         xycoords=("axes fraction", "data"), bbox=box_props)

ax1.set_ylabel("Stock Price ($)")
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.grid()

ax2 = plt.subplot(212, sharex=ax1)

data.evebitda.plot(ax=ax2, color="green", linewidth=LINE_WIDTH, alpha=ALPHA)
plt.annotate(str(float(data.evebitda.iloc[-1])), (1, data.evebitda.iloc[0]), 
         xycoords=("axes fraction", "data"), bbox=box_props)

ax2.set_ylabel("EV/EBITDA")
ax2.set_xlabel(None)
ax2.grid()

plt.savefig("charts/graph.jpg")
plt.clf()

## COMPANY EVENTS
events = events[(events.index > min(data.index)) & (events.index < max(data.index))]

fig = plt.figure()

ax = fig.add_subplot(111)

data.close.plot(ax=ax, color="green", linewidth=LINE_WIDTH, alpha=ALPHA)

for row in events.itertuples():
    plt.annotate("*", (row.Index, data["close"][data.index == row.Index]), ha="center")

ax.set_xlabel(None)
ax.set_ylabel("Stock Price ($)")
plt.grid()

fig.savefig("charts/events.jpg")
plt.clf()
