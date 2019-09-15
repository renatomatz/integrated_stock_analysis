"""Class to process data from API Interfaces
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List

from Interface import Interface, Equity, Comps, Custom


class Grapher:
    """Abstract class for generating graphs specific to an interface with preset
    style

    Most of the methods in <Grapher> instances will have an <inter> argument,
    which will always be an <Interface> instance to be used 

    line_width: line width for line charts
    alpha: transparency
    colors: colors to be used in order when multiple lines are plotted
    """

    asset: Interface
    line_width: float
    alpha: float
    colors: List[str]

    def __init__(self, line_width=1, alpha=1,
                 colors=None):
        """Initialize Grapher instance and register matplotlib converters
        for pandas
        """
        pd.plotting.register_matplotlib_converters()

        self.line_width = line_width
        self.alpha = alpha

        self.colors = colors if colors else ["red", "green", "blue", "orange",
                                             "yellow", "purple"]

class EquityGrapher(Grapher):
    """Grapher class for <Equity> instances
    """

    def __init__(self, line_width=1, alpha=1, colors=None):

        if not colors:
            colors = ["red", "green", "blue", "orange", "yellow", "purple"] 

        super().__init__(line_width, alpha, colors)

    def get_price(self, inter, save=True):

        prices = inter.get_daily_metrics()
        
        if prices.empty:
            return None 

        price = str(float(prices["close"].iloc[0]))

        if save:
            with open(_chart_path(inter, "price.txt"), "w") as f:
                f.write(price)

        return price
        

    def fundamentals_chart(self, inter, save=True):

        funds = inter.get_fundamentals().iloc[0:9, 1:]

        funds.columns = ["Stock Price", "Ent. Value", "EBITDA", "Gross Margin",
                         "Sales", "Free Cash Flow", "Working Cap.", "EV/EBITDA", 
                         "EV/Sales", "EV/FCF", "Working Cap. % of Sales"]

        funds.index.names = ["Date"]

        funds = inter._round_col(funds,
                                 ["Ent. Value", "EBITDA", "Sales", 
                                  "Free Cash Flow", "Working Cap."], 
                                 to="B")

        funds.index = funds.index.map(lambda x: x.strftime("%b %y"))

        funds = funds.apply(lambda x: round(x, 2)).transpose()

        if save:
            funds.to_csv(_chart_path(inter, "fundamentals_chart.csv"))

        return funds

    def fundamentals_graph(self, inter, save=True):

        funds = inter.get_fundamentals()
        daily = inter.get_daily_metrics()
        
        fig = plt.figure()

        box_props = {"boxstyle": "round", "fc":"white"}

        ax1 = fig.add_subplot(221)

        funds["ev/sales"].plot(ax=ax1,
                               color="green",
                               linewidth=self.line_width,
                               alpha=self.alpha)

        ax1.set_ylabel("EV/Sales")
        ax1.set_xlabel("Date")
  
        ax1.grid()

        ax2 = fig.add_subplot(223, sharex=ax1)

        funds.ebitda.plot(ax=ax2, 
                              color="green",
                              linewidth=self.line_width,
                              alpha=self.alpha)

        funds.workingcapital.plot(ax=ax2, 
                                  color="blue",
                                  linewidth=self.line_width,
                                  alpha=self.alpha)

        ax2.set_ylabel(None)
        ax2.set_xlabel(None)
        ax2.legend()
  
        ax2.grid()

        ax3 = fig.add_subplot(222)

        daily.close.plot(ax=ax3,
                         color="blue",
                         linewidth=self.line_width,
                         alpha=self.alpha,
                         legend=False)

        ax3.annotate(str(float(daily.close.iloc[-1])), (1, daily.close.iloc[0]), 
                     xycoords=("axes fraction", "data"),
                     bbox=box_props)

        ax3.set_ylabel("Stock Price ($)")
        # ax3.setp(ax3.get_xticklabels(), visible=False)
        ax3.grid()

        ax4 = fig.add_subplot(224, sharex=ax3) #223 as this will make it below price

        daily.evebitda.plot(ax=ax4,
                           color="green",
                           linewidth=self.line_width,
                           alpha=self.alpha)

        ax4.annotate(str(float(daily.evebitda.iloc[-1])), 
                     (1, daily.evebitda.iloc[0]), 
                     xycoords=("axes fraction", "data"),
                     bbox=box_props)
  
        ax4.set_ylabel("EV/EBITDA")
        ax4.set_xlabel(None)
        ax4.grid()

        if save:
            fig.savefig(_chart_path(inter, "fundamentals_graph.jpg"))

        return fig

    def sales_growth(self, inter, save=True):

        funds = inter.get_fundamentals()

        sales_growth = funds[["revenue"]].iloc[::-1] \
                                         .pct_change(fill_method="ffill") \
                                         .iloc[1:] * 100

        sales_growth["positive"] = sales_growth["revenue"] > 0
        sales_growth.index = sales_growth.index.map(lambda x: x.strftime("%Y"))

        fig = plt.figure()

        ax = fig.add_subplot(111)

        ax.bar(sales_growth.index,
               sales_growth.revenue, 
               color=sales_growth["positive"].map({True: 'g', False: 'r'}),
               linewidth=self.line_width, 
               alpha=self.alpha)

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


        if save:
            fig.savefig(_chart_path(inter, "sales_growth.jpg"))

        
        return fig

    def events(self, inter, save=True):

        events = inter.get_events()
        data = inter.get_daily_metrics()

        events = inter._filter_dates(events, 
                                     start=min(data.index),
                                     end=max(data.index))

        fig = plt.figure()

        ax = fig.add_subplot(111)

        data.close.plot(ax=ax, 
                        color="green",
                        linewidth=self.line_width,
                        alpha=self.alpha)

        for row in events.itertuples():
            ax.annotate("*", 
                         (row.Index, data["close"][data.index == row.Index]),
                         ha="center")

        ax.set_xlabel(None)
        ax.set_ylabel("Stock Price ($)")
        plt.grid()

        if save:
            fig.savefig(_chart_path(inter, "events.jpg"))

        return fig
        

class CompsGrapher(Grapher):
    """Grapher class for <Comps> instances
    """

    def __init__(self, line_width=1, alpha=1,
                 colors=None):

        if not colors:
            colors = ["red", "green", "blue", "orange", "yellow", "purple"]

        super().__init__(line_width, alpha, colors)

    def fundamentals_chart(self, inter, save=True):

        comps = inter.get_comps()
        
        funds = inter.get_fundamentals()

        funds = funds.groupby("ticker") \
                     .apply(lambda x: x[x.index.get_level_values(1) \
                                        == max(x.index.get_level_values(1))])
        # leave only latest metrics

        funds.index = funds.index.droplevel(1)
        # keep ticker as index

        comps_merged = pd.merge(left=funds, 
                                right=comps, 
                                on=["ticker"],
                                left_index=True,
                                how="left")

        comps_merged = comps_merged[["name", "ticker", "price", "ev",
                                     "ev/ebitda", "ev/sales", "ev/fcf"]]
        # maintail relevant columns

        comps_merged.loc[:, "price":] = comps_merged.loc[:, "price":] \
                                                    .apply(lambda x: round(x, 2))

        comps_merged.columns = ["Name", "Ticker", "Stock Price", "Ent. Value",
                                "EV/EBITDA", "EV/SALES", "EV/FCF"]

        comps_merged = inter._round_col(comps_merged, ["Ent. Value"], to="B")
                                
        comps_merged = comps_merged.set_index("Ticker")
        # make ticker index

        if save:
            comps_merged.to_csv(_chart_path(inter, "comps.csv"))

        return comps_merged

    def comps_graph(self, inter, mkt=None, save=True): 
        """Create comps graph of daily indexed prices and ev/ebitda

        mkt: <Custom> instance for market data collection
        """

        if not mkt:
            mkt = Custom(inter._mem_file, inter._API_KEY)

        company = inter.get_daily_metrics().drop("evebitda", axis=1)

        industry = inter.get_industry_prices()
        
        market = mkt.get_yahoo_market_index(inter.market, columns=["Close"])

        baseline_prices = company
        
        if not industry.empty:
            industry = industry.set_index("ticker", append=True) \
                               .reorder_levels(["ticker", "date"])

        if not market.empty:
            market["ticker"] = np.array([inter.market]*len(market))

            market.index = market.index.rename("date")

            market = market.rename({"Close":"close"}, axis=1)

            market = market.set_index("ticker", append=True) \
                           .reorder_levels(["ticker", "date"])

        baseline_prices = pd.concat([company, industry, market])
        # inner join between market, fund and company prices

        del company
        del industry
        del market
        
        daily_evebitda = inter.get_daily_metrics().drop("close", axis=1)

        fun = lambda x: (x / x.loc[(x.index.get_level_values(0).min(),
                                    x.index.get_level_values(1)[0])]) * 100
        # indexing function

        baseline_prices = baseline_prices.groupby("ticker").transform(fun)

        fig = plt.figure()

        ax1 = fig.add_subplot(211)

        self._plot_groups(baseline_prices, drop_level=0, ax=ax1)

        ax1.set_xlabel(None)
        ax1.set_ylabel("Indexed Stock Price")
        ax1.legend(baseline_prices.index.get_level_values(0).unique())

        plt.grid()

        ax2 = fig.add_subplot(212, sharex=ax1)

        self._plot_groups(daily_evebitda, drop_level=0, ax=ax2)

        ax2.set_xlabel(None)
        ax2.set_ylabel("EV/EBITDA")
        ax2.legend(daily_evebitda.index.get_level_values(0).unique())

        plt.grid()

        if save:
            fig.savefig(_chart_path(inter, "comps_graph.jpg"))

        return fig

    def _plot_groups(self, ts, level=0, drop_level=None, ax=None):
        """Plot a time-series by a grouped level

        Precondition: len(groups) <= 6
        groups to be plotted based on index
        """

        grouped = ts.groupby(level=level)

        for i, group_info in enumerate(grouped):

            if drop_level:
                group_info[1].index = group_info[1].index.droplevel(drop_level)

            group_info[1].plot(ax=ax, color=self.colors[i])
            # plot grapth for each group precondition of at most 6 groups


def _chart_path(inter, f):

    path = "charts/" + \
        (inter.ticker[0] if isinstance(inter, Comps) else inter.ticker) + \
        "_C"*isinstance(inter, Comps) 

    if not os.path.exists(path):
        os.mkdir(path)

    return path + "/" + f

