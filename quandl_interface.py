"""General model to deal with standard online data APIs and process
data
"""

import os
import json
from datetime import datetime
from typing import List

import quandl
import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt

class Interface:
    """Abstract class for an interface between Quandl and local memory

    _mem_file: path to file containing this equity's already-imported data
    _API_KEY: quandl api key
    """

    _mem_file: str
    _API_KEY: int

    def __init__(self, mem_file=None, key=None):
        
        self._mem_file = mem_file
        self.set_key(key)

    def set_key(self, key):

        if ".txt" in key:

            with open(key) as f:
                self._API_KEY = f.readline().rstrip("\n")

        else:
            self._API_KEY = key

    def set_mem_file(self, new):
        
        self._mem_file = new

    @staticmethod
    def _ensure_api_enabled(f):
        """Decorator that checks if an API key is set. Handles problem or throws
        propper error message
        """

        def check(*args, **kwargs):
            
            if quandl.ApiConfig.api_key is None:

                args[0].enable() # args[0] == self

            return f(*args, **kwargs)

        return check

class Equity(Interface):
    """Basic interface for interacting with online and in-memory data from the
    Sharadar Core US Equity Bundle

    Attributes:
    ticker: company ticker symbol
    market: company's general market index ticker symbol
    industry: company's industry index ticker symbol
    start: start date for data
    end: end data for data
    """

    ticker: str
    market: str
    industry: str
    start: datetime
    end: datetime

    def __init__(self, name, market=None, industry=None, start=None, end=None,
                 mem_file=None, key=None):
        """Initialize an <Equity> object 

        name: either a ticker symbol or a json file containing information on 
        this equity.
        -- values defined in the json file will take priority
        over the ones defined in the initialization

        market: market index ticker symbol

        industry: industry fund ticker symbol

        start/end: <datetime.datetime> objects, str objects with "%Y-%m-%d"
        format or None

        mem_file: path to directory which will contain saved data
        -- if None, automatic path of format "%datasets/%ticker" will be used

        key: either a Quandl API key or a .txt file pointing to one

        No datasets will be imported upon initialization, instead, they should
        be either specified in .update_dataset() or automatically saved as
        needed from one of the method calls
        """

        if ".json" in name:

            with open(name) as f:
                cfg = json.load(f)

            self.ticker = cfg["ticker"] if cfg["ticker"] else name
            self.market = cfg["market"] if cfg["market"] else market
            self.industry = cfg["industry"] if cfg["industry"] else industry
            self.start = cfg["start"] if cfg["start"] else start
            self.end = cfg["end"] if cfg["end"] else end

        else:
            self.ticker = name
            self.market = market
            self.industry = industry
            self.start = start
            self.end = end

        self.set_start(start)

        self.set_end(end)

        if not mem_file:
            mem_file = "data/" + self.ticker

        super().__init__(mem_file=mem_file, key=key)

    def enable(self):
        
        if self._API_KEY:
            quandl.ApiConfig.api_key = self._API_KEY
        else:
            ValueError("_API_KEY is not set, set it with self.set_key()")

    def set_start(self, new):

        if isinstance(new, str):
            new = pd.Timestamp(new)

        self.start = new

    def set_end(self, new):

        if isinstance(new, str):
            new = pd.Timestamp(new)

        self.end = new

    def set_market(self, new):
        self.market = new

    def set_industry(self, new):
        self.industry = new

    @Interface._ensure_api_enabled
    def get_dataset(self, name, columns=None, save_new=True, industry=False):
        """Get dataset from memory or import it from Quandl (and save into
        memory by default), optionally return only selected columns

        industry: whether to use the industry or company ticker
        """
        
        if self._exists(name):

            data = pd.read_csv(self._make_path(name))
            data.set_index("None")
            # Imported index is normally labeled as "None" and saved as column

            for col in [c for c in data.columns if "date" in c]:
            # Dates are not automatically converted to <pd.Timestamp> objects
            # assumes <pd.Timestamp> columns have "date" in their names
                data[col] = data[col].map(lambda x: pd.Timestamp(x))

        else:

            data = quandl.get_table("SHARADAR/"+name, 
                                    paginate=True,
                                    ticker=self.ticker if not industry \
                                                else self.industry)

            if save_new:

                if not os.path.exists(self._mem_file):
                    os.mkdir(self._mem_file)

                data.to_csv(self._make_path(name))

        return data.loc[:, columns] if columns else data

    def update_dataset(self, name):
        """Update existing dataset or import it from Quandl given its name
        """
        quandl.get_table("SHARADAR/" + name, ticker=self.ticker) \
            .to_csv(self._make_path(name))

    def available_saved(self):
        """Return the saved datasets available
        """
        return [*os.listdir(self._mem_file)]

    def _exists(self, name):
        """Return whether a dataset exists
        """
        return os.path.exists(self._make_path(name))

    def _make_path(self, name):
        """Return a path (not necessarily there) to a .csv file with a given
        name
        """

        return self._mem_file + "/" + name + ".csv"

    def _filter_dates(self, data, start=None, end=None):
        """Filter data in accordance to the values set in <self.start> and
        <self.end> or no bound if none are set

        Assumes data is indexed by date at level 0
        """
        if not start:
            start = self.start

        if not end:
            end = self.end
         
        if start: 
            data = data[data.index > start]

        if end:
            data = data[data.index <= end]

        return data

    def _ts_index(self, data, date_col=None):
        """Return data with a time series index at <date_col>

        If <date_col> is None, use the first column in the dataset containing
        the word "date"
        """

        if not date_col:
            date_col = [c for c in data.columns if "date" in c]

        return data.set_index(date_col)

    def _round_col(self, data, cols, to="M"):
        """Round columns to the nearest power of a thousand
        """
        new_cols = {}
        div = 10**(9 if to == "B" else 6 if to == "M" else 3 if to == "k" else 0)

        for c in cols:
            new_cols[c] = "{} (${})".format(c, to)
            data[c] = data[c] / div

        return data.rename(columns=new_cols, inplace=False)

    @Interface._ensure_api_enabled
    def get_daily_metrics(self):

        evebitda = self.get_dataset("DAILY", 
                                    columns=["ticker", "date", "evebitda"])

        prices = self.get_dataset("SEP",
                                  columns=["ticker", "date", "close"])
        
        data = pd.merge(left=evebitda,
                        right=prices,
                        on=["ticker", "date"],
                        how="inner")

        data = self._ts_index(data, date_col="date")

        data = self._filter_dates(data)

        return data

    @Interface._ensure_api_enabled
    def get_fundamentals(self):

        data = self.get_dataset("SF1", 
                                columns=["calendardate", "ticker", "price", 
                                         "ev", "ebitda", "grossmargin", 
                                         "revenue", "fcf", "workingcapital"])

        data["ev/ebitda"] = data["ev"] / data["ebitda"]
        data["ev/sales"] = data["ev"] / data["revenue"]
        data["ev/fcf"] = data["ev"] / data["fcf"]
        data["nwc_percent_sales"] = (data["workingcapital"] / data["revenue"])*100

        data = self._ts_index(data, date_col="calendardate")

        data = self._filter_dates(data)

        return data

    @Interface._ensure_api_enabled
    def get_events(self):

        events = self.get_dataset("EVENTS",
                                  columns=["date", "ticker", "eventcodes"])
        
        events["eventcodes"] = events["eventcodes"].map(lambda x: x.split("|"))

        events = self._ts_index(events, date_col="date")

        events = self._filter_dates(events)

        return events

    @Interface._ensure_api_enabled
    def get_industry_prices(self):
        
        if not self.industry:
            self.get_industry()

        if not self.industry:
            return pd.DataFrame()
            # if industry is not set by now, it means there is no close fund 
            # to the stock's industry

        else:

            data = self.get_dataset("SFP", 
                                    columns=["date", "ticker", "close"],
                                    industry=True)

        data = self._ts_index(data, date_col="date")

        data = self._filter_dates(data)
        
        return data

    @Interface._ensure_api_enabled
    def get_industry(self, set_new=True):
        """Get company industry based on its sic code

        As this method gathers all available data 
        """
        if self.industry and not set_new:
            return self.industry

        data = self.get_dataset("TICKERS")
        data = data[data.ticker == self.ticker][data.table == "SF1"]
        # guarantee there will be a single sic code available

        code = int(data["siccode"].iloc[0])

        data = quandl.get_table("SHARADAR/TICKERS",
                                paginate=True,
                                table="SFP",
                                qopts={"columns":["ticker", "siccode", 
                                                  "isdelisted"]})

        data = data[data.isdelisted == 'N'].drop("isdelisted", axis=1)

        funds = pd.DataFrame()
        i = 0

        while funds.empty and (i <= 3):

            funds = data[(code // (10**i)) == data["siccode"] \
                                             .apply(lambda x: x // (10**i) if x \
                                                                         else x)]

            i += 1

        funds = funds.iloc[0] if not funds.empty else None
        # if there are more than one fund selected keep the first if there
        # where no matching funds, then keep is as None

        if set_new:
            self.industry = funds.ticker

        return funds

    @Interface._ensure_api_enabled
    def get_comps(self):
        """Get an equity's comps based on market cap and sic code similarity

        As this method always gathers all available data from the TICKERS 
        dataset, it's better only to use it once to get an equity's comps
        """

        data = quandl.get_table("SHARADAR/TICKERS",
                                paginate=True,
                                table=["SEP", "SF1"],
                                qopts={"columns": ["ticker", "name", "category",
                                                   "siccode", "scalemarketcap",
                                                   "lastupdated", "isdelisted"]}
                                )

        data = data[data.isdelisted == 'N'].drop("isdelisted", axis=1)

        data["scalemarketcap"] = data["scalemarketcap"] \
                                     .apply(lambda x: int(x[0]) if x else None)
        # keep only scale category number
        data = data.groupby("ticker") \
                   .apply(lambda x: x[x.index == max(x.index)])
        # remove name duplicates, selecting most recent
        data.index = data.droplevel(level=1)
        data = data.drop_duplicates()
        # drop second index level created from aggregation and keep uniques

        ticker_data = data[data["ticker"] == self.ticker]

        if ticker_data.empty:
            raise ValueError("Ticker does not exist")

        ticker_cap = int(ticker_data["scalemarketcap"])

        data = data[ticker_cap-1 <= data["scalemarketcap"]]
        data = data[data["scalemarketcap"] <= ticker_cap+1]
        # keep only data of companies with similar market cap
        
        i = 0
        # such that one digit is revealed at a time
        comps = pd.DataFrame()

        while (len(comps) < 3) and (i <= 3):

            # make sic code become broader until there are at least three comps
            # or first sic code digit
            comps = data[(int(ticker_data["siccode"]) // (10**i)) \
                          == data["siccode"] \
                                .apply(lambda x: x // (10**i) if x else x)]

            i += 1
         
        if len(comps) > 6:

            if len(comps[comps.scalemarketcap == ticker_cap]) > 2:
                comps = comps[comps.scalemarketcap == ticker_cap]
            
            if len(comps[comps.category == str(ticker_data.category)]) > 2:
                comps = comps[comps.category == ticker_data.category]

            if len(comps) > 6:
                comps = comps.iloc[0:6]
            # guarantees there will be between 3 and 6 comps

        comps.index = np.arange(len(comps))

        return comps

class Comps(Equity):
    """Comps class similar to equity but dealing with more than a single ticker
    and a Multilevel Index

    Assumes main ticker is the first on the list
    """

    name: List[str]

    def __init__(self, name, market=None, industry=None, start=None, end=None,
                 mem_file=None, key=None):
        """Initialize <Comps> object, if <name> is not a list of comps tickers
        the <Equity.get_comps()> method will be used on it to try and find its
        comps and use those
        """

        if not mem_file and isinstance(name, list):
            mem_file = "data/" + name[0] + "_C"

        super().__init__(name, market=market, industry=industry, start=start, 
                         end=end, mem_file=mem_file, key=key)

        if not isinstance(name, list):
            self.name = super().get_comps()["ticker"]

    @Interface._ensure_api_enabled
    def get_comps(self):
        """Same as parent class but with memory import and no processing
        """

        data = self.get_dataset("TICKERS", 
                                columns=["ticker", "name", "category",
                                         "siccode", "scalemarketcap",
                                         "lastupdated"])

        data["scalemarketcap"] = data["scalemarketcap"] \
                                     .apply(lambda x: int(x[0]) if x else None)
        # keep only scale category number
        data = data.groupby("ticker") \
                   .apply(lambda x: x[x.index == max(x.index)])
        # remove name duplicates, selecting most recent
        data.index = data.droplevel(level=1)
        data = data.drop_duplicates()
        # drop second index level created from aggregation and keep uniques

        data.index = np.arange(len(data))

        return data

    def get_industry(self, set_new=True):
        """Exactly like in an <Equity> instance, using the main ticker
        """
        
        bu = self.ticker
        self.ticker = self.ticker[0]

        ret = super().get_industry(set_new=set_new)

        self.ticker = bu

        return ret

    def _ts_index(self, data, date_col=None):
        """Same as parent class, but also setting "ticker" as an index
        """

        if not date_col:
            date_col = [c for c in data.columns if "date" in c][0]

        return data.set_index(["ticker", date_col])

    def _filter_dates(self, data, start=None, end=None):
        """Filter data in accordance to the values set in <self.start> and
        <self.end> or no bound if none are set

        Assumes data is indexed by date at level 1
        """
        if not start:
            start = self.start

        if not end:
            end = self.end
         
        if start:
            data = data[data.index.get_level_values(1) > start]

        if end:
            data = data[data.index.get_level_values(1) <= end]

        return data

class Custom(Interface):
    """Interface for custom datasets not necessarily from Quandl or Sharadar
    Custom data imports should each gave their own cleaning and methods
    """

    def __init__(self, mem_file=None, key=None):

        if not mem_file:
            mem_file = "data"

        super().__init__(mem_file, key)

    def get_yahoo_market_index(self, name, columns=None):
        """Get market data from Yahoo! finance using a <pandas.data_reader>

        name: market index name as available in Yahoo! finance
        """
        if name is None:
            return pd.DataFrame()

        path = self._mem_file+"/"+name+".csv"

        if os.path.exists(path):

            data = pd.read_csv(path)

            data["Date"] = data["Date"].map(lambda x: pd.Timestamp(x))

            data = data.set_index("Date")

        else:
            
            data = web.DataReader(name, "yahoo")

            if not os.path.exists(self._mem_file):
                os.mkdir(self._mem_file)

            data.to_csv(path)
            # unlike Quandl tables, <DataReader> calls automatically index
            # data on dates, <Interface> calls are supposed to return data with
            # a standardized index, and optionally make them time series 

        return data.loc[:, columns] if columns else data


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
            with open(inter._mem_file+"/price.txt", "w") as f:
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
            funds.to_csv(inter._make_path("fundamentals_chart"))

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
            fig.savefig(inter._mem_file+"/fundamentals_graph.jpg")

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
            fig.savefig(inter._mem_file+"/sales_growth.jpg")

        
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
            fig.savefig("charts/events.jpg")

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
            comps_merged.to_csv("charts/comps.csv")

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
            fig.savefig("charts/comps_graph.jpg")

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
