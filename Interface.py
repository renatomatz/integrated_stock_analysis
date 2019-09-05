"""General model to deal with standard online data APIs
"""

import os
import json
from typing import List

import quandl
import pandas as pd
import pandas_datareader as web
import numpy as np

class Interface:
    """Abstract class for an interface between Quandl and local memory

    _mem_file: path to file containing this equity's already-imported data
    _API_KEY: quandl api key
    """

    _mem_file: str
    _API_KEY: int

    def __init__(self, key, mem_file=None):
        
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
    start: pd.Timestamp
    end: pd.Timestamp

    def __init__(self, name, key, market=None, industry=None, start=None, end=None,
                 mem_file=None):
        """Initialize an <Equity> object 

        name: either a ticker symbol or a json file containing information on 
        this equity.
        -- values defined in the json file will take priority
        over the ones defined in the initialization

        key: either a string containg a key, a file containing the key or an
        Interface object containing a key

        market: market index ticker symbol

        industry: industry fund ticker symbol

        start/end: <pd.Timstamps> objects, str objects with "%Y-%m-%d"
        format or None

        mem_file: path to directory which will contain saved data
        -- if None, automatic path of format "%datasets/%ticker" will be used
        -- if key is an Interface object and this attribute is None, the 
        interface.mem_file object will be used 

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

        if isinstance(key, Interface):
            key = key._API_KEY
            if mem_file is None:
                mem_file = key._mem_file

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
    def get_dataset(self, name, columns=None, save_new=True, industry=False, ts=False):
        """Get dataset from memory or import it from Quandl (and save into
        memory by default), optionally return only selected columns

        industry: whether to use the industry or company ticker

        ts: either a pd.Timestamp column to index on or a bool
        -- if True, the first column with "date" on it will be used as the 
        time series index
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

        if ts:
            if isinstance(ts, bool):
                ts = [c for c in data.columns if "date" in c][0]
            data = data.set_index(ts)

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

        if isinstance(name, Equity):
            name = name.get_comps()
            market = name.market
            industry = name.industry
            start = name.start
            end = name.end
            mem_file = name._mem_file + "_C"
            key = name._API_KEY

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
