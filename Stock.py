"""Class for Stock object used in DCF ananlysis
"""

import pandas_datareader.data as web
import matplotlib.pyplot as plt

from Interface import Equity
from formulas import *

class Stock:
    """Stock class for storing results from Discounted Cash Flow analyses and
    relevant stock data used by models

    ticker: company ticker symbol
    data: <Equity> interace for interacting with quandl data
    pgr: Perpetuity Growth Rate
    ke: Cost of Equity
    FCFE/DDM/REL: DCF <Model>.calculate() results

    _rfa: Risk-Free Asset
    _mi: Market Index
    _cr: Moody's credit rating
    """

    ticker: str
    data: Equity
    pgr: int
    ke: int
    roe: int
    beta: int
    FCFE: object
    DDM: object
    REL: object

    _rfa: str
    _mi: str
    _cr: str

    _rfa_presets = {"US10Y": "DGS10"}
    _mi_presets = {"SP500": "SPY"}

    def __init__(self, ticker, key, rfa="US10Y", mi="SP500", cr="BAA", do_stats=False, mem_file=None):
        """Import necessary Quandl stock data and, set ticker and risk-free 
        asset names and optionally calculate and set key statistics

        rfa: either a specific FRED code for an asset or an asset preset
        available in <Stock>._rfa_presets
        mi: either a specific Yahoo! Finance market index code or an index
        preset available in <Stock>._mi_presets
        cr: Moody's credit rating
        -- Currently only "AAA" and "BAA" are supported
        """

        self.pgr = self.ke = self.roe = self.beta = \
        self.FCFE = self.DDM = self.REL = None

        self.ticker = ticker
        self.data = Equity(ticker, key, mem_file=mem_file)

        self._rfa = self._rfa_presets.get(rfa, rfa)
        self._mi = self._mi_presets.get(mi, mi)
        self._cr = cr

        if do_stats:
            self.do_stats(do_all=True)

    def set_rfa(self, code="US10Y"):
        """Set risk-free asset FRED code either to a specific code or one 
        of the pre-made assets available in Stock._rfa_presets
        """
        self._rfa = self._rfa_presets.get(code, code)

    def set_mi(self, code="SP500"):
        self._mi = self._mi_presets.get(code, code)

    def set_cr(self, code="BAA"):
        self._cr = code

    def get_rfa(self):
        return self._rfa

    def get_rfa_presets(self):
        return list(self._rfa_presets.keys())

    def get_mi(self):
        return self._mi

    def get_mi_presets(self):
        return list(self._mi_presets.keys())

    def get_cr(self):
        return self._cr

    def update_ke(self):
        """Update Cost of Equity

        .beta must be set
        """

        if not self.beta:
            self.update_beta()

        start = min(self.data.get_dataset("SF1", columns="calendardate"))
        end = max(self.data.get_dataset("SF1", columns="calendardate"))

        rf = web.DataReader(self._rfa, "fred", start, end).iloc[:, 0] / 100

        rm = web.DataReader(self._mi, "yahoo", start, end).Close \
                                                          .resample('Y') \
                                                          .ffill() \
                                                          .pct_change()[1:]

        rf, rm = same_len([rf, rm])

        self.ke = capm_ret(self.beta, rf, rm)

    def update_pgr(self):

        if not self.ke:
            self.update_ke()

        self.pgr = pgr(self.data.get_dataset("SF1", columns="capex").iloc[0], 
                       self.data.get_dataset("SF1", columns="workingcapital")\
                                                   .diff()[1],
                       self.data.get_dataset("SF1", columns="depamor").iloc[0],
                       self.data.get_dataset("SF1", columns="netinc").iloc[0],
                       self.ke)

    def update_roe(self):

        self.roe = roe(self.data.get_dataset("SF1", columns="eps")[0], 
                       self.data.get_dataset("SF1", columns="bvps")[0])

    def update_beta(self):

        rs = self.data.get_dataset("SF1", columns="price").resample("Y") \
                            .ffill() \
                            .pct_change()[1:] 

        rm = web.DataReader(self._mi, "yahoo", min(rs.index),
                                               max(rs.index)).Close \
                                                             .resample("Y") \
                                                             .ffill() \
                                                             .pct_change()[1:]

        rs, rm = same_len([rs, rm])
       
        self.beta = beta(rs, rm)

    def do_stats(self, do_all=False):
        """Compute any missing stat attributes, leaving existing ones as they
        are unless do_all=True
        """

        if do_all:
            self.beta = self.roe = self.pgr = self.ke = None            

        if not self.beta:
            self.update_beta()

        if not self.ke:
            self.update_ke()

        if not self.roe:
            self.update_roe()

        if not self.pgr:
            self.update_pgr()

    def plot_models(self):

        x = ["FCFE", "DDM", "REL"]

        height = [self.FCFE if self.FCFE else 0,
                  self.DDM if self.DDM else 0,
                  self.REL if self.REL else 0]

        plt.bar(x, height)
