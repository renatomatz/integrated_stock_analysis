from typing import List, Tuple

import quandl
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Stock:
    """Stock class for storing results from Discounted Cash Flow analyses and
    relevant stock data used by models

    ticker: company ticker symbol
    data: DataFrame containing all the company's fundamental data
    pgr: Perpetuity Growth Rate
    ke: Cost of Equity
    FCFE/DDM/REL: DCF <Model>.calculate() results

    _rfa: Risk-Free Asset
    _mi: Market Index
    _cr: Moody's credit rating
    """

    ticker: str
    data: pd.DataFrame
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

    def __init__(self, ticker, rfa="US10Y", mi="SP500", cr="BAA", do_stats=False):
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
        self.data = quandl.get_table("SHARADAR/SF1", ticker=ticker)
        self.data.set_index(pd.DatetimeIndex(self.data["calendardate"]),
                            inplace=True)

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

        start = min(self.data.index)
        end = max(self.data.index)

        rf = web.DataReader(self._rfa, "fred", start, end).iloc[:, 0] / 100

        rm = web.DataReader(self._mi, "yahoo", start, end).Close \
                                                          .resample('Y') \
                                                          .ffill() \
                                                          .pct_change()[1:]

        rf, rm = _same_len([rf, rm])

        self.ke = capm_ret(self.beta, rf, rm)

    def update_pgr(self):

        if not self.ke:
            self.update_ke()

        self.pgr = pgr(self.data.capex.iloc[0], 
                       self.data.workingcapital.iloc[0:2].diff()[1],
                       self.data.depamor.iloc[0],
                       self.data.netinc.iloc[0],
                       self.ke)

    def update_roe(self):

        self.roe = roe(self.data.eps[0], self.data.bvps[0])

    def update_beta(self):

        rs = self.data.price.resample("Y") \
                            .ffill() \
                            .pct_change()[1:] 

        rm = web.DataReader(self._mi, "yahoo", min(rs.index),
                                               max(rs.index)).Close \
                                                             .resample("Y") \
                                                             .ffill() \
                                                             .pct_change()[1:]

        rs, rm = _same_len([rs, rm])
       
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


class Model:
    """Abstract class for a DCF model containing model-specific configurations

    Models will often include configurable functions ending with:
    _f: actual function used to calculate a value
    _w: <Transformer> instance used to process data in a <Stock> instance
    -- This is done to make specific functions used to be more transparent while
    still allowing for flexibility.
    -- the transformer must have already been defined by the user
    -- the ._w <Transformer> must be designed to take these into consideration
    """

    def __init__(self):
        """Initialize model with any configurations
        """
        pass

    def calculate(self, stock, set_att=False):
        """Calculate <Stock> equity value 

        set_att: whether to set the result as the Model's attribute in <Stock>
        """
        raise NotImplementedError()

    def summary(self):
        """Return a summary of the model

        Summary Includes:
        -- Name
        -- Configurations
        -- Functions
        -- Wrappers
        -- Keyword Arguments
        """
        raise NotImplementedError()

    def __str__(self):
        """.summary call
        """
        self.summary()


class FCFE(Model):

    _forecast_f: object
    _forecast_w: object
    
    def __init__(self, forecast_f=None, forecast_w=None):

        self._forecast_f = forecast_f
        self._forecast_w = forecast_w

        super().__init__()

    def calculate(self, stock, set_att=False):
        
        fc = self._forecast_w(self._forecast_f)(stock)

        ev_pred = eq_val_fcfe(fc, stock.pgr, stock.ke)

        if set_att:
            stock.FCFE = ev_pred

        return ev_pred / stock.data.sharesbas[0]

    def summary(self):
        pass

    def set_method(self, forecast_f=None, forecast_w=None):
        
        if forecast_f:
            self._forecast_f = forecast_f
            
        if forecast_w:
            self._forecast_w = forecast_w


class DDM(Model):

    stages: Tuple[List[int]]

    def __init__(self, stages=None):
        """
        stages: predicted dividend stages of form list(tuple(start, stop, div))
        being that the last tuple should have stop == None
        -- if stages is None, <Stock>'s latest dividend rate will be used
        """
        
        self.stages = stages

        super().__init__()

    def calculate(self, stock, set_att=False):

        if self.stages is None:
            self.stages = [(0, None, stock.data.dps[0])]

        eq_pred = eq_val_ddm(self.stages, stock.pgr, stock.ke)

        if set_att:
            stock.DDM = eq_pred

        return eq_pred

    def summary(self):
        pass

class REL(Model):

    d_col: str

    _mult_f: object
    _mult_w: object
    
    def __init__(self, d_col=None, mult_f=None, mult_w=None):
        """
        d_col: denominator column
        """

        self._mult_f = mult_f
        self._mult_w = mult_w

        self.d_col = d_col

        super().__init__()

    def calculate(self, stock, set_att=False):

        mult = self._mult_w(self._mult_f)(stock)

        eq_v = eq_val_rel(mult, stock.data.loc[0, self.d_col])

        if set_att:
            stock.REL = eq_v

        return eq_v

    def summary(self):
        pass


class CAPM(Model):
    """Returns cost of equity instead of the equity value like other models

    More complete way of setting a <Stock> instance's .ke attribute than simply
    calling .update_ke()
    """
    
    _unlever_f: object
    _unlever_w: object

    _relever_f: object
    _relever_w: object

    def __init__(self, unlever_f=None, unlever_w=None,
                       relever_f=None, relever_w=None):

        self._unlever_f = unlever_f
        self._unlever_w = unlever_w

        self._relever_f = relever_f
        self._relever_w = relever_w

        super().__init__()

    def calculate(self, stock, set_att=False):

        stock.beta = self._unlever_w(self._unlever_f)(stock)
        
        if set_att:
            stock.update_ke()
        else:
            temp = stock.ke
            stock.update_ke()
            ret = stock.ke
            stock.ke = temp

        stock.beta = self._relever_w(self._relever_f)(stock)

        return stock.ke if set_att else ret

    def summary(self):
        pass

    def set_method(self, unlever_f=None, unlever_w=None, 
                         relever_f=None, relever_w=None):

        if unlever_f:
            self._unlever_f = unlever_f

        if unlever_w:
            self._unlever_w = unlever_w

        if relever_f:
            self._relever_f = relever_f

        if relever_w:
            self._relever_w = relever_w

class Transformer:
    """Abstract Transformer decorator class

    Used inside <Model> instances to process contents of a <Stock> instance in
    order to fit the required arguments to some function <f>

    Instances of this class essentially "translate" contents of a table 
    returned by the Quandl API into the necessary arguments in any function,
    not only the ones pre-made here
    """

    def __init__(self, *args, **kwargs):
        """Initializers should contain any options used in the transformer

        args/kwargs: will be passed onto whichever function is being wrapped
        """
        raise NotImplementedError() 

    def __call__(self, f):
        """Transformers are always callable, and inside this __call__ 
        definition we should define a wrapper whose first argument is the
        <Stock> instance

        Any args and kwargs used in the function should be defined upon 
        initialization of the Transformer
        """
        raise NotImplementedError() 

class Hamada_UL(Transformer):
    
    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs

    def __call__(self, _hamada_u):

        def wrapper(stock):

            assert stock.beta
            assert stock.data is not None

            tax_rate = stock.data.taxexp[0] / stock.data.ebt[0]
            
            return _hamada_u(stock.beta, 
                             tax_rate,
                             stock.data.debt[0],
                             stock.data.equity[0],
                             *self.args,
                             **self.kwargs)

        return wrapper

class Hamada_RL(Transformer):
    
    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs

    def __call__(self, _hamada_r):

        def wrapper(stock):

            assert stock.beta
            assert stock.data is not None

            tax_rate = stock.data.taxexp[0] / stock.data.ebt[0]

            return _hamada_r(stock.beta, 
                             tax_rate,
                             stock.data.debt[0],
                             stock.data.equity[0],
                             *self.args,
                             **self.kwargs)

        return wrapper

class Fernandez_UL(Transformer):
    
    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs

    def __call__(self, _fernandez_u):
    
        def wrapper(stock):

            assert stock._cr
            assert stock._mi

            rd = web.DataReader(stock.get_cr(), "fred").iloc[:, 0] \
                                                       .resample('Y') \
                                                       .ffill()

            rm = web.DataReader(stock.get_mi(), "yahoo").Close\
                                                        .resample('Y') \
                                                        .ffill() \
                                                        .pct_change()[1:]

            rd, rm = _same_len([rd, rm])

            beta_debt = beta(rd, rm)

            tax_rate = stock.data.taxexp[0] / stock.data.ebt[0]

            return _fernandez_u(stock.beta,
                                beta_debt,
                                tax_rate,
                                stock.data.debt[0],
                                stock.data.equity[0],
                                *self.args,
                                **self.kwargs)

        return wrapper

class Fernandez_RL(Transformer):
    
    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs

    def __call__(self, _fernandez_r):

        def wrapper(stock):

            assert stock._cr
            assert stock._mi

            rd = web.DataReader(stock.get_cr(), "fred").iloc[:, 0]\
                                                       .resample('Y') \
                                                       .ffill()

            rm = web.DataReader(stock.get_mi(), "yahoo").Close\
                                                        .resample('Y') \
                                                        .ffill()\
                                                        .pct_change()[1:]

            rd, rm = _same_len([rd, rm])

            beta_debt = beta(rd, rm)

            tax_rate = stock.data.taxexp[0] / stock.data.ebt[0]
            
            return _fernandez_r(stock.beta,
                                beta_debt,
                                tax_rate,
                                stock.data.debt[0],
                                stock.data.equity[0],
                                *self.args,
                                **self.kwargs)
        return wrapper

class Central_Change_FC(Transformer):
    """Wrapper for centre functions used to a value based on past values

    Tested Functions:
    np.mean
    np.median
    np.mode
    """
    
    def __init__(self, col, n_fc, *args, **kwargs):
        """
        n_fc: number of steps ahead to forecast
        """

        self.col = col
        self.n_fc = n_fc

        self.args = args
        self.kwargs = kwargs

    def __call__(self, _centre):

        def wrapper(stock):

            assert stock.data is not None
            assert self.col in stock.data.columns

            ts = stock.data[self.col].resample('Y').ffill()
            p = ts.iloc[0]
            # get latest value before converting to percentage change
            ts = ts.pct_change()
            
            m_ch = _centre(ts, *self.args, **self.kwargs)

            fc = [p]

            for _ in range(self.n_fc):
                fc.append(fc[-1] * (1 + m_ch))

            return fc

        return wrapper

# TODO: integrate relative models to <Interface> classed instead of using
# the outdated <get_comps> function

# class Mean_Rel(Transformer):
#     """Predict company metric by taking a centre metric of comps
# 
#     Tested Methods:
#     np.mean
#     np.median
#     np.mode
#     """
#     
#     def __init__(self, col, *args, **kwargs):
# 
#         self.col = col
# 
#         self.args = args
#         self.kwargs = kwargs
# 
#     def __call__(self, _centre):
#         
#         def wrapper(stock):
# 
#             assert stock.data is not None
#             assert self.col in stock.data.columns
# 
#             comps = [*get_comps({"ticker":stock.ticker})["ticker"]]
#             # import comp ticker names
#             
#             c_data = quandl.get_table("SHARADAR/SF1", 
#                                       ticker=comps,
#                                       qopts={"columns":[self.col]}).iloc[0]
# 
#             c_data = c_data.groupby("ticker")\
#                            .apply(lambda x: x[x.index.get_level_values(0) == \
#                                             max(x.index.get_level_values(0))])
#             # keep latest values
# 
#             return _centre(c_data[self.col], *args, **kwargs)
# 
#         return wrapper
# 
# 
# class LM_Rel(Transformer):
#     """Predict stock multiple given a linear model <_lm> fit to comps data
# 
#     Models use the scikit learn workflow
# 
#     Tested Models:
#     LinearRegression
#     """
#     
#     def __init__(self, y_col, x_col, *args, **kwargs):
#         
#         self.y_col = y_col
#         self.x_col = x_col
# 
#         self.args = args
#         self.kwargs = kwargs
# 
#     def __call__(self, _lm):
#         
#         def wrapper(stock):
# 
#             assert stock.data is not None
#             assert self.y_col in stock.data.columns
#             assert all([c in stock.data.columns for c in self.x_col])
# 
# 
#             comps = [*get_comps({"ticker":stock.ticker})["ticker"]]
# 
#             c_data = quandl.get_table("SHARADAR/SF1",
#                                       ticker=comps,
#                                       qopts={"columns":[[self.y_col] \
#                                                        + self.x_col]})
# 
#             reg = _lm(*self.args, **self.kwargs) \
#                         .fit(c_data.loc[:, self.x_col].values, 
#                              c_data[self.y_col].values.reshape(-1, 1))
# 
#             pred = reg.predict(stock.data.loc[:, self.x_col])
# 
#             return pred
# 
#         return wrapper


def p_val(n, r=0, t=1):
    """Return present value of a single value <n> at time <t> or the total present 
    value of an np.array/list <n> starting at time <t> 
    """
    if isinstance(n, float):
        return n / (1 + r)**t

    elif isinstance(n, np.ndarray):
        return sum(n / (1 + r)*np.array(range(t, t+len(n))))

    elif isinstance(n, list):
        return sum([elem / (1 + r)**i for i, elem in zip(range(t, t+len(n)), n)])

    else:
        raise TypeError("<n> is not of a supported type")          

    
def term_val(val, pgr, ke):
    """Return an asset's terminal value

    fcfe: Free Cash Flow to Equity value
    r: Perpetuity Growth Rate
    ke: Cost of Equity
    """

    return (val*(1+pgr)) / (ke - pgr)

def eq_val_fcfe(fcfe, pgr, ke):
    """Return equity value using the FCFE method given an estimation period

    fcfe: list containing forecasted Free Cash Flow to Equity values
    r: Perpetuity Growth Rate
    ke: Cost of Equity
    """
    return p_val(fcfe, r=ke) + p_val(term_val(fcfe[-1], pgr=pgr, ke=ke), r=ke, t=len(fcfe))

def eq_val_ddm(divs, pgr, ke):
    """Return equity value using the Multi-Stage DDM model given estimated
    dividend stages

    divs: list of tuples containing (start_period, stop_period, dividend)
          (,stop_period) should be None on the last period 
    """
    eq_val = 0

    for start, stop, div in divs:

        if stop is None:
            eq_val += ((div * (1 + pgr)**(start+1)) / (ke - pgr)) / (1 + ke)**start
            break

        period = np.arange(start, stop)
        eq_val += sum((div * (1 + pgr)**(period+1)) / (1 + ke)**(period+1))

    else:
        
        if stop is not None:
            raise ValueError("last stop should be None")
        # if the loop has ended and stop is not None, perpetuity value would
        # not be included, which should not be the case

    return eq_val
      
def eq_val_rel(rel_mult, f_val):
    """Return equity value using Relative Valuation, given a list of 
    its competitor's multiples

    rel_mult: multiple arrived at using comparable company data
    f_val: firm's own denominator value
    """

    return rel_mult * f_val

def pgr(capex, wc_inc, dep_am, net_inc, ke):
    return ((-capex + wc_inc - dep_am) / net_inc) * ke

def roe(ern, book):
    return ern / book

def capm_ret(beta, rf, rm):
    
    if not isinstance(rf, np.ndarray):
        rf = np.array(rf)

    if not isinstance(rm, np.ndarray):
        rf = np.array(rf)

    return np.mean(rf) + beta*np.mean(rm - rf)

def beta(rs, rm):
    """Calculate Beta using linear regression

    rs: Stock Returns
    rm: Market Returns
    """
    
    if not isinstance(rs, np.ndarray):
        rs = np.array(rs)

    if not isinstance(rm, np.ndarray):
        rm = np.array(rm)

    rs = rs.reshape(-1, 1)

    reg = LinearRegression().fit(rs, rm)

    return round(float(reg.coef_), 3)

def hamada_u(beta_l, tax_r, debt, equity):
    
    return beta_l / (1 + (1 - tax_r) * (debt / equity))

def hamada_r(beta_u, tax_r, debt, equity):
    
    return beta_u * (1 + (1 - tax_r) * (debt / equity))

def fernandez_u(beta_l, beta_d, tax_r, debt, equity):

    return (beta_l + beta_d*(1 - tax_r)*(debt/equity)) / (1 + (1 - tax_r)*(debt/equity))

def fernandez_r(beta_u, beta_d, tax_r, debt, equity):
    
    return beta_u + (beta_u - beta_d)*(1 - tax_r)*(debt/equity)

def _same_len(data):
    """Return data as arrays of the same length

    data: list of pd.Series
    """
    return pd.concat(data, join="inner", axis=1).transpose().values