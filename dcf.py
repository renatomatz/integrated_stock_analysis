import quandl
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from import_data import get_comps

class Stock:
    """Stock class for storing results from Discounted Cash Flow analyses and 
    relevant stock data used by models

    pgr: Perpetuity Growth Rate
    ke: Cost of Equity
    FCFE/DDM/REL: DCF <Model> calculate results

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
        """

        self.pgr = self.ke = self.roe = self.beta = \
        self.FCFE = self.DDM = self.REL = None

        self.ticker = ticker
        self.data = quandl.get_table("SHARADAR/SF1", ticker=ticker)
        self.data.set_index(pd.DateTimeIndex(self.data["datekey"]),
                            inplace=True)

        self._rfa = self._rfa_presets.get(rfa, rfa)
        self._mi = self._ma_presets.get(mi, mi)

        if do_stats:
            self.do_stats(do_all=True)

    def set_rfa(self, code="US10Y"):
        """Set risk-free asset FRED code either to a specific code or one 
        of the pre-made assets available in Stock._rfa_presets
        """
        self._rfa = self._rfa_presets.get(code, code)

    def set_mi(self, code="SP500"):
        self._mi = self._mi_presets.get(mi, mi)

    def set_cr(self, code="BAA"):
        self._cr = code

    def update_ke(self):
        """Update Cost of Equity

        .beta must be set
        """

        assert self.beta

        start = min(self.data.index)
        end = max(self.data.index)

        rf = web.DataReader(self._rfa, "fred", start, end).iloc[:, 0] / 100

        rm = web.DataReader(self._mi, "yahoo", start, end).Close.resample('Y')\
                                                          .ffill().pct_change()

        self.ke = np.mean(rf) + self.beta*np.mean(rm - rf)

    def update_pgr(self):
        
        self.pgr =  pgr(self.data.capex.iloc[0], 
                    self.data.workingcapital.iloc[0:2].diff()[1],
                    self.data.depamor.iloc[0],
                    self.data.netinc.iloc[0])

    def update_roe(self):

        self.roe = roe(self.eps, self.bvps)

    def update_beta(self):

        rm = web.DataReader(self._mi, "yahoo", min(self.data.index),
                                               max(self.data.index)).Close
       
        self.beta = beta(self.data.price.resample("Y").ffill().pct_change(), 
                         rm.resample("Y").ffill().pct_change())

    def do_stats(self, do_all=False):
        """Compute any missing stat attributes, leaving existing ones as they
        are unless do_all=True
        """

        if not self.beta:
            self.update_beta()

        if not self.roe:
            self.update_roe()

        if not self.pgr:
            self.update_pgr()

        if not self.ke:
            self.update_ke()

    def analyze_models(self):
        pass

    def set_rfa(self, new):
        self._rfa = self._rfa_presets.get(new, new)

    def get_rfa(self):
        return self._rfa

    def get_rfa_presets(self):
        return list(self._rfa_presets.keys())

    def set_mi(self, new):
        self._mi = self._mi_presets.get(new, new)

    def get_mi(self):
        return self._mi

    def get_mi_presets(self):
        return list(self._mi_presets.keys())


class Summary:
    """Object containing information about <Model>.calculate() results
    """

    result: int
    model: str
    methods: dict
    stats: object
    params: dict

    def __init__(self, result, model, methods, stats, params):
        self.result = result
        self.model = model
        self.methods = methods
        self.stats = stats
        self.params = params

    def __repr__(self):
        pass


class Model:
    """Abstract class for a DCF model containing model-specific configurations

    Models will often include configurable functions ending with:
    _f: actual function used to calculate a value
    _w: <Transformer> instance used to process data in  a <Stock> instance
    This is done to make specific functions used to be more transparent while
    still allowing for flexibility
    """

    def __init__(self):
        pass

    def calculate(self, Stock, summary=False, set_att = False):
        """Calculate <Stock> equity value 

        summary: whether to return the result as a float or a <Summary> object 

        set_att: whether to set a result <Summary> object as the Model's
        attribute in <Stock>
        """
        pass


class FCFE(Model):

    _forecast_f: object
    _forecast_w: object
    
    def __init__(self, forecast_f = None, forecast_w = None):

        self._forecast_f = forecast_f
        self._forecast_w = forecast_w

    def calculate(self, Stock, summary=False, set_att = False):
        
        wrap = self._forecast_w(self._forecast_f)



    def set_method(self, forecast_f=None, forecast_w=None):
        
        if forecast_f:
            self._forecast_f = forecast_f
        if forecast_w:
            self._forecast_w = forecast_w


class DDM(Model):

    def __init__(self, stages=None):
        pass

    def calculate(self, Stock, summary=False, set_att=False):
        pass


class REL(Model):
    
    _unlever_f: object
    _unlever_w: object

    _relever_f: object
    _relever_w: object

    def __init__(self):
        pass

    def calculate(self, Stock, summary=False, set_att = False):
        pass

    def set_method(self, unlever_f=None, unlever_w=None, 
                         relever_f=None, relever_w=None)

        if unlever_f:
            self._unlever_f = unlever_f
        if unlever_w:
            self._unlever_w = unlever_w
        if relever_f:
            self._relever_f = relever_f
        if relever:
            self._relever_w = relever_w

class Transformer:
    """Abstract Transformer decorator class

    Used inside <Model> instances to process contents of a <Stock> instance in
    order to fit the required arguments to some function <f>

    Instances of this class essentially "translate" contents of a table 
    returned by the Quandl API into the necessary arguments in any formula, 
    not only the ones pre-made here
    """

    def __init__(self, Stock):
        raise NotImplementedError() 

    def __call__(self, f):
        raise NotImplementedError() 

class Hamada_UL(Transformer):
    
    def __init__(self, Stock):

        assert Stock.beta
        assert Stock.data
        
        self.Stock = Stock

    def __call__(self, _hamada_u):

        def wrapper(*args):
                  
            return _hamada_u(*args,
                     self.Stock.beta, 
                     self.Stock.data.taxexp,
                     self.Stock.data.debt,
                     self.Stock.data.equity)

        return wrapper

class Hamada_RL(Transformer):
    
    def __init__(self, Stock):

        assert Stock.beta
        assert Stock.data

        self.Stock = Stock

    def __call__(self, _hamada_r):

        def wrapper(*args):

            return _hamada_r(*args,
                     self.Stock.beta, 
                     self.Stock.data.taxexp,
                     self.Stock.data.debt,
                     self.Stock.data.equity)

        return wrapper

class Fernandez_UL(Transformer):
    
    def __init__(self, Stock):

        assert Stock.beta
        assert Stock._cr
        assert Stock._mi
        assert Stock.data

        self.Stock = Stock

    def __call__(self, _fernandez_u):
    
        def wrapper(*args):

            rd = web.DataReader(self.Stock._cr, "fred").iloc[:, 0]\
                                                       .resample('Y').ffill()

            rm = web.DataReader(self.Stock._mi, "yahoo").Close\
                                                        .resample('Y').ffill()\
                                                        .pct_change()

            beta_debt = beta(rd, rm)

            return _fernandez_u(*args,
                     self.Stock.beta,
                     beta_debt,
                     self.Stock.data.taxexp,
                     self.Stock.data.debt,
                     self.Stock.data.equity)

        return wrapper

class Fernandez_RL(Transformer):
    
    def __init__(self, Stock):

        assert Stock.beta
        assert Stock._cr
        assert Stock._mi
        assert Stock.data

        self.Stock = Stock

    def __call__(self, _fernandez_r):

        def wrapper(*args):

            rd = web.DataReader(self.Stock._cr, "fred").iloc[:, 0]\
                                                       .resample('Y').ffill()

            rm = web.DataReader(self.Stock._mi, "yahoo").Close\
                                                        .resample('Y').ffill()\
                                                        .pct_change()

            beta_debt = beta(rd, rm)
            
            return _fernandez_r(*args,
                     self.Stock.beta,
                     beta_debt,
                     self.Stock.data.taxexp,
                     self.Stock.data.debt,
                     self.Stock.data.equity)
        
        return wrapper

class Central_Change_FC(Transformer):
    """Wrapper for centre functions used to a value based on past values

    Tested Functions:
    np.mean
    np.median
    np.mode
    """
    
    def __init__(self, Stock, col, n_fc):

        assert Stock.data
        assert col in Stock.columns

        self.Stock = Stock
        self.col = col

    def __call__(self, _centre):

        def wrapper():

            ts = self.Stock[self.col].resample('Y').ffill()
            p = ts.iloc[0]
            ts = ts.pct_change()
            
            m_ch = _centre(ts)

            fc = []

            for _ in range(self.n_fc):
                fc.append(fc[-1] * m_ch)

            return fc

        return wrapper

'''
class LM_FC(Transformer):
    """Wrapper for linear models that use past variables to predict another
    
    sklearn.linear_model. model architecture is assumed

    Tested Models:
    LinearRegression
    """
    
    def __init__(self, Stock, col_y, col_x):

        assert Stock.data
        assert col_y in Stock.columns
        assert all([col in Stock.columns for col in col_x])

        self.Stock = Stock
        self.col_y = col_y
        self.col_x = col_x

    def __call__(self, _lm):

        def wrapper(*args, **kwargs):

            Y = np.array(self.Stock.data.loc[:, self.col_y]).reshape(-1, 1)
            x = np.array(self.Stock.data.loc[:, self.col_x])

            mod = _lm(*args, **kwargs).fit(Y, x)

            
'''

class Mean_Rel(Transformer):
    
    def __init__(self, Stock, col):
       pass

    def __call__(self, f):
        pass

class LM_Rel(Transformer):
    
    def __init__(self, Stock):
       pass

    def __call__(self, f):
        pass

def p_val(n, r = 0, t = 1):
"""Return present value of a single value <n> at time <t> or the total present 
value of an np.array/list <n> starting at time <t> 
"""
    if isinstance(n, int):
        n / (1 + r)**t
    elif isinstance(n, np.ndarray):
        sum(n / (1 + r)*np.array(range(t, t+len(n))))
    elif isinstance(n, list):
        sum([elem / (1 + r)**i for i, elem in zip(range(t, t+len(n)), n)])
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
    return p_val(fcfe, pgr=ke) + p_val(term_val(fcfe[-1], pgr=pgr, ke=ke), pgr=ke, t = len(fcfe))

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

        period = np.array(range(start, stop))
        eq_val += sum((div * (1 + pgr)**(period+1)) / (1 + ke)**(period+1))

    else:
        if stop is not None:
            raise ValueError("last stop should be None")

    return eq_val
        
def eq_val_rel(mults, f_val):
"""Return equity value using Relative Valuation, given a list of 
its competitor's multiples

mults: comparable company multiples
f_val: firm's own value used in comparisson
    """
     pass

def pgr(capex, wc_inc, dep_am, net_inc):
    return (capex + wc_inc - dep_am) / net_inc 

def roe(ern, book):
    return ern / book

def capm_ret(rf, beta, rm):
    
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

    rm = rm.reshape(-1, 1)

    reg = LinearRegression().fit(rs, rm)

    return round(float(reg.coef_), 3)

def hamada_u(beta_l, tax, debt, equity):
    
    return beta_l / (1 + (1 - tax) * (debt / equity))

def hamada_r(beta_u, tax, debt, equity):
    
    return beta_u * (1 + (1 - tax) * (debt / equity))

def fernandez_u(beta_l, beta_d, tax, debt, equity):

    return (beta_l + beta_d*(1 - tax)*(debt/equity)) / (1 + (1 - T)*(debt/equity))

def fernandez_r(beta_u, beta_d, tax, debt, equity):
    
    return beta_u + (beta_u - beta_d)*(1 - T)(debt/equity)
