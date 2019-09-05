"""formuals for some some specific functions, mostly used for educational
purposes
"""

from typing import List, Tuple

import quandl
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from quandl_interface import Equity

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
