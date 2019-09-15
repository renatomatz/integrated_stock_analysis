"""Define Transformers which convert Interface data into formats usable by models
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web

from formulas import *

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

        Any args and kwargs use)d in the function should be defined upon 
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
            assert stock.data.get_dataset("SF1") is not None

            tax_rate = stock.data.get_dataset("SF1", columns="taxexp")[0] \
                       / stock.data.get_dataset("SF1", columns="ebt")[0]
            
            return _hamada_u(stock.beta, 
                             tax_rate,
                             stock.data.get_dataset("SF1", columns="debt")[0],
                             stock.data.get_dataset("SF1", columns="equity")[0],
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
            assert stock.data.get_dataset("SF1") is not None

            tax_rate = stock.data.get_dataset("SF1", columns="taxexp")[0] \
                       / stock.data.get_dataset("SF1", columns="ebt")[0]

            return _hamada_r(stock.beta, 
                             tax_rate,
                             stock.data.get_dataset("SF1", columns="debt")[0],
                             stock.data.get_dataset("SF1", columns="equity")[0],
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

            rd, rm = same_len([rd, rm])

            beta_debt = beta(rd, rm)

            tax_rate = stock.data.get_dataset("SF1", columns="taxexp")[0] / stock.data.get_dataset("SF1", columns="ebt")[0]

            return _fernandez_u(stock.beta,
                                beta_debt,
                                tax_rate,
                                stock.data.get_dataset("SF1", columns="debt")[0],
                                stock.data.get_dataset("SF1", columns="equity")[0],
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

            rd, rm = same_len([rd, rm])

            beta_debt = beta(rd, rm)

            tax_rate = stock.data.get_dataset("SF1", columns="taxexp")[0] / stock.data.get_dataset("SF1", columns="ebt")[0]
            
            return _fernandez_r(stock.beta,
                                beta_debt,
                                tax_rate,
                                stock.data.get_dataset("SF1", columns="debt")[0],
                                stock.data.get_dataset("SF1", columns="equity")[0],
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

            assert stock.data.get_dataset("SF1") is not None
            assert self.col in stock.data.get_dataset("SF1")

            ts = stock.data.get_dataset("SF1", columns=self.col, ts="calendardate") \
                                .resample('Y').ffill()
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
#             assert stock.data.get_dataset("SF1") is not None
#             assert self.col in stock.data.get_dataset("SF1, columns="columns"")
# 
#             c_data = Comps(self.data).get_dataset("SF1",
#                                                   columns=[self.col]).iloc[0]
#             # create a <Comps> instance to make possibly get the data from
#             # memory
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
#             assert stock.data.get_dataset("SF1") is not None
#             assert self.y_col in stock.data.get_dataset("SF1, columns="columns"")
#             assert all([c in stock.data.get_dataset("SF1", columns="columns ")for c in self.x_col])
# 
#             c_data = Comps(self.data).get_dataset("SF1",
#                                                   columns=[self.y_col, 
#                                                            self.x_col]) \
#                                      .iloc[0]
# 
#             reg = _lm(*self.args, **self.kwargs) \
#                         .fit(c_data.loc[:, self.x_col].values, 
#                              c_data[self.y_col].values.reshape(-1, 1))
# 
#             pred = reg.predict(stock.data.get_dataset("SF1").loc[:, self.x_col])
# 
#             return pred
# 
#         return wrapper
