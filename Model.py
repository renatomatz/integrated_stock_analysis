"""Define Models used for DCF analysis
"""

from typing import List, Tuple

from formulas import *

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

        return ev_pred / stock.data.get_dataset("SF1", columns="sharesbas")[0]

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
            self.stages = [(0, None, stock.data.get_dataset("SF1", 
                                                            columns="dps")[0])]

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
