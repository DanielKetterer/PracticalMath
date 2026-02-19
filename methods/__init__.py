"""
PracticalMath methods -- from-scratch econometric estimators.

Each sub-module implements one core method using only numpy / scipy,
with no black-box econometrics packages.
"""

from .utils import ols_fit, add_const
from . import ols
from . import fwl
from . import heteroskedasticity
from . import iv_2sls
from . import panel_fe
from . import did
from . import rdd
from . import binary_outcomes
from . import mle
from . import bootstrap
