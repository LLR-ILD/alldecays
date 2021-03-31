#!/usr/bin/env python

"""Python implementation of an exhaustive HEP decay analysis.

TODO: Longer description.
Basic usage :
TODO.

Further information:

* Code: https://github.com/LLR-ILD/alldecays
"""

from .data_handling import CombinedDataSet, DataSet
from .fitting import Fit
from .plotting import all_plots
from .version import __version__

__all__ = [
    "all_plots",
    "CombinedDataSet",
    "DataSet",
    "Fit",
    "__version__",
]
