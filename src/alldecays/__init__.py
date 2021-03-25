#!/usr/bin/env python

"""Python implementation of an exhaustive HEP decay analysis.

TODO: Longer description.
Basic usage :
TODO.

Further information:

* Code: https://github.com/LLR-ILD/alldecays
"""

from .data_handling import CombinedDataSet, DataSet
from .version import __version__

__all__ = [
    "CombinedDataSet",
    "DataSet",
    "__version__",
]
