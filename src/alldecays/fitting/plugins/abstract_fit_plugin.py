from abc import ABC, abstractmethod
import numpy as np

from iminuit import Minuit


class AbstractFitPlugin(ABC):
    """Minuit wrapper to standardize usage with different likelihood function
    definitions and parameter transformations.
    """

    def __init__(self, data_set, use_expected_counts=True, rng=None):
        self._data_set = data_set
        self._use_expected_counts = use_expected_counts
        self.rng = rng
        self._counts = {}
        self._matrix = {}

        fcn = self._create_likelihood()
        internal_starters = self.transform_to_internal(data_set.fit_start_brs)
        self.Minuit = Minuit(fcn, internal_starters)

    def __repr__(self):
        return self.__class__.__name__

    @property
    def errors(self):
        return np.array(self.covariance).diagonal() ** 0.5

    @abstractmethod
    def _create_likelihood(self):
        pass

    @abstractmethod
    def transform_to_internal(self, values):
        pass

    @property
    @abstractmethod
    def brs(self):
        pass

    @property
    @abstractmethod
    def covariance(self):
        pass
