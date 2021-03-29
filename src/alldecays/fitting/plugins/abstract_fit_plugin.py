from abc import ABC, abstractmethod
import numpy as np

from iminuit import Minuit


class AbstractFitPlugin(ABC):
    """Minuit wrapper to standardize usage with different likelihood function
    definitions and parameter transformations.
    """

    def __init__(self, data_set, use_expected_counts=True, rng=None, has_limits=False):
        self._data_set = data_set
        self._use_expected_counts = use_expected_counts
        self.rng = rng
        self._counts = {}
        self._matrix = {}

        fcn = self._create_likelihood()
        internal_starters = self.transform_to_internal(data_set.fit_start_brs)
        self.Minuit = Minuit(fcn, internal_starters)
        self.has_limits = has_limits

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

    @property
    @abstractmethod
    def _default_limits(self):
        pass

    @property
    def has_limits(self):
        inf = float("infinity")
        return self.Minuit.limits != [(-inf, inf)] * len(self.Minuit.limits)

    @has_limits.setter
    def has_limits(self, new_has_limits):
        if not isinstance(new_has_limits, bool):
            raise TypeError(
                f"Expected Bool. {type(new_has_limits)=}, {new_has_limits=}."
            )
        if new_has_limits == self.has_limits:
            pass
        elif new_has_limits:
            self.Minuit.limits = self._default_limits
        else:
            inf = float("infinity")
            self.Minuit.limits = [(-inf, inf)] * len(self.Minuit.limits)
