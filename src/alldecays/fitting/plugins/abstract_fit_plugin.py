"""The interface defining class for fit modes."""
from abc import ABC, abstractmethod
import numpy as np

from iminuit import Minuit


class AbstractFitPlugin(ABC):
    """Minuit wrapper to standardize usage with different likelihood function
    definitions and parameter transformations.
    """

    def __init__(
        self,
        data_set,
        use_expected_counts=True,
        rng=None,
        has_limits=False,
        print_brs_sum_not_1=True,
    ):
        self._data_set = data_set
        self._use_expected_counts = use_expected_counts
        self.rng = rng
        self._counts = {}

        fcn = self._create_likelihood()
        internal_starters = self.transform_to_internal(data_set.fit_start_brs)
        self.Minuit = Minuit(fcn, internal_starters)
        self.has_limits = has_limits

        if not self._enforces_brs_sum_to_1 and print_brs_sum_not_1:
            print(
                f"INFO: The chosen minimizer plugin {self.__class__.__name__} "
                "does not enforce the branching ratios to sum to 1. \n"
                "      On top of being conceptually problematic, this will break "
                "if the signal cross section does not match with the expectation."
            )

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
    def values(self):
        pass

    @property
    @abstractmethod
    def parameters(self):
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

    @property
    @abstractmethod
    def _enforces_brs_sum_to_1(self) -> bool:
        """A True/False hook to allow some relaxing for sub-ideal likelihood descriptions."""
        pass

    def _prepare_numpy_y_M(self, return_dummy_M=False):
        """Prepare the MC counts matrix and box counts as numpy arrays.

        Args:
            return_dummy_M:
                If True, only build the array for y. This is useful for cases
                were M is already available (e.g. toy fits, where only y changes).
        """
        data_set = self._data_set

        n_boxes_per_channel = {
            k: len(ch.box_names) for k, ch in data_set.get_channels().items()
        }
        n_boxes = sum(n_boxes_per_channel.values())
        n_bkg = 1
        y = np.empty(n_boxes)

        if return_dummy_M:
            M = None
        else:
            n_parameters = len(data_set.decay_names)
            M = np.empty((n_boxes, n_parameters + n_bkg))

        i_stop = 0
        for name, channel in data_set.get_channels().items():
            i_start = i_stop
            i_stop = i_start + n_boxes_per_channel[name]

            # Fill y (in every toy)
            if self._use_expected_counts:
                self._counts[name] = channel.get_expected_counts()
            else:
                self._counts[name] = channel.get_toys(rng=self.rng)
            y[i_start:i_stop] = self._counts[name]

            # Fill M (if necessary)
            if not return_dummy_M:
                signal_factor = channel.signal_cs_default * channel.signal_scaler
                M[i_start:i_stop, :-n_bkg] = channel.mc_matrix[channel.decay_names]
                M[i_start:i_stop, :-n_bkg] *= signal_factor

                bkg_box_probabilities = (
                    channel.mc_matrix[channel.bkg_names]
                    * channel.bkg_cs_default
                    / channel.bkg_cs_default.sum()
                ).sum(axis=1)
                M[i_start:i_stop, -n_bkg] = bkg_box_probabilities
                M[i_start:i_stop, -n_bkg] *= channel.bkg_cs_default.sum()
                M[i_start:i_stop, :] *= channel.luminosity_ifb
        return y, M, n_bkg
