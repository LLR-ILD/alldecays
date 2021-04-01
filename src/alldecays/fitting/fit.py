import numpy as np
import tqdm

from ..data_handling.abstract_data_set import AbstractDataSet
from .plugins import get_fit_mode
from .toy_values import ToyValues


class FitException(Exception):
    pass


default_fit_mode = "GaussianLeastSquares"
get_fit_mode(default_fit_mode)  # To make sure that this is a valid choice.


class Fit:
    """Wrapper around a Minuit fitting procedure on a DataSet.

    Example:
        >>> import alldecays
        >>> assert isinstance(data_set, alldecays.DataSet)
        >>> fit = Fit(data_set)
        >>> fit_internal = fit.Minuit
        >>> fit_physics = fit.fit_mode

    The most important property names of the iminuit.Minuit class
    are replicated in fit_mode (in the fitting plugins).
    Thus it is simple to switch between the usage
    of the internal parameters (for checks)
    and of the physics parameters (for the numbers that are of actual interest)
    in downstream code (e.g. plots): `m = fit.Minuit` or `m = fit.fit_mode`.
    """

    def __init__(
        self,
        data_set,
        fit_mode=default_fit_mode,
        use_expected_counts=True,
        rng=None,
        has_limits=False,
        do_run_fit=True,
    ):
        if not isinstance(data_set, AbstractDataSet):
            raise FitException(
                "The provided data set does not follow the required protocol.\n"
                f"    {type(data_set) = }.\n"
                f"    {data_set = }"
            )
        self._data_set = data_set
        FitModeClass = get_fit_mode(fit_mode)
        self.fit_mode = FitModeClass(data_set, use_expected_counts, rng, has_limits)
        if do_run_fit:
            self._standard_fitting_step()

    def __repr__(self):
        return (
            f"{self.__class__.__name__} with:\n"
            f" - Fit mode: {self.fit_mode}\n"
            f" - Data set: {self._data_set}\n"
        )

    def _standard_fitting_step(self):
        self.Minuit.migrad(ncall=10_000)

    @property
    def Minuit(self):
        return self.fit_mode.Minuit

    def fill_toys(self, n_toys=100, rng=None):
        """TODO: Multiprocessing"""
        if rng is None:
            rng = self.fit_mode.rng
        internal = np.zeros((n_toys, len(self.Minuit.parameters)))
        physics = np.zeros((n_toys, len(self.fit_mode.parameters)))
        valid = np.zeros(n_toys, dtype=bool)
        accurate = np.zeros(n_toys, dtype=bool)
        nfcn = np.zeros(n_toys, dtype=int)

        for i in tqdm.trange(n_toys, total=n_toys, unit=" toy minimizations"):
            # Set `do_run_fit=False` to check that most of the time is indeed
            # spent in the fitting step, and not in setup of the Data objects.
            toy_fit = Fit(
                data_set=self._data_set,
                fit_mode=type(self.fit_mode),
                use_expected_counts=False,
                rng=rng,
                has_limits=self.fit_mode.has_limits,
                do_run_fit=True,
            )
            internal[i] = toy_fit.Minuit.values
            physics[i] = toy_fit.fit_mode.values
            valid[i] = toy_fit.Minuit.valid
            accurate[i] = toy_fit.Minuit.accurate
            nfcn[i] = toy_fit.Minuit.nfcn

        self.toys = ToyValues(
            internal,
            physics,
            valid,
            accurate,
            nfcn,
        )
        return self.toys
