from ..data_handling.abstract_data_set import AbstractDataSet
from .plugins import get_fit_mode


class FitException(Exception):
    pass


default_fit_mode = "GaussianLeastSquares"
get_fit_mode(default_fit_mode)  # To make sure that this is a valid choice.


class Fit:
    """This class defines the input protocol for the fitting step.

    Example:
        >>> import alldecays
        >>> decay_names = ["X→AA", "X→BB", "X→CC"]
        >>> pol_dir = "/path/to/polarized/files/directory")
        >>> ds = alldecays.DataSet(decay_names, polarization=(-0.8, 0.3))
        >>> ds.add_channel("channel1", pol_dir)

    This assumes files `eLpL.csv`, `eLpR.csv`, `eRpL.csv`, `eRpR.csv`
    in `pol_dir` with at least the `decay_names` rows.

    As a design choice, channels are added by the path to their data file.
    This emphasizes that `_DataChannel`s are only meant to be used internally.
    You should be careful when modifying `_DataChannel` objects directly.
    Args:
        data_brs: default is  flat branching ratio.
        fit_start_brs: If not specified, defaults to `data_brs`.
    """

    def __init__(self, data_set, fit_mode=default_fit_mode, do_run_fit=True):
        if not isinstance(data_set, AbstractDataSet):
            raise FitException(
                "The provided data set does not follow the required protocol.\n"
                f"    {type(data_set) = }.\n"
                f"    {data_set = }"
            )
        self._data_set = data_set
        FitModeClass = get_fit_mode(fit_mode)
        self.fit_mode = FitModeClass(data_set)
        if do_run_fit:
            self.Minuit.migrad(ncall=10_000)

    def __repr__(self):
        return (
            f"{self.__class__.__name__} with:\n"
            f" - Fit mode: {self.fit_mode}\n"
            f" - Data set: {self._data_set}\n"
        )

    @property
    def Minuit(self):
        return self.fit_mode.Minuit
