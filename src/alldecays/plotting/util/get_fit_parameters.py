"""Define a data class for the fit plot data"""
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class FitParameters:
    names: List[str]
    values: np.ndarray
    errors: np.ndarray
    covariance: np.ndarray
    starting_values: np.ndarray
    param_space: str
    is_from_toys: bool = False


valid_param_spaces = ["internal", "physics"]


def _get_fit_parameters_from_fit(fit, param_space):
    if param_space == "internal":
        fp = FitParameters(
            names=fit.Minuit.parameters,
            values=fit.Minuit.values,
            errors=fit.Minuit.errors,
            covariance=fit.Minuit.covariance,
            starting_values=fit.fit_mode.transform_to_internal(
                fit._data_set.fit_start_brs
            ),
            param_space=param_space,
        )
    elif param_space == "physics":
        fp = FitParameters(
            names=fit.fit_mode.parameters,
            values=fit.fit_mode.values,
            errors=fit.fit_mode.errors,
            covariance=fit.fit_mode.covariance,
            starting_values=fit._data_set.fit_start_brs,
            param_space=param_space,
        )
    else:
        raise NotImplementedError(f"`param_space` must be one of {valid_param_spaces}.")
    return fp


def _get_fit_parameters_from_toys(fit, param_space):
    if param_space == "internal":
        covariance = np.cov(fit.toys.internal.T)
        fp = FitParameters(
            names=fit.Minuit.parameters,
            values=fit.toys.internal.mean(axis=0),  # TODO: Is mean the best choice?
            errors=covariance.diagonal() ** 0.5,
            covariance=covariance,
            starting_values=fit.fit_mode.transform_to_internal(
                fit._data_set.fit_start_brs
            ),
            param_space=param_space,
            is_from_toys=True,
        )
    elif param_space == "physics":
        covariance = np.cov(fit.toys.physics.T)
        fp = FitParameters(
            names=fit.fit_mode.parameters,
            values=fit.toys.physics.mean(axis=0),
            errors=covariance.diagonal() ** 0.5,
            covariance=covariance,
            starting_values=fit._data_set.fit_start_brs,
            param_space=param_space,
            is_from_toys=True,
        )
    else:
        raise NotImplementedError(f"`param_space` must be one of {valid_param_spaces}.")
    return fp


def get_fit_parameters(fit, param_space, use_toys=False):
    """Return a data class that contains the data needed for fit plots.

    Args:
        fit: An alldecays.Fit object. If instead it is a FitParameter already ,
            it will be simply passed through his function.
        param_space: One of `internal` or `physics`.
        use_toys: If True, use values, errors and correlations
            obtained in the toy study (`fit.fill_toys`)
            instead of those obtained from only the fit on the expected counts.
            The default is False.
    """
    if isinstance(fit, FitParameters):
        return fit

    if use_toys:
        fp = _get_fit_parameters_from_toys(fit, param_space)
    else:
        fp = _get_fit_parameters_from_fit(fit, param_space)

    assert len(fp.names) == len(fp.values), f"{fp.names} ↔ {fp.values}"
    assert len(fp.errors) == len(fp.values), f"{fp.errors} ↔ {fp.values}"
    msg = f"{fp.starting_values} ↔ {fp.values}"
    assert len(fp.starting_values) == len(fp.values), msg
    msg = f"{fp.covariance.shape} ↔ {fp.values}"
    assert fp.covariance.shape[0] == len(fp.values), msg
    return fp
