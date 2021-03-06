"""Plotting utility code."""
import numpy as np

from .basic_kwargs_check import basic_kwargs_check
from .experiment_tags import get_experiment_tag
from .get_fit_parameters import FitParameters, get_fit_parameters, valid_param_spaces


def get_expected_matrix(channel):
    """Multiply the channel's `mc_matrix` with the necessary channel information."""
    n_signal = channel.luminosity_ifb * channel.signal_cs_default
    n_signal *= channel.signal_scaler
    brs = n_signal * channel.data_brs
    bkg = channel.luminosity_ifb * channel.bkg_cs_default
    process_scaler = np.concatenate([brs, bkg])
    expected_matrix = channel.mc_matrix * process_scaler
    return expected_matrix


__all__ = [
    "basic_kwargs_check",
    "FitParameters",
    "get_expected_matrix",
    "get_experiment_tag",
    "get_fit_parameters",
    "valid_param_spaces",
]
