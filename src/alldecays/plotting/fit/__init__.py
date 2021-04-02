"""Plotting options for fits."""
from ..util import basic_kwargs_check  # , get_experiment_tag


def all_fit_plots(fit, plot_folder=None, **kwargs):
    """Convenience wrapper around the provided plot options for a Fit.

    Can be useful for getting a quick overview,
    or as a template for your own plotting script.

    Returns:
        dict[matplotlib figure.Figure]: Used in the module test suite.
    """
    kwargs["allow_unused_kwargs"] = True
    basic_kwargs_check(**kwargs)
    return {}
