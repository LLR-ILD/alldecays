"""Plotting options for fits."""
import matplotlib.pyplot as plt
from pathlib import Path

from .correlations_plot import fit_correlations
from ..util import basic_kwargs_check, valid_param_spaces


def all_fit_plots(fit, plot_folder=None, **kwargs):
    """Convenience wrapper around the provided plot options for a Fit.

    Can be useful for getting a quick overview,
    or as a template for your own plotting script.

    Returns:
        dict[matplotlib figure.Figure]: Used in the module test suite.
    """
    kwargs["allow_unused_kwargs"] = True
    basic_kwargs_check(**kwargs)
    figs = {}

    def save_wrap(fig, name):
        if plot_folder is not None:
            fig.tight_layout()
            if plot_folder is not None:
                fig.savefig(Path(plot_folder) / f"{name}.png")
        figs[name] = fig

    for param_space in valid_param_spaces:
        for use_toys in [True, False]:
            name = f"correlations_{param_space}"
            if use_toys:
                name += "_from_toys"
            fig, ax = plt.subplots(figsize=(5, 5))
            fit_correlations(fit, ax, param_space, use_toys, **kwargs)
            save_wrap(fig, name)

    return figs


__all__ = [
    "all_fit_plots",
    "fit_correlations",
]
