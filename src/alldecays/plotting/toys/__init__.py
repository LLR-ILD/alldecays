"""Plotting options for fits with toy study."""
from pathlib import Path

import matplotlib.pyplot as plt

from ..util import basic_kwargs_check, valid_param_spaces
from .diagnostics import toy_counts_channel
from .toy_hists import toy_hists
from .toy_util import get_valid_toy_values


def toy_diagnostics_plots(fit, plot_folder=None, **kwargs):
    """Run all diagnostics plots on a fit with toy study."""
    kwargs["allow_unused_kwargs"] = True
    figs = {}
    try:
        get_valid_toy_values(fit, channel_counts_needed=True)
    except AttributeError as ae:
        print(str(ae))
        return figs
    if plot_folder is not None:
        diagnostics_folder = Path(plot_folder) / "diagnostics"
        diagnostics_folder.mkdir(exist_ok=True)

    for channel_name, channel in fit._data_set.get_channels().items():
        fig, ax = plt.subplots()
        toy_counts_channel(fit, channel_name, ax, **kwargs)
        fig.tight_layout()
        fig_name = f"box_counts_{channel_name}"
        if plot_folder is not None:
            fig.savefig(diagnostics_folder / f"{fig_name}.png", dpi=800)
        figs[fig_name] = fig
    return figs


def all_toy_plots(fit, plot_folder=None, **kwargs):
    """Convenience wrapper around the provided plot options for the toys of a Fit.

    Can be useful for getting a quick overview,
    or as a template for your own plotting script.

    Returns:
        dict[matplotlib figure.Figure]: Used in the module test suite.
    """
    kwargs["allow_unused_kwargs"] = True
    basic_kwargs_check(**kwargs)
    figs = {}

    try:
        get_valid_toy_values(fit)
    except AttributeError as ae:
        print(str(ae))
        return figs

    for param_space in valid_param_spaces:
        toy_hists_figs = toy_hists(fit, plot_folder, param_space=param_space, **kwargs)
        figs.update(
            {f"toy_hists_{param_space}:{k}": v for k, v in toy_hists_figs.items()}
        )

    diagnostics_figs = toy_diagnostics_plots(fit, plot_folder, **kwargs)
    figs.update({f"diagnostics:{k}": v for k, v in diagnostics_figs.items()})
    return figs


__all__ = [
    "all_toy_plots",
    "toy_counts_channel",
    "toy_diagnostics_plots",
    "toy_hists",
]
