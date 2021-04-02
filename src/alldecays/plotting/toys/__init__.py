import matplotlib.pyplot as plt
from pathlib import Path

from ..util import basic_kwargs_check
from .diagnostics import channel_toy_counts, get_valid_toy_values


def diagnostics_plots(fit, plot_folder, **kwargs):
    figs = {}
    try:
        get_valid_toy_values(fit, channel_counts_needed=True)
    except AttributeError as ae:
        print(str(ae))
        return figs
    diagnostics_folder = Path(plot_folder) / "diagnostics"
    diagnostics_folder.mkdir(exist_ok=True)

    for channel_name, channel in fit._data_set.get_channels().items():
        fig, ax = plt.subplots()
        channel_toy_counts(fit, channel_name, ax, **kwargs)
        fig.tight_layout()
        fig_name = f"box_counts_{channel_name}"
        fig.savefig(diagnostics_folder / f"{fig_name}.png", dpi=800)
        figs[fig_name] = fig
    return figs


def all_toy_plots(fit, plot_folder, **kwargs):
    """Convenience wrapper around the provided plot options for the toys of a Fit.

    Can be useful for getting a quick overview,
    or as a template for your own plotting script.

    Returns:
        dict[matplotlib figure.Figure]: Used in the module test suite.
    """
    basic_kwargs_check(**kwargs)
    figs = {}

    try:
        get_valid_toy_values(fit)
    except AttributeError as ae:
        print(str(ae))
        return figs

    # Put toy plots here.
    diagnostics_figs = diagnostics_plots(fit, plot_folder, **kwargs)
    figs.update({f"diagnostics:{k}": v for k, v in diagnostics_figs.items()})
    return figs
