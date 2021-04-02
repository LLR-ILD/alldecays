"""A function that runs all defined and possible plots at once."""
from pathlib import Path

from .channel import all_channel_plots
from .fit import all_fit_plots
from .toys import all_toy_plots
from .util import basic_kwargs_check


def all_plots(fit, plot_folder=None, **kwargs):
    """Convenience wrapper around the provided plot options.

    Can be useful for getting a quick overview,
    or as a template for your own plotting script.

    Returns:
        dict[matplotlib figure.Figure]: Used in the module test suite.
    """
    kwargs["allow_unused_kwargs"] = True
    basic_kwargs_check(**kwargs)
    figs = {}
    if plot_folder is None:
        channels_folder = None
        channel_plot_folder = None
        toys_folder = None
    else:
        channels_folder = Path(plot_folder) / "channels"
        channels_folder.mkdir(exist_ok=True)
        toys_folder = Path(plot_folder) / "toys"
        toys_folder.mkdir(exist_ok=True)

    for channel_name, channel in fit._data_set.get_channels().items():
        if channels_folder is not None:
            channel_plot_folder = channels_folder / channel_name
            channel_plot_folder.mkdir(exist_ok=True)
        cpd = all_channel_plots(channel, channel_plot_folder, **kwargs)
        for key in cpd:
            figs[f"{channel_name}:{key}"] = cpd[key]

    fpd = all_fit_plots(fit, plot_folder, **kwargs)
    figs.update(fpd)

    tpd = all_toy_plots(fit, toys_folder, **kwargs)
    figs.update({f"toys:{k}": v for k, v in tpd.items()})
    return figs
