from pathlib import Path

from .channel import all_channel_plots
from .fit import all_fit_plots
from .util import basic_kwargs_check


def all_plots(fit, plot_folder, **kwargs):
    """Convenience wrapper around the provided plot options.

    Can be useful for getting a quick overview,
    or as a template for your own plotting script.

    Returns:
        dict[matplotlib figure.Figure]: Used in the module test suite.
    """
    basic_kwargs_check(**kwargs)
    plot_dir = {}

    for channel_name, channel in fit._data_set.get_channels().items():
        channel_plot_folder = Path(plot_folder) / channel_name
        channel_plot_folder.mkdir(exist_ok=True)
        cpd = all_channel_plots(channel, channel_plot_folder, **kwargs)
        for key in cpd:
            plot_dir[f"{channel_name}:{key}"] = cpd[key]

    fpd = all_fit_plots(fit, plot_folder, **kwargs)
    plot_dir.update(fpd)
    return plot_dir
