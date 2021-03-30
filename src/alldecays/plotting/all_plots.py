from pathlib import Path

from .channel import all_channel_plots
from .fit import all_fit_plots


def all_plots(fit, plot_folder, **kwargs):
    for channel_name, channel in fit._data_set.get_channels().items():
        channel_plot_folder = Path(plot_folder) / channel_name
        channel_plot_folder.mkdir(exist_ok=True)
        all_channel_plots(channel, channel_plot_folder, **kwargs)
    all_fit_plots(fit, plot_folder, **kwargs)
