"""Define per.channel plots for the toy box counts."""
import matplotlib.pyplot as plt
import numpy as np

from alldecays.plotting.channel.matrix_plots import _my_format
from alldecays.plotting.util import basic_kwargs_check, get_experiment_tag
from ..toy_util import get_valid_toy_values


def toy_counts_channel(
    fit, channel_name, ax=None, experiment_tag=None, allow_unused_kwargs=False, **kwargs
):
    """Visualize the channel counts from toy fits.

    This might help to find why some fits are not finding the expected minimum.
    Especially targeted at cases where some fits gave
    minima for which the covariance matrix is not accurate.
    Args:
        fit: A Fit with fit.toys populated or directly a ToyValues object.
        channel_name: One of the names in the fit's _data_set.
        ax: matplotlib axis that the plot is drawn onto.
            By default, create a new axis object.
        experiment_tag: Add a watermark to the axis.
        allow_unused_kwargs: This can be nice to have for `all_plots`like calls.
    """
    if allow_unused_kwargs:
        basic_kwargs_check(**kwargs)
    elif kwargs:
        raise TypeError(f"{', '.join(kwargs)} is an invalid keyword argument.")

    toy_values = get_valid_toy_values(fit, channel_counts_needed=True)
    channel = fit.fit_mode._data_set.get_channels()[channel_name]
    expected_counts = channel.get_expected_counts()
    x = np.arange(len(channel.box_names))
    color2 = "tab:blue"

    if ax is None:
        fig, ax = plt.subplots()
    if experiment_tag:
        get_experiment_tag(experiment_tag)(ax)

    def draw_count_ratios(ax):
        for i in range(len(toy_values)):
            channel_counts = toy_values._channel_counts[i][channel_name]
            if toy_values.accurate[i]:
                kw = dict(ls=":", lw=0.7, alpha=0.7)
            else:
                kw = dict(ls="-", lw=1, alpha=0.8)
            ax.plot(x, channel_counts / expected_counts, **kw)
        if ax.get_ylim()[0] < 0.2:
            ax.set_ylim((0, None))
        ax.set_xticks(x)
        ax.set_xticklabels(channel.box_names, rotation=90)
        ax.set_xlabel("box names")
        ax.set_ylabel("toy counts / expected counts")

    def draw_bars(ax_bar):
        bar_container = ax_bar.bar(x, expected_counts, alpha=0.2, color=color2)
        ax_bar.tick_params(axis="y", labelcolor=color2)
        ax_bar.set_ylabel("expected counts", color=color2)
        return bar_container

    def add_numbers(ax_bar, bar_container):
        y_text = 0.98 * ax_bar.get_ylim()[0] + 0.02 * ax_bar.get_ylim()[1]
        for idx, rectangle in enumerate(bar_container):
            ax_bar.text(
                rectangle.get_x() + rectangle.get_width() / 2.0,
                y_text,
                _my_format(expected_counts[idx]),
                color=color2,
                ha="center",
                va="bottom",
                rotation=90,
            )

    ax.set_title(
        f"{len(toy_values.accurate)} toys in data channel {channel_name}\n"
        f"{sum(~toy_values.accurate)} straight lines for non-accurate covariance"
    )
    draw_count_ratios(ax)
    ax_bar = ax.twinx()
    bar_container = draw_bars(ax_bar)
    add_numbers(ax_bar, bar_container)

    return ax
