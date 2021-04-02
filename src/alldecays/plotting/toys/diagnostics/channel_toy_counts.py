import matplotlib.pyplot as plt
import numpy as np

from alldecays.fitting.toy_values import ToyValues
from alldecays.plotting.channel.matrix_plots import _my_format
from alldecays.plotting.util import basic_kwargs_check, get_experiment_tag


def get_valid_toy_values(fit, channel_counts_needed=False):
    if hasattr(fit, "toys"):
        toy_values = fit.toys
        assert isinstance(toy_values, ToyValues)
    else:
        raise AttributeError(
            "Plots skipped: Fit passed without throwing toys first, or not a fit."
        )
    if channel_counts_needed and (
        not hasattr(toy_values, "_channel_counts") or toy_values._channel_counts is None
    ):
        raise AttributeError(
            "Plots skipped: _channel_counts not filled for the toys. \n"
            "Set `store_channel_counts=True` in fit.fill_toys (Usually a few \n"
            "(<< 100) toys are enough for diagnostics)."
        )
    return toy_values


def channel_toy_counts(fit, channel_name, ax=None, **kwargs):
    """Visualize the channel counts from toy fits.

    This might help to find why some fits are not finding the expected minimum.
    Especially targeted at cases where some fits gave
    minima for which the covariance matrix is not accurate.
    Args:
        fit: A Fit with fit.toys populated or directly a ToyValues object.
        channel_name: One of the names in the fit's _data_set.
        ax: matplotlib axis that the plot is drawn onto.
            By default, create a new axis object.
    """
    basic_kwargs_check(**kwargs)
    toy_values = get_valid_toy_values(fit, channel_counts_needed=True)
    channel = fit.fit_mode._data_set.get_channels()[channel_name]
    expected_counts = channel.get_expected_counts()
    x = np.arange(len(channel.box_names))
    color2 = "tab:blue"

    if ax is None:
        fig, ax = plt.subplots()
    if "experiment_tag" in kwargs:
        get_experiment_tag(kwargs["experiment_tag"])(ax)

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