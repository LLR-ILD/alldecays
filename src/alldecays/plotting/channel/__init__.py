"""Plotting options for data channels."""
import matplotlib.pyplot as plt
from pathlib import Path

from .box_counts import box_counts
from .matrix_plots import expected_counts_matrix, probability_matrix
from ..util import basic_kwargs_check


def all_channel_plots(channel, plot_folder=None, **kwargs):
    """Convenience wrapper around the provided plot options for a _DataChannel.

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

    fig, ax = plt.subplots(figsize=(8, 10))
    expected_counts_matrix(channel, ax=ax, **kwargs)
    save_wrap(fig, "expected_counts")

    fig, ax = plt.subplots(figsize=(8, 10))
    probability_matrix(channel, ax=ax, **kwargs)
    save_wrap(fig, "probability_matrix")

    fig, ax = plt.subplots(figsize=(4, 4))
    box_counts(channel, ax=ax, **kwargs)
    save_wrap(fig, "box_counts")

    fig, ax = plt.subplots(figsize=(4, 4))
    box_counts(channel, ax=ax, combine_boxes=True, **kwargs)
    save_wrap(fig, "sample_counts")

    fig, ax = plt.subplots(figsize=(4, 4))
    box_counts(channel, ax=ax, is_normalized=True, **kwargs)
    save_wrap(fig, "box_contributions")

    return figs


__all__ = [
    "all_channel_plots",
    "box_counts",
    "expected_counts_matrix",
    "probability_matrix",
]
