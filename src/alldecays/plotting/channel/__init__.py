import matplotlib.pyplot as plt
from pathlib import Path

from .matrix_plots import expected_counts_matrix, probability_matrix
from ..util import basic_kwargs_check


def all_channel_plots(channel, plot_folder=None, **kwargs):
    """Convenience wrapper around the provided plot options for a _DataChannel.

    Can be useful for getting a quick overview,
    or as a template for your own plotting script.

    Returns:
        dict[matplotlib figure.Figure]: Used in the module test suite.
    """
    basic_kwargs_check(**kwargs)
    figs = {}

    fig, ax = plt.subplots(figsize=(8, 10))
    expected_counts_matrix(channel, ax=ax, **kwargs)
    if plot_folder is not None:
        fig.tight_layout()
        fig.savefig(Path(plot_folder) / "expected_counts.png")
    figs["expected_counts"] = fig

    fig, ax = plt.subplots(figsize=(8, 10))
    probability_matrix(channel, ax=ax, **kwargs)
    if plot_folder is not None:
        fig.tight_layout()
        fig.savefig(Path(plot_folder) / "probability_matrix.png")
    figs["probability_matrix"] = fig
    return figs


__all__ = [
    "all_channel_plots",
    "expected_counts_matrix",
    "probability_matrix",
]
