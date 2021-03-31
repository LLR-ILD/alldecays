import matplotlib.pyplot as plt
from pathlib import Path

from .matrix_plots import expected_counts_matrix, probability_matrix


def all_channel_plots(channel, plot_folder=None, **kwargs):
    fig, ax = plt.subplots(figsize=(8, 10))
    expected_counts_matrix(channel, ax=ax, **kwargs)
    if plot_folder is not None:
        fig.tight_layout()
        fig.savefig(Path(plot_folder) / "expected_counts.png")

    fig, ax = plt.subplots(figsize=(8, 10))
    probability_matrix(channel, ax=ax, **kwargs)
    if plot_folder is not None:
        fig.tight_layout()
        fig.savefig(Path(plot_folder) / "probability_matrix.png")


__all__ = [
    "all_channel_plots",
    "expected_counts_matrix",
    "probability_matrix",
]
