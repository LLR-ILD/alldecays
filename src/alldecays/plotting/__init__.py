"""Plotting top module displaying (hopefully) all available plotting options."""
from .all_plots import all_plots
from .channel import (
    all_channel_plots,
    box_counts,
    expected_counts_matrix,
    probability_matrix,
)
from .fit import (
    all_fit_plots,
    compare_errors_only,
    compare_values,
    compare_values_only,
    fit_correlations,
)
from .toys import all_toy_plots, toy_counts_channel, toy_diagnostics_plots, toy_hists

__all__ = [
    "all_plots",
    "all_channel_plots",
    "all_fit_plots",
    "all_toy_plots",
    "box_counts",
    "compare_errors_only",
    "compare_values_only",
    "compare_values",
    "expected_counts_matrix",
    "fit_correlations",
    "probability_matrix",
    "toy_counts_channel",
    "toy_diagnostics_plots",
    "toy_hists",
]
