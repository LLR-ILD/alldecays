"""Simple plots based on a toy study.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alldecays.plotting.util import (
    basic_kwargs_check,
    get_experiment_tag,
    get_fit_parameters,
    valid_param_spaces,
)

from .toy_util import get_valid_toy_values


def _gauss(x, mu, sigma):
    """1D Gaussian distribution"""
    return (2 * np.pi * sigma**2) ** -0.5 * np.exp(-0.5 * (x - mu) ** 2 / sigma**2)


def _toy_hist(fit_params, i, ax):
    """1D toy histogram helper"""
    bins = 20
    n_toys = fit_params.toy_values.shape[0]
    start_value = fit_params.starting_values[i]
    ax.set_title(fit_params.names[i])

    n, edges, _ = ax.hist(
        fit_params.toy_values[:, i],
        bins,
        label="\n".join(
            [
                f"Minima from {n_toys} fits",
                "on toy counts",
                "(MC2, Multinomial draws)",
            ]
        ),
        color="C1",
        density=True,
    )

    mask = fit_params.not_accurate
    if sum(mask) > 0:
        counts, _ = np.histogram(fit_params.toy_values[mask, i], edges, density=True)
        ax.bar(
            (edges[:-1] + edges[1:]) / 2,
            counts * sum(mask) / len(mask),
            (edges[:-1] - edges[1:]),
            hatch="///",
            color="gray",
            alpha=0.5,
            label="fits tagged 'not accurate'",
        )

    ax.axvline(
        start_value,
        color="grey",
        linestyle=":",
        linewidth=2,
        zorder=3,
        label=f"{start_value:0.5f} SM BR",
    )

    exp_br_i = fit_params.values[i]
    exp_err_i = fit_params.errors[i]
    ax.axvline(exp_br_i, color="black", label=f"{exp_br_i:0.5f} EECF Minimum")
    x = np.linspace(edges[0], edges[-1], 1000)
    ax.plot(
        x,
        _gauss(x, exp_br_i, exp_err_i),
        color="C0",
        label=f"{exp_err_i:0.5f} σ from EECF",
    )
    ax.legend(
        title="\n".join(
            [
                "EECF: Minuit fit on",
                "expected event counts (MC2)",
            ]
        )
    )


def toy_hists(
    fit,
    plot_folder=None,
    param_space="physics",
    ignore_accuracy=False,
    experiment_tag=None,
    allow_unused_kwargs=False,
    **kwargs,
):
    """The toy histograms for each parameter.

    They are compared with the values and errors
    from the fit on the expected counts.

    Args:
        fit: A Fit with fit.toys populated or directly a ToyValues object.
        plot_folder: Folder into which to save the histogram figures.
            If None, the figures are not saved.
        ignore_accuracy: With the default `ignore_accuracy=False`, toys with
            `ToyValues.accurate[i] == False` are marked in a special way.
        experiment_tag: Add a watermark to the axis.
        allow_unused_kwargs: This can be nice to have for `all_plots`like calls.
    """
    if allow_unused_kwargs:
        basic_kwargs_check(**kwargs)
    elif kwargs:
        raise TypeError(f"{', '.join(kwargs)} is an invalid keyword argument.")
    if plot_folder is not None:
        toy_hist_folder = Path(plot_folder) / f"toy_hists_{param_space}"
        toy_hist_folder.mkdir(exist_ok=True)

    toy_values = get_valid_toy_values(fit)
    fit_params = get_fit_parameters(fit, param_space, use_toys=False)
    if param_space == "internal":
        fit_params.toy_values = toy_values.internal
    elif param_space == "physics":
        fit_params.toy_values = toy_values.physics
    else:
        raise ValueError(f"`param_space` must be one of {valid_param_spaces}.")
    fit_params.not_accurate = [] if ignore_accuracy else ~toy_values.accurate

    figs = {}
    for i, name in enumerate(fit_params.names):
        fig, ax = plt.subplots(figsize=(4, 4))
        _toy_hist(fit_params, i, ax)
        if experiment_tag:
            get_experiment_tag(experiment_tag)(ax)
        if plot_folder is not None:
            fig.savefig(toy_hist_folder / f"{name}.png")
        figs[name] = fig
    return figs
