"""Visualize the correlation data of the fit"""
import matplotlib.pyplot as plt
import numpy as np

from alldecays.plotting.util import (
    basic_kwargs_check,
    get_experiment_tag,
    get_fit_parameters,
)


def fit_correlations(
    fit,
    ax=None,
    param_space="physics",
    use_toys=False,
    experiment_tag=None,
    allow_unused_kwargs=False,
    **kwargs,
):
    """Correlations between the fit parameters.

    Args:
        fit: An alldecays.Fit or alldecays.plotting.util.FitParameters object.
        ax: matplotlib axis that the plot is drawn onto.
            By default, create a new axis object.
        param_space: One of `internal` or `physics`.
        use_toys: If True, use values, errors and correlations
            obtained in the toy study (`fit.fill_toys`)
            instead of those obtained from only the fit on the expected counts.
            The default is False.
        experiment_tag: Add a watermark to the axis.
        allow_unused_kwargs: This can be nice to have for `all_plots`like calls.
    """
    if allow_unused_kwargs:
        basic_kwargs_check(**kwargs)
    elif kwargs:
        raise TypeError(f"{', '.join(kwargs)} is an invalid keyword argument.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    if experiment_tag:
        get_experiment_tag(experiment_tag)(ax)
    fit_params = get_fit_parameters(fit, param_space, use_toys)

    cov = fit_params.covariance
    corr = (cov / cov.diagonal() ** 0.5).T / cov.diagonal() ** 0.5
    corr = corr - np.eye(corr.shape[0])  # We do not want to color the diagonal.
    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_xticklabels(fit_params.names, rotation=90)
    ax.set_yticklabels(fit_params.names)

    for text_y, row in enumerate(corr):
        for text_x, val in enumerate(row):
            if text_x == text_y:
                continue
            color = "black"  # if val > 0.3 * corr.max() else "white"
            ax.text(
                text_x,
                text_y,
                f"{val:.3f}".replace("0.", "."),
                ha="center",
                va="center",
                color=color,
            )
    ax.imshow(corr, cmap=plt.get_cmap("bwr"), vmin=-1, vmax=1)
    ax.set_title("Correlations")
    return ax
