"""Plots that compare different fits, or fits against external results."""
import matplotlib.pyplot as plt
import numpy as np

from alldecays.plotting.util import (
    basic_kwargs_check,
    get_experiment_tag,
    get_fit_parameters,
)


def compare_values(
    fits,
    param_space="physics",
    use_toys=False,
    experiment_tag=None,
    allow_unused_kwargs=False,
    shift_x=False,
    **kwargs,
):
    """Comparison of the parameter values from multiple fits.

    Measurements that are performed external to this module can be used
    by wrapping them as `FitParameter` objects.

    Args:
        fit: An alldecays.Fit or alldecays.plotting.util.FitParameters object.
        param_space: One of `internal` or `physics`.
        use_toys: If True, use values, errors and correlations
            obtained in the toy study (`fit.fill_toys`)
            instead of those obtained from only the fit on the expected counts.
            The default is False.
        experiment_tag: Add a watermark to the axis.
        allow_unused_kwargs: This can be nice to have for `all_plots`like calls.
        shift_x: Default=False. If True, add some jitter to the x values so that
            the fits that we want to compare do not fully overlap any more.
    """
    if allow_unused_kwargs:
        basic_kwargs_check(**kwargs)
    elif kwargs:
        raise TypeError(f"{', '.join(kwargs)} is an invalid keyword argument.")
    fig, axs = plt.subplots(figsize=(4, 6), nrows=2, sharex=True)
    if experiment_tag:
        get_experiment_tag(experiment_tag)(axs[0])
    local_kwargs = {k: v for k, v in kwargs.items()}
    local_kwargs["param_space"] = param_space
    local_kwargs["use_toys"] = use_toys
    local_kwargs["experiment_tag"] = None  # Only put watermark once (at most).
    local_kwargs["allow_unused_kwargs"] = allow_unused_kwargs
    local_kwargs["shift_x"] = shift_x
    compare_values_only(fits, axs[0], **local_kwargs)
    local_kwargs["as_relative_coupling_error"] = False
    compare_errors_only(fits, axs[1], **local_kwargs)
    axs[1].get_legend().remove()  # No need for 2 legends.
    fig.tight_layout()
    return fig


def _shift_x(i, old_x, n_instances):
    """Shift values to avoid overlap."""
    return old_x + 0.4 * (0.5 + i - n_instances / 2) / n_instances


def _get_val_and_err(valid_names, fit_params, fit_name=None):
    """Allows fit combinations to work even if the parameter orders differ."""
    valid_name_set = set(valid_names)
    fit_names = fit_params.names
    unseen_names = valid_name_set.union(fit_names) - valid_name_set
    if unseen_names:
        s = f"in '{fit_name}' " if fit_name is not None else ""
        raise NotImplementedError(
            f"Parameter names occurred {s}"
            "that were not used in the first fit:\n"
            f"{', '.join(unseen_names)}."
        )
    val = np.empty(len(valid_names))
    val[:] = np.NAN
    err = val.copy()
    for i, name in enumerate(valid_names):
        try:
            fit_idx = fit_names.index(name)
        except ValueError:
            continue
        val[i] = fit_params.values[fit_idx]
        err[i] = fit_params.errors[fit_idx]
    return val, err


def compare_values_only(
    fits,
    ax=None,
    param_space="physics",
    use_toys=False,
    experiment_tag=None,
    allow_unused_kwargs=False,
    shift_x=False,
    **kwargs,
):
    """Comparison of the parameter values from multiple fits.

    For a figure with both values and errors, `compare_values` can be used.

    Measurements that are performed external to this module can be used
    by wrapping them as `FitParameter` objects.

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
        shift_x: Default=False. If True, add some jitter to the x values so that
            the fits that we want to compare do not fully overlap any more.
    """
    if allow_unused_kwargs:
        basic_kwargs_check(**kwargs)
    elif kwargs:
        raise TypeError(f"{', '.join(kwargs)} is an invalid keyword argument.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    if experiment_tag:
        get_experiment_tag(experiment_tag)(ax)

    for i, (fit_name, fit) in enumerate(fits.items()):
        fit_params = get_fit_parameters(fit, param_space, use_toys)
        if i == 0:
            names = [n for n in fit_params.names]
            x = np.arange(len(names))
            sv = fit_params.starting_values
            ax.bar(x, sv, alpha=0.75, label="Input BRs", color="C0")
            values, errors = fit_params.values, fit_params.errors
        else:
            values, errors = _get_val_and_err(names, fit_params, fit_name)

        ax.errorbar(
            _shift_x(i, x, len(fits)) if shift_x else x,
            values,
            errors,
            xerr=0.3,
            fmt="o",
            label=fit_name,
            color=f"C{i+1}",
        )
    ax.set_ylabel("branching ratios")
    ax.legend(loc="center right")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)
    return ax


def compare_errors_only(
    fits,
    ax=None,
    param_space="physics",
    use_toys=False,
    experiment_tag=None,
    as_relative_coupling_error=True,
    allow_unused_kwargs=False,
    shift_x=False,
    **kwargs,
):
    """Comparison of the parameter errors from multiple fits.

    For a figure with both values and errors, `compare_values` can be used.

    Measurements that are performed external to this module can be used
    by wrapping them as `FitParameter` objects.

    Args:
        fit: An alldecays.Fit or alldecays.plotting.util.FitParameters object.
        ax: matplotlib axis that the plot is drawn onto.
            By default, create a new axis object.
        param_space: One of `internal` or `physics`.
        use_toys:
            If True, use values, errors and correlations obtained
            in the toy study (`fit.fill_toys`) instead of those
            obtained from only the fit on the expected counts.
            The default is False.
        experiment_tag: Add a watermark to the axis.
        as_relative_coupling_error:
            Instead of the total uncertainty on the
            branching ratio, display the relative error.
            Reformulating it as a coupling introduces a factor 1/2.
            This is more convenient for a comparison with external results.
        allow_unused_kwargs: This can be nice to have for `all_plots`like calls.
        shift_x: Default=False. If True, add some jitter to the x values so that
            the fits that we want to compare do not fully overlap any more.
    """
    if allow_unused_kwargs:
        basic_kwargs_check(**kwargs)
    elif kwargs:
        raise TypeError(f"{', '.join(kwargs)} is an invalid keyword argument.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    if experiment_tag:
        get_experiment_tag(experiment_tag)(ax)

    for i, (fit_name, fit) in enumerate(fits.items()):
        fit_params = get_fit_parameters(fit, param_space, use_toys)
        if i == 0:
            names = [n for n in fit_params.names]
            x = np.arange(len(names))
            values, errors = fit_params.values, fit_params.errors
        else:
            values, errors = _get_val_and_err(names, fit_params, fit_name)
        if as_relative_coupling_error:
            errors = errors / values / 2  # From BR or CS to coupling: / 2.
        x_i = _shift_x(i, x, len(fits)) if shift_x else x
        m = "*" if i == 0 else "_"
        ax.scatter(x_i, errors, marker=m, color=f"C{i+1}", label=fit_name)
        if as_relative_coupling_error:
            sel = np.argsort([x_i])
            sel = sel[~np.isnan(errors[sel])]
            ax.plot(x_i[sel], errors[sel], color=f"C{i+1}", alpha=0.3)

    if as_relative_coupling_error:
        ax.set_ylabel(r"$\Delta_X = g_X / g_X^{SM} - 1$")
    else:
        ax.set_ylabel("HESSE 67% CL interval")
    ax.legend(loc="center right")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)
    return ax
