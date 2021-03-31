import matplotlib.pyplot as plt
import numpy as np

from ..util import basic_kwargs_check, get_expected_matrix, get_experiment_tag


def _my_format(val):
    """Enforce a format style with 5 digits maximum including the decimal dot."""
    if val < 1e2:
        return f"{val:.2f}"
    if val < 1e3:
        return f"{val:.1f}"

    if val < 1e6:
        v_new, suffix = val / 1e3, "k"
    elif val < 1e9:
        v_new, suffix = val / 1e6, "M"
    elif val < 1e12:
        v_new, suffix = val / 1e9, "B"
    else:
        raise NotImplementedError(f"Value is larger than forseen: {val}.")

    if v_new < 10:
        return f"{v_new:.2f}{suffix}"
    elif v_new < 100:
        return f"{v_new:.1f}{suffix}"
    else:
        return f"{v_new:.0f}{suffix}"


def _plot_matrix(matrix, ax, **kwargs):
    basic_kwargs_check(**kwargs)
    if "experiment_tag" in kwargs:
        get_experiment_tag(kwargs["experiment_tag"])(ax)
    omit_zero = "omit_zero" in kwargs and kwargs["omit_zero"] is True

    def set_labels(ax):
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_ylabel("class", fontsize=14)
        try:
            ax.set_yticklabels(matrix.index, fontsize=12)
        except AttributeError:
            pass
        ax.set_xlabel("BR", fontsize=14)
        try:
            ax.set_xticklabels(matrix.columns, rotation=90, fontsize=12)
        except AttributeError:
            pass

    def set_numbers(ax):
        np_matrix = np.array(matrix)  # To make it work also for pd.DataFrame.
        for text_y, row in enumerate(np_matrix):
            for text_x, val in enumerate(row):
                if omit_zero and val == 0:
                    continue

                if val > 0.3 * np_matrix.max():
                    color = "black"
                else:
                    color = "white"

                ax.text(
                    text_x,
                    text_y,
                    _my_format(val),
                    ha="center",
                    va="center",
                    color=color,
                )

    ax.imshow(matrix)
    set_labels(ax)
    set_numbers(ax)
    return ax


def expected_counts_matrix(channel, ax=None, **kwargs):
    """A 2D matrix showing the expected counts per process and box.

    Args:
        channel: an alldecays.DataChannel
        ax: matplotlib axis that the plot is drawn onto.
            By default, create a new axis object.
        no_bkg: If True, exclude the background processes from the plot.
        combine_bkg: If True, combine all background processes into one.
    """
    basic_kwargs_check(**kwargs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))

    expected_matrix = get_expected_matrix(channel)
    n_signal = expected_matrix[channel.decay_names].sum().sum()

    if "no_bkg" in kwargs and kwargs["no_bkg"] is True:
        expected_matrix = expected_matrix[channel.decay_names]
    elif "combine_bkg" in kwargs and kwargs["combine_bkg"] is True:
        expected_matrix["bkg"] = expected_matrix[channel.bkg_names].sum(axis=1)
        expected_matrix = expected_matrix[channel.decay_names + ["bkg"]]

    _plot_matrix(expected_matrix, ax, **kwargs)
    ax.set_title(
        (
            f"Distribution of the {n_signal:_.0f} signal events\n"
            "in the channel as expected for SM BRs"
        ),
        fontsize=14,
    )
    return ax


def probability_matrix(channel, ax=None, **kwargs):
    """A 2D matrix showing the box probabilities per process.

    Column entries do not necessarily add up to 100%.
    The missing probability accounts for
    the selection efficiency of the data channel.

    Args:
        channel: an alldecays.DataChannel
        ax: matplotlib axis that the plot is drawn onto.
            By default, create a new axis object.
        no_bkg: If True, exclude the background processes from the plot.
        combine_bkg: If True, combine all background processes into one.
    """
    basic_kwargs_check(**kwargs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    percent_matrix = 100 * channel.mc_matrix

    if "no_bkg" in kwargs and kwargs["no_bkg"] is True:
        percent_matrix = percent_matrix[channel.decay_names]
    elif "combine_bkg" in kwargs and kwargs["combine_bkg"] is True:
        weight = channel.bkg_cs_default / channel.bkg_cs_default.sum()
        percent_matrix["bkg"] = percent_matrix[channel.bkg_names].dot(weight)
        percent_matrix = percent_matrix[channel.decay_names + ["bkg"]]

    _plot_matrix(percent_matrix, ax, **kwargs)
    ax.set_title("Matrix entries P(Class|BR) [%]", fontsize=14)
    return ax
