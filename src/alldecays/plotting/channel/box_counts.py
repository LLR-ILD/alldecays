"""Per-channel plotting focusing on the data boxes."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..util import basic_kwargs_check, get_expected_matrix, get_experiment_tag


def box_counts(
    channel,
    ax=None,
    experiment_tag=None,
    no_bkg=False,
    combine_bkg=False,
    combine_boxes=False,
    is_normalized=False,
    allow_unused_kwargs=False,
    **kwargs,
):
    """A simple bar blot with one bar per box in the channel.

    Args:
        channel: an alldecays.DataChannel
        ax: matplotlib axis that the plot is drawn onto.
            By default, create a new axis object.
        experiment_tag: Add a watermark to the axis.
        no_bkg: If True, exclude the background processes from the plot.
        combine_bkg: If True, combine all background processes into one.
        combine_boxes: If True, combine all background processes into one.
        is_normalized: If True, each stacked bar goes up to 1.
        allow_unused_kwargs: This can be nice to have for `all_plots`like calls.
    """
    if allow_unused_kwargs:
        basic_kwargs_check(**kwargs)
    elif kwargs:
        raise TypeError(f"{', '.join(kwargs)} is an invalid keyword argument.")
    expected = get_expected_matrix(channel)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    if experiment_tag:
        get_experiment_tag(experiment_tag)(ax)
    if no_bkg:
        expected = expected[channel.decay_names]
    elif combine_bkg:
        expected["bkg"] = expected[channel.bkg_names].sum(axis=1)
        expected = expected[channel.decay_names + ["bkg"]]
    if combine_boxes:
        expected = pd.DataFrame(expected.sum()).T
        expected.index = ["in sample"]
    if is_normalized:
        expected = expected.div(expected.sum(axis=1), axis=0)
        ax.set_xlabel("expected signal composition")
    else:
        ax.set_xlabel("expected signal counts")

    y = np.arange(len(expected))
    left = np.zeros_like(y, dtype=float)
    ax.set_yticks(y)
    ax.set_yticklabels(expected.index)
    for process_name in expected:
        counts = expected[process_name]
        ax.barh(y, counts, left=left, label=process_name)
        left += counts
    ax.legend(title="(truth) Higgs decay mode", loc="center right")
    return ax
