import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..util import basic_kwargs_check, get_expected_matrix, get_experiment_tag


def box_counts(channel, ax=None, **kwargs):
    """A simple bar blot with one bar per box in the channel.

    Args:
        channel: an alldecays.DataChannel
        ax: matplotlib axis that the plot is drawn onto.
            By default, create a new axis object.
        no_bkg: If True, exclude the background processes from the plot.
        combine_bkg: If True, combine all background processes into one.
        combine_boxes: If True, combine all background processes into one.
        is_normalized: If True, each stacked bar goes up to 1.
    """
    basic_kwargs_check(**kwargs)
    expected = get_expected_matrix(channel)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    if "experiment_tag" in kwargs:
        get_experiment_tag(kwargs["experiment_tag"])(ax)
    if "no_bkg" in kwargs and kwargs["no_bkg"] is True:
        expected = expected[channel.decay_names]
    elif "combine_bkg" in kwargs and kwargs["combine_bkg"] is True:
        expected["bkg"] = expected[channel.bkg_names].sum(axis=1)
        expected = expected[channel.decay_names + ["bkg"]]
    if "combine_boxes" in kwargs and kwargs["combine_boxes"] is True:
        expected = pd.DataFrame(expected.sum()).T
        expected.index = ["in sample"]
    if "is_normalized" in kwargs and kwargs["is_normalized"] is True:
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
