import datetime as dt
from pathlib import Path
import pytest

from alldecays.data_handling.data_channel import _DataChannel
from alldecays.plotting.channel import (
    all_channel_plots,
    expected_counts_matrix,
    probability_matrix,
)


timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
test_plot_dir = Path(__file__).parent / "test_plots" / timestamp
test_plot_dir.mkdir(exist_ok=True, parents=True)


def get_data_channel():
    channel_path = Path(__file__).parent / "data/unpolarized/channel1.csv"
    channel = _DataChannel(channel_path, decay_names=[f"dec{i}" for i in "ABC"])
    return channel


@pytest.mark.parametrize(
    "matrix_plot_function", [expected_counts_matrix, probability_matrix]
)
def test_matrix_plots(matrix_plot_function):
    channel = get_data_channel()
    n_boxes = len(channel.box_names)
    n_decays = len(channel.decay_names)
    n_bkg = len(channel.bkg_names)
    # Just to make sure that the channel can show what we want it to show.
    assert n_bkg > 1

    ax = matrix_plot_function(channel)
    m_default = ax.get_images()[0].get_array().data
    assert m_default.shape == (n_boxes, n_decays + n_bkg)

    ax = matrix_plot_function(channel, no_bkg=True)
    m_no_bkg = ax.get_images()[0].get_array().data
    assert m_no_bkg.shape == (n_boxes, n_decays)

    ax = matrix_plot_function(channel, combine_bkg=True)
    m_combine_bkg = ax.get_images()[0].get_array().data
    assert m_combine_bkg.shape == (n_boxes, n_decays + 1)


def test_all_channel_plots():
    channel = get_data_channel()
    all_channel_plots(channel, test_plot_dir)
