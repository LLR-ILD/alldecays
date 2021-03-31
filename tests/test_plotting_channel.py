import pytest

from alldecays.plotting.channel import (
    all_channel_plots,
    expected_counts_matrix,
    probability_matrix,
)


@pytest.mark.parametrize(
    "matrix_plot_function", [expected_counts_matrix, probability_matrix]
)
def test_matrix_plots(matrix_plot_function, channel1):
    n_boxes = len(channel1.box_names)
    n_decays = len(channel1.decay_names)
    n_bkg = len(channel1.bkg_names)
    # Just to make sure that the channel can show what we want it to show.
    assert n_bkg > 1

    ax = matrix_plot_function(channel1)
    m_default = ax.get_images()[0].get_array().data
    assert m_default.shape == (n_boxes, n_decays + n_bkg)

    ax = matrix_plot_function(channel1, no_bkg=True)
    m_no_bkg = ax.get_images()[0].get_array().data
    assert m_no_bkg.shape == (n_boxes, n_decays)

    ax = matrix_plot_function(channel1, combine_bkg=True)
    m_combine_bkg = ax.get_images()[0].get_array().data
    assert m_combine_bkg.shape == (n_boxes, n_decays + 1)


def test_all_channel_plots(test_plot_dir, channel1):
    all_channel_plots(channel1, test_plot_dir)
