import matplotlib
import matplotlib.pyplot as plt

import alldecays


def test_experiment_tag_ILD(fit1, test_plot_dir):
    """As it is saving a number of files, this test is rather time-consuming.

    E.g. when writing this comment: 4s out of 10s total test suite running.
    Thereof, > 2s just for saving (with folder != None).
    """
    tag_name = "ILD_preliminary"
    tag_text = "ILD preliminary"

    def tag_is_applied_to_ax(ax):
        def is_tag_object(ax_child):
            return (
                isinstance(ax_child, matplotlib.text.Text)
                and ax_child.get_text() == tag_text
            )

        return any(map(is_tag_object, ax.get_children()))

    folder = test_plot_dir / "test_experiment_tag_ILD"
    folder.mkdir()
    figures = alldecays.all_plots(fit1, folder, experiment_tag=tag_name)
    for fig_name, fig in figures.items():
        # fig_name to know which plot had the problem.
        assert fig_name and any(tag_is_applied_to_ax(ax) for ax in fig.get_axes())
    plt.close("all")


def test_no_saving(fit1):
    alldecays.all_plots(fit1)
    plt.close("all")
