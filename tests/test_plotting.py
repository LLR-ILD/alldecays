import matplotlib

import alldecays


def test_experiment_tag_ILD(fit1, test_plot_dir):
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
        for ax in fig.get_axes():
            assert tag_is_applied_to_ax(ax)
