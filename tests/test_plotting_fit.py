import matplotlib.pyplot as plt
import numpy as np

import alldecays
from alldecays.plotting.util import FitParameters


def test_fit_comparison_external(fit1, test_plot_dir):
    fits = {"fit": fit1}
    names = fit1._data_set.decay_names
    fits["external_similar"] = FitParameters(
        names=names,
        values=fit1._data_set.data_brs * (0.5 + np.random.random(len(names))),
        errors=0.1 * np.random.random(len(names)),
        covariance=None,
        starting_values=None,
    )
    fits["external_first_param_missing"] = FitParameters(
        names=names[1:],
        values=2 * np.random.random(len(names) - 1) / (len(names) + 1),
        errors=0.2 * np.random.random(len(names) - 1),
        covariance=None,
        starting_values=None,
    )
    fig = alldecays.plotting.compare_values(fits)
    fig.savefig(test_plot_dir / "comparison_values.png")

    fig, ax = plt.subplots(figsize=(4, 3))
    alldecays.plotting.compare_errors_only(fits, ax)
    fig.tight_layout()
    fig.savefig(test_plot_dir / "comparison_errors.png")
