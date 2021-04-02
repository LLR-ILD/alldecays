import pytest

import alldecays
from alldecays.plotting.toys.diagnostics import get_valid_toy_values


def test_get_valid_toy_values(data_set1):
    fit = alldecays.Fit(data_set1)
    with pytest.raises(AttributeError) as excinfo:
        get_valid_toy_values(fit)
    expected_info = "Plots skipped: Fit passed without throwing toys"
    assert str(excinfo.value)[: len(expected_info)] == expected_info

    fit.fill_toys(n_toys=2)
    get_valid_toy_values(fit)
    with pytest.raises(AttributeError) as excinfo:
        get_valid_toy_values(fit, channel_counts_needed=True)
    expected_info = "Plots skipped: _channel_counts not filled"
    assert str(excinfo.value)[: len(expected_info)] == expected_info

    fit.fill_toys(n_toys=2, store_channel_counts=True)
    get_valid_toy_values(fit, channel_counts_needed=True)
