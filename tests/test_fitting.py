from pathlib import Path
import pytest

import alldecays
from alldecays.fitting.plugins import available_fit_modes, get_fit_mode


def get_data_set():
    data_set = alldecays.DataSet(decay_names=[f"dec{i}" for i in "ABC"])
    channel_path = Path(__file__).parent / "data/unpolarized/channel1.csv"
    data_set.add_channel("no_pol", channel_path)
    return data_set


def test_fit_mode_validity():
    from alldecays.fitting.plugins.abstract_fit_plugin import AbstractFitPlugin

    for name, FitModeClass1 in available_fit_modes.items():
        FitModeClass2 = get_fit_mode(name)
        assert FitModeClass1 == FitModeClass2
        assert isinstance(FitModeClass1(get_data_set()), AbstractFitPlugin)
    assert FitModeClass1 == get_fit_mode(FitModeClass1)
    with pytest.raises(NotImplementedError):
        get_fit_mode("Non-existing name")


def test_limit_setting():
    data_set = get_data_set()
    for fit_mode_name in available_fit_modes:
        f = alldecays.Fit(data_set, fit_mode=fit_mode_name, has_limits=True)
        assert f.fit_mode.has_limits is True
    f.fit_mode.has_limits = False
    assert f.fit_mode.has_limits is False
    f = alldecays.Fit(data_set, fit_mode=fit_mode_name, has_limits=False)
    assert f.fit_mode.has_limits is False
    f.fit_mode.has_limits = True
    assert f.fit_mode.has_limits is True
