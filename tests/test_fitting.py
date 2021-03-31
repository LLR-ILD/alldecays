import pytest

import alldecays
from alldecays.fitting.plugins import available_fit_modes, get_fit_mode
from alldecays.fitting.plugins.abstract_fit_plugin import AbstractFitPlugin


def test_fit_mode_validity(data_set1):
    for name, FitModeClass1 in available_fit_modes.items():
        FitModeClass2 = get_fit_mode(name)
        assert FitModeClass1 == FitModeClass2
        assert isinstance(FitModeClass1(data_set1), AbstractFitPlugin)
    assert FitModeClass1 == get_fit_mode(FitModeClass1)
    with pytest.raises(NotImplementedError):
        get_fit_mode("Non-existing name")


def test_limit_setting(data_set1):
    for fit_mode_name in available_fit_modes:
        f = alldecays.Fit(data_set1, fit_mode=fit_mode_name, has_limits=True)
        assert f.fit_mode.has_limits is True
    f.fit_mode.has_limits = False
    assert f.fit_mode.has_limits is False
    f = alldecays.Fit(data_set1, fit_mode=fit_mode_name, has_limits=False)
    assert f.fit_mode.has_limits is False
    f.fit_mode.has_limits = True
    assert f.fit_mode.has_limits is True
