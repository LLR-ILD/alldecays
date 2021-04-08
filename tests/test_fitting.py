import pytest

import alldecays
from alldecays.fitting.plugins import available_fit_modes, get_fit_mode
from alldecays.fitting.plugins.abstract_fit_plugin import AbstractFitPlugin


def test_fit_mode_choice(data_set1):
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


@pytest.mark.parametrize("fit_mode_name", available_fit_modes.keys())
def test_fit_mode_physics_parameter_consistency(fit_mode_name, data_set1):
    fit = alldecays.Fit(data_set1, fit_mode=fit_mode_name)
    m = fit.fit_mode
    n_physics_params = len(m.parameters)
    assert m.values.shape == (n_physics_params,)
    fit_performed_yet = m.covariance.shape != tuple()
    assert fit_performed_yet  # Raised e.g. if `fit_step=lambda x: None`.
    if fit_performed_yet:
        assert m.covariance.shape == (n_physics_params, n_physics_params)
        assert m.errors ** 2 == pytest.approx(m.covariance.diagonal())


@pytest.mark.parametrize("fit_mode_name", available_fit_modes.keys())
def test_fit_mode_design_choices_regression(fit_mode_name, data_set1, capsys):
    fit = alldecays.Fit(data_set1, fit_mode=fit_mode_name)
    m = fit.fit_mode
    if m._enforces_brs_sum_to_1:
        assert m.values.sum() == 1
    else:
        out, err = capsys.readouterr()
        expected_txt = "does not enforce the branching ratios to sum to 1"
        assert expected_txt in out

    fcn = m._create_likelihood()
    assert hasattr(fcn, "errordef")
    assert fcn.errordef == 0.5  # Consistently use Minuit.LIKELIHOOD.


@pytest.mark.parametrize(
    "fit_step",
    [lambda x: None, None],
    ids=["no fitting", "default value"],
)
def test_fit_step_valid(fit_step, data_set1):
    alldecays.Fit(data_set1, fit_step=fit_step)


def test_fit_step_invalid(data_set1):
    def fit_step(x):
        x.migrad(2)

    with pytest.raises(alldecays.exceptions.InvalidFitException):
        alldecays.Fit(data_set1, fit_step=fit_step)
    alldecays.Fit(data_set1, fit_step=fit_step, raise_invalid_fit_exception=False)
