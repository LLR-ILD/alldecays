"""Collects the available options for fitting procedures."""
from .abstract_fit_plugin import AbstractFitPlugin
from .binomial_least_squares import BinomialLeastSquares
from .gaussian_least_squares import GaussianLeastSquares
from .poisson import Poisson

available_fit_modes = {
    "BinomialLeastSquares": BinomialLeastSquares,
    "GaussianLeastSquares": GaussianLeastSquares,
    "Poisson": Poisson,
}


def get_fit_mode(name):
    """Returns the fit_mode plugin of the given name.

    Allow direct passing of a class, instead of the name.
    This enables custom fit modes, if they inherit from `AbstractFitPlugin`.
    """
    try:
        if issubclass(name, AbstractFitPlugin):
            return name
    except TypeError:
        pass

    if name not in available_fit_modes:
        raise NotImplementedError(
            f"No fit mode with the name {name} is implemented.\n"
            f"Choose between: {available_fit_modes.keys()}."
        )
    return available_fit_modes[name]


__all__ = [
    "available_fit_modes",
    "get_fit_mode",
]
