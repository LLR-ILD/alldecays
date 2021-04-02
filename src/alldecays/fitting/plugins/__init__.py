from .abstract_fit_plugin import AbstractFitPlugin
from .binomial_least_squares import BinomialLeastSquares
from .gaussian_least_squares import GaussianLeastSquares


available_fit_modes = {
    "BinomialLeastSquares": BinomialLeastSquares,
    "GaussianLeastSquares": GaussianLeastSquares,
}


def get_fit_mode(name):
    # Allow direct passing of the class, instead of the name.
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
