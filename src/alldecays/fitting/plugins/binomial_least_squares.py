"""Adaptation of GaussianLeastSquares fit mode with binomial uncertainties."""
import numpy as np
import scipy.stats.distributions as dist

from .gaussian_least_squares import LeastSquares


def binomialProportionMeanAndCL(total_h, pass_h, cl=0.683, a=1, b=1):
    """a,b: Parameters of the prior (Beta function B(a, b)).
    B(1, 1) is the uniform distribution U(0,1).
    Jeffreys prior: B(.5, .5)

    Expected value is NOT nSelected/nEvents.
    While that is the most probable value / mode,the expected value / mean is
    given by (nSelected+a)/(nEvents+a+b).

    From: https://arxiv.org/pdf/0908.0130.pdf
    Unused source: https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf
    """
    mean = (pass_h + a) / (total_h + a + b)
    p_lower = dist.beta.ppf((1 - cl) / 2.0, pass_h + a, total_h - pass_h + b)
    p_upper = dist.beta.ppf(1 - (1 - cl) / 2.0, pass_h + a, total_h - pass_h + b)
    err_lower = mean - p_lower
    err_upper = p_upper - mean
    return mean, err_lower, err_upper


def get_binomial_1sigma_simplified(x):
    """Wraps binomialProportionMeanAndCL."""
    mean, err_lower, err_upper = binomialProportionMeanAndCL(x.sum(), x)
    return (err_lower + err_upper) / 2.0


class BinomialLeastSquares(LeastSquares):
    """A chi-square styled cost function with the variance transformed.

    This mainly differs from a standard Gaussian LeastSquares approach
    for those box probabilities that are close to 0 or 1.
    Especially, it gets rid of the 0-variance issue at 0 observed box counts.
    In the Gaussian case, 0-count boxes are effectively restricted
    to models that predict exactly 0 counts in the box.
    """

    def variance_maker(self, y):
        """Get the binomial variance given the observed counts per box."""
        y_variance = np.empty_like(y)
        i_stop = 0
        for name, counts in self._counts.items():
            i_start = i_stop
            i_stop = i_start + len(counts)
            y_variance[i_start:i_stop] = (
                counts.sum() ** 2 * get_binomial_1sigma_simplified(counts) ** 2
            )
        return y_variance
