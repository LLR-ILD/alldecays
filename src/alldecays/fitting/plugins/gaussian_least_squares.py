"""GaussianLeastSquares fit mode class."""
import numpy as np
from iminuit import Minuit

from .abstract_fit_plugin import AbstractFitPlugin


class LeastSquares(AbstractFitPlugin):
    """A least square fitting procedure.

    `variance_maker(self, y_dict)` has to be overwritten by an inheriting class.
    """

    def variance_maker(self, y_dict):
        """Should be overwritten by inheriting classes."""
        raise NotImplementedError

    def _create_likelihood(self):
        y, M, n_bkg = self._prepare_numpy_y_M()
        y_variance = self.variance_maker(y)

        def fcn(x):
            f_x = M[:, :-n_bkg].dot(x) + M[:, -n_bkg:].sum(axis=1)
            return 0.5 * (np.power(y - f_x, 2) / y_variance).sum()

        fcn.errordef = Minuit.LIKELIHOOD
        return fcn

    def transform_to_internal(self, values):
        """Given the parameters in the physics space,
        return their internal representation for Minuit.
        """
        return np.array(values)

    @property
    def _default_limits(self):
        return [(0, 1)] * len(self.Minuit.limits)

    @property
    def values(self):
        return np.array(self.Minuit.values)

    @property
    def parameters(self):
        return tuple(self._data_set.decay_names)

    @property
    def covariance(self):
        if self.Minuit.covariance is None:
            print("WARNING: Covariance not yet calculated by a Minuit fit.")
        return np.array(self.Minuit.covariance)

    @property
    def _enforces_brs_sum_to_1(self):
        """TODO? Conceptually, this actually seems rather hard to fix.

        In principle, `values -> values / sum(values)` would fix this:
            p_i = q_i / ∑q_k,
            J_ij = ∂p_i / ∂q_j,
            V_new = Jᵀ V J.
        Even for `∑q_k ≈ 1`, this can severely change the variances.
        Arguably these new variances are more correct.
        But this adds some non-obvious behavior to this plugin.

        The current opinion is that if `∑BR = 1` is desired,
        a different plugin should be chosen.
        """
        return False


class GaussianLeastSquares(LeastSquares):
    """The standard chi-square procedure.

    Has problematic behavior when zero counts are observed in a box.
    Those boxes are then effectively restrict the fit
    to models that predict exactly 0 counts in the box.

    This should not happen when working with the expected counts,
    but can easily occur in a toy study fluctuation
    for small (< 5) expected counts in a box.
    BinomialLeastSquares is provided to circumvent this issue.

    TODO: Add the references we found for this issue in iminit-BRS repo.
    """

    def variance_maker(self, y):
        """Get the variance given the observed counts per box."""
        return y
