"""Poisson fit mode class."""
import numpy as np
from iminuit import Minuit

from .abstract_fit_plugin import AbstractFitPlugin


class Poisson(AbstractFitPlugin):
    """A Poisson likelihood fit."""

    def _create_likelihood(self):
        y, M, n_bkg = self._prepare_numpy_y_M()

        def fcn(x):
            nu = M[:, :-n_bkg].dot(x) + M[:, -n_bkg:].sum(axis=1)
            return nu.sum() - y.dot(np.log(nu))

        fcn.errordef = Minuit.LIKELIHOOD
        return fcn

    def transform_to_internal(self, values):
        """Given the parameters in the physics space,
        return their internal representation for Minuit.
        """
        return np.array(values)

    @property
    def _default_limits(self):
        return [(0, float("inf"))] * len(self.Minuit.limits)

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
