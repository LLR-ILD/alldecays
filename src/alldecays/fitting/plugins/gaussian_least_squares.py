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
        sig_weight, bkg_weight = {}, {}
        for name, channel in self._data_set.get_channels().items():
            bkg_box_probabilities = (
                channel.mc_matrix[channel.bkg_names]
                * channel.bkg_cs_default
                / channel.bkg_cs_default.sum()
            ).sum(axis=1)
            self._matrix[name] = channel.mc_matrix[channel.decay_names].copy()
            self._matrix[name]["bkg"] = bkg_box_probabilities
            if self._use_expected_counts:
                self._counts[name] = channel.get_expected_counts()
            else:
                self._counts[name] = channel.get_toys(rng=self.rng)
            sig_weight[name] = channel.signal_cs_default * channel.signal_scaler
            bkg_weight[name] = channel.bkg_cs_default.sum()

        M = self._matrix
        y = self._counts
        y_variance = self.variance_maker(y)

        def fcn(x):
            return 1 * sum(
                (
                    np.power(
                        M[n].dot(
                            np.concatenate([sig_weight[n] * x, [bkg_weight[n]]])
                            / (sig_weight[n] + bkg_weight[n])
                        )
                        * y[n].sum()
                        - y[n],
                        2,
                    )
                    / y_variance[n]
                ).sum()
                for n in M.keys()
            )

        fcn.errordef = Minuit.LEAST_SQUARES
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

    def variance_maker(self, y_dict):
        """Get the variance given the observed counts per box."""
        return {k: v for k, v in y_dict.items()}
