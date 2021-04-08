"""Poisson fit mode class."""
import numpy as np
from iminuit import Minuit

from .abstract_fit_plugin import AbstractFitPlugin


class Poisson(AbstractFitPlugin):
    """A Poisson likelihood fit."""

    def _create_likelihood(self):
        sig_weight, bkg_weight = {}, {}
        channel_luminosity = {}
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
            channel_luminosity[name] = channel.luminosity_ifb

        luminosity_times_M_dict = {
            n: M * channel_luminosity[n] for n, M in self._matrix.items()
        }
        y = self._counts

        def fcn(x):
            nu = {}
            for n, luminosity_times_M in luminosity_times_M_dict.items():
                nu[n] = luminosity_times_M.dot(
                    np.concatenate([sig_weight[n] * x, [bkg_weight[n]]])
                )
            return sum(nu[n].sum() - y[n].dot(np.log(nu[n])) for n in y.keys())

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
