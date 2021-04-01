import numpy as np
from iminuit import Minuit

from .abstract_fit_plugin import AbstractFitPlugin


class GaussianLeastSquares(AbstractFitPlugin):
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
        y_variance = {k: v for k, v in y.items()}  # Could be changed?

        def fcn(x):
            return 2 * sum(
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
