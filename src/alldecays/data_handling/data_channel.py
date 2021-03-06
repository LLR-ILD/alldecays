"""Class for loading a data channel."""
from pathlib import Path

import numpy as np
import pandas as pd

from alldecays.exceptions import DataChannelError

from .pure_data_channel import _PureDataChannel
from .util import _polarization_cases, get_polarization_weights


class _DataChannel:
    """Meant to be used within a `DataSet`."""

    def __init__(
        self,
        channel_path,
        decay_names,
        polarization=None,
        data_brs=None,
        luminosity_ifb=1_000,
        signal_scaler=1.0,
        ignore_limited_mc_statistics_bias=False,
    ):
        self._decay_names = decay_names
        self.data_brs = self._set_brs(data_brs)
        self.luminosity_ifb = luminosity_ifb
        self.signal_scaler = signal_scaler
        self._set_initial_polarization(polarization)
        self._pure_channels = self._link_pure_channels(
            channel_path, ignore_limited_mc_statistics_bias
        )
        self._set_polarization_dependent_values()

    def _set_brs(self, brs=None):
        n_decays = len(self.decay_names)
        if brs is None:
            return np.ones(n_decays) / n_decays
        if len(brs) != n_decays:
            raise DataChannelError(f"{brs=}, {n_decays=}.")
        return brs

    def _set_initial_polarization(self, polarization):
        self._polarization = polarization
        self._validate_polarization(polarization)

    def _validate_polarization(self, pol):
        if self._polarization == pol:
            pass
        elif self._polarization is None:
            txt = f"An unpolarized {self.__class__.__name__} cannot be changed "
            txt += f"into a polarized one. {pol=}."
            raise DataChannelError(txt)
        elif pol is None:
            txt = f"A polarized {self.__class__.__name__} cannot be changed "
            txt += f"into an unpolarized one. {self._polarization=}."
            raise DataChannelError(txt)
        if pol is None:
            return
        if (
            not hasattr(pol, "__len__")
            or len(pol) != 2
            or not min(pol) >= -1
            or not max(pol) <= 1
        ):
            raise DataChannelError(f"Invalid polarization: {pol=}.")

    def _link_pure_channels(self, channel_path, ignore_limited_mc_statistics_bias):
        if self.polarization is None:
            return {
                "pure": _PureDataChannel(
                    channel_path, self.decay_names, ignore_limited_mc_statistics_bias
                )
            }
        pure_channel_store = {}
        csv_path = Path(channel_path)
        dir_files = list(csv_path.glob("*.csv"))
        file_stems = {p.stem: p for p in dir_files}
        train_stems = {}
        test_stems = {}
        for pol, pure_path in file_stems.items():
            if pol.startswith("train_"):
                train_stems[pol[len("train_") :]] = pure_path
            if pol.startswith("test_"):
                test_stems[pol[len("test_") :]] = pure_path
        if set(train_stems).issuperset(_polarization_cases):
            assert set(test_stems).issuperset(_polarization_cases)
            file_stems = train_stems
        elif set(file_stems).issuperset(_polarization_cases):
            pass
        else:
            txt = f"Missing polarized data file in {str(csv_path)}."
            if len(file_stems) == 0:
                txt += "\nNone found."
            else:
                txt += "\n" + f"Only found {sorted(set(file_stems))}."
            txt += f" We need: {_polarization_cases}."
            raise DataChannelError(txt)
        for pol in _polarization_cases:
            pure_channel_store[pol] = _PureDataChannel(
                file_stems[pol],
                self.decay_names,
                ignore_limited_mc_statistics_bias,
                allow_zero_signal=True,
            )
        return pure_channel_store

    @property
    def polarization(self):
        return self._polarization

    @polarization.setter
    def polarization(self, pol):
        self._validate_polarization(pol)
        self._polarization = pol
        self._set_polarization_dependent_values()

    def _set_polarization_dependent_values(self):
        if self.polarization is None:
            pc = self._pure_channels["pure"]
            self.signal_cs_default = pc.signal_cs_default
            self.bkg_cs_default = pc.bkg_cs_default
            self.mc_matrix = pc.mc_matrix
            self._data_faker = pc._data_faker
            return

        weights = get_polarization_weights(self.polarization)
        pcs = list(self._pure_channels.items())
        bkg_names = set(pcs[0][1].bkg_names)
        box_names = pcs[0][1].box_names
        for _, pc in pcs:
            bkg_names |= set(pc.bkg_names)
            if set(box_names) != set(pc.box_names):
                raise DataChannelError(f"{box_names=} != {pc.box_names=}.")

        def polarized_matrix(pure_dfs):
            pm = pd.DataFrame(
                0, columns=self.decay_names + sorted(bkg_names), index=box_names
            )
            for process in pm.columns:
                norm = 0
                for (pol, df) in pure_dfs.items():
                    if process not in df.columns or np.isnan(sum(df[process])):
                        continue
                    w = weights[pol]
                    norm += w
                    pm[process] = pm[process] + w * df[process]
                if norm != 0:
                    pm[process] = pm[process] / norm
            return pm

        self.mc_matrix = polarized_matrix({p: pc.mc_matrix for (p, pc) in pcs})
        self._data_faker = polarized_matrix({p: pc._data_faker for (p, pc) in pcs})

        self.signal_cs_default = sum(
            weights[p] * pc.signal_cs_default for (p, pc) in pcs
        )
        self.bkg_cs_default = np.zeros(len(self.bkg_names))
        for i, bkg_name in enumerate(self.bkg_names):
            for p, pc in pcs:
                try:
                    idx = pc.bkg_names.index(bkg_name)
                except ValueError:
                    continue
                self.bkg_cs_default[i] += pc.bkg_cs_default[idx] * weights[p]

    @property
    def decay_names(self):
        return self._decay_names

    @decay_names.setter
    def decay_names(self, new_names):
        for pc in self._pure_channels.values():
            pc.decay_names = new_names
        if len(self.decay_names) != len(new_names):
            raise Exception(f"{self.decay_names=}, {new_names=}.")
        for df in [self._data_faker, self.mc_matrix]:  # Order matters!
            df.rename(
                columns={old: new for (old, new) in zip(self.decay_names, new_names)},
                inplace=True,
            )
        self._decay_names = new_names

    def drop_bkg(self, bkg_names):
        old_bkg_names = self.bkg_names
        if isinstance(bkg_names, str):
            bkg_names = [bkg_names]
        if not set(bkg_names).issubset(old_bkg_names):
            missing_bkg = set(bkg_names) - set(old_bkg_names)
            raise Exception(f"{missing_bkg} bkg not found.")
        for pc in self._pure_channels.values():
            pure_names = pc.mc_matrix.columns
            pure_bkg_names = [n for n in pure_names if n not in self.decay_names]
            pure_drop_names = [n for n in bkg_names if n in pure_bkg_names]
            drop_ids = [pure_bkg_names.index(name) for name in pure_drop_names]
            pc._data_faker.drop(columns=pure_drop_names, inplace=True, errors="ignore")
            pc.mc_matrix.drop(columns=pure_drop_names, inplace=True, errors="ignore")
            pc.bkg_cs_default = np.delete(pc.bkg_cs_default, drop_ids)
        self._set_polarization_dependent_values()

    @property
    def bkg_names(self):
        return [n for n in self.mc_matrix.columns if n not in self.decay_names]

    @bkg_names.setter
    def bkg_names(self, new_names):
        for pc in self._pure_channels.values():
            old_pure_names = pc.bkg_names
            new_pure_names = [
                new_names[i]
                for (i, old_name) in enumerate(self.bkg_names)
                if old_name in old_pure_names
            ]
            pc.bkg_names = new_pure_names
        if len(self.bkg_names) != len(new_names):
            raise Exception(f"{self.bkg_names=}, {new_names=}.")
        for df in [self._data_faker, self.mc_matrix]:  # Order matters!
            df.rename(
                columns={old: new for (old, new) in zip(self.bkg_names, new_names)},
                inplace=True,
            )

    @property
    def box_names(self):
        return self.mc_matrix.index

    @box_names.setter
    def box_names(self, new_names):
        for pc in self._pure_channels.values():
            pc.box_names = new_names
        if len(self.box_names) != len(new_names):
            raise Exception(f"{self.box_names=}, {new_names=}.")
        for df in [self._data_faker, self.mc_matrix]:  # Order matters!
            df.rename(
                index={old: new for (old, new) in zip(self.box_names, new_names)},
                inplace=True,
            )

    def __repr__(self):
        txt = "\n - ".join(
            [f"{self.__class__.__name__} with contributions:"]
            + [f"{k}: {v}" for k, v in self._pure_channels.items()]
        )
        return txt

    def get_expected_counts(self, data_brs=None):
        """Get the expected counts for this channel.

        This uses statistics that are independent from those
        used for the `mc_matrix` in the likelihood building.
        """
        if data_brs is None:
            data_brs = self.data_brs
        else:
            if (
                sum(data_brs) != 1
                or min(data_brs) < 0
                or len(data_brs) != len(self.data_brs)
            ):
                raise DataChannelError(f"Invalid BR hypothesis: {data_brs=}.")
            data_brs = np.array(data_brs)
        cs_signal = data_brs * self.signal_cs_default * self.signal_scaler
        cs = np.concatenate([cs_signal, self.bkg_cs_default])
        expected_process_counts = cs * self.luminosity_ifb
        expected_matrix_counts = self._data_faker * expected_process_counts
        expected_counts = expected_matrix_counts.sum(axis=1)
        return expected_counts

    def get_toys(self, size=None, data_brs=None, rng=None):
        """Smear the expected counts with respect to statistical uncertainties."""
        expected_counts = self.get_expected_counts(data_brs)
        n_data = int(sum(expected_counts))
        box_probabilities = expected_counts / sum(expected_counts)
        if rng is None:
            rng = np.random.default_rng()
        return rng.multinomial(n_data, box_probabilities, size=size)
