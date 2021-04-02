"""Data set class.

These objects are used throughout the module to interact with the physics data.
"""
import numpy as np
from pathlib import Path

from alldecays.exceptions import DataSetError
from .abstract_data_set import AbstractDataSet
from .data_channel import _DataChannel


class DataSet(AbstractDataSet):
    """Defines the input protocol for the fitting step.

    Example:
        >>> import alldecays
        >>> decay_names = ["X→AA", "X→BB", "X→CC"]
        >>> pol_dir = "/path/to/polarized/files/directory"
        >>> ds = alldecays.DataSet(decay_names, polarization=(-0.8, 0.3))
        >>> ds.add_channel("channel1", pol_dir)

    This assumes files `eLpL.csv`, `eLpR.csv`, `eRpL.csv`, `eRpR.csv`
    in `pol_dir` with at least the `decay_names` rows.

    As a design choice, channels are added by the path to their data file.
    This emphasizes that `_DataChannel`s are only meant to be used internally.
    You should be careful when modifying `_DataChannel` objects directly.
    Args:
        data_brs: default is  flat branching ratio.
        fit_start_brs: If not specified, defaults to `data_brs`.
    """

    def __init__(
        self,
        decay_names,
        polarization=None,
        data_brs=None,
        fit_start_brs=None,
        luminosity_ifb=1_000,
        signal_scaler=1.0,
    ):
        self._channels = {}
        self._decay_names = decay_names
        self._polarization = polarization
        self._data_brs = self._set_brs(data_brs)
        if fit_start_brs is None:
            self._fit_start_brs = np.array(self._data_brs)
        else:
            self._fit_start_brs = self._set_brs(fit_start_brs)
        self._luminosity_ifb = luminosity_ifb
        self._signal_scaler = signal_scaler

    def get_channels(self):
        """Return a dict of all channels."""
        return self._channels

    @property
    def decay_names(self):
        return self._decay_names

    @decay_names.setter
    def decay_names(self, new_names):
        if len(self.decay_names) != len(new_names):
            raise Exception(f"{self.decay_names=}, {new_names=}.")
        for dc in self._channels.values():
            dc.decay_names = new_names
        self._decay_names = new_names

    @property
    def polarization(self):
        return self._polarization

    @polarization.setter
    def polarization(self, new_pol):
        for dc in self._channels.values():
            dc.polarization = new_pol
        self._polarization = new_pol

    def _set_brs(self, brs=None):
        n_decays = len(self.decay_names)
        if brs is None:
            return np.ones(n_decays) / n_decays
        if len(brs) != n_decays:
            raise DataSetError(f"{brs=}, {n_decays=}.")
        return brs

    @property
    def data_brs(self):
        return self._data_brs

    @data_brs.setter
    def data_brs(self, new_brs):
        for dc in self._channels.values():
            dc.data_brs = new_brs
        self._data_brs = new_brs

    @property
    def fit_start_brs(self):
        return self._fit_start_brs

    @fit_start_brs.setter
    def fit_start_brs(self, new_brs):
        self._fit_start_brs = new_brs

    @property
    def luminosity_ifb(self):
        return self._luminosity_ifb

    @luminosity_ifb.setter
    def luminosity_ifb(self, new_value):
        for dc in self._channels.values():
            dc.luminosity_ifb = new_value
        self._luminosity_ifb = new_value

    @property
    def signal_scaler(self):
        return self._signal_scaler

    @signal_scaler.setter
    def signal_scaler(self, new_value):
        for dc in self._channels.values():
            dc.signal_scaler = new_value
        self._signal_scaler = new_value

    def add_channel(self, name, channel_path=None):
        """Add a channel to the DataSet."""
        if channel_path is None:
            channel_path = Path(name)
            name = channel_path.stem
        name = str(name)
        if name in self._channels:
            raise DataSetError(
                f"A channel with {name=} already exists in the "
                f"{self.__class__.__name__}: {list(self._channels.keys())}."
            )
        self._channels[name] = _DataChannel(
            channel_path,
            self.decay_names,
            polarization=self.polarization,
            data_brs=self._data_brs,
            luminosity_ifb=self._luminosity_ifb,
            signal_scaler=self.signal_scaler,
        )

    def add_channels(self, channel_path_dict):
        """Convenience wrapper around `add_channel`."""
        for name, channel_path in channel_path_dict.items():
            self.add_channel(name, channel_path)

    def drop_channels(self, names):
        """Remove channels from the DataSet by name."""
        for name in names:
            self._channels.pop(name)

    def __repr__(self):
        n_channels = len(self._channels)
        text = f"{self.__class__.__name__} with {n_channels} channels.\n"
        if n_channels != 0:
            text += f"  Channel names: {list(self._channels.keys())}.\n"

        if self.polarization is None:
            pol_str = "unpolarized."
        else:
            pol_str = f"with polarization (e-, e+)=({int(100*self.polarization[0])}%, {int(100*self.polarization[1])}%)."
        text += f"  Luminosity: {self.luminosity_ifb} ifb {pol_str}\n"
        if self.signal_scaler != 1:
            text += f"  The signal strength is rescaled by {self.signal_scaler}.\n"
        text += f"  Considered signal decays: {self.decay_names}.\n"
        return text
