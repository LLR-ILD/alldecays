"""Combination of data sets class"""
import numpy as np

from alldecays.exceptions import DataSetError
from .abstract_data_set import AbstractDataSet


class CombinedDataSet(AbstractDataSet):
    """Convenience wrapper around multiple `DataSet` objects.

    Example:
        >>> import alldecays
        >>> isinstance(ds1, alldecays.DataSet) and isinstance(ds2, alldecays.DataSet)
        True
        >>> combined = alldecays.CombinedDataSet(decay_names, {"ds1": ds1})
        >>> combined.add_data_sets({"ds2": ds2})
    """

    def __init__(
        self,
        decay_names,
        data_sets=None,
        data_brs=None,
        fit_start_brs=None,
        signal_scaler=1.0,
    ):
        self._decay_names = decay_names
        self._data_brs = self._set_brs(data_brs)
        if fit_start_brs is None:
            self._fit_start_brs = np.array(self._data_brs)
        else:
            self._fit_start_brs = self._set_brs(fit_start_brs)
        self._signal_scaler = signal_scaler
        self._data_sets = data_sets if data_sets is not None else {}
        for ds in self._data_sets.values():
            self._validate_data_set(ds)

    def _validate_data_set(self, ds):
        """Validate that a dataset fits to this CombinedDataSet."""
        assert self._decay_names == ds._decay_names
        # Note: You get an AssertionError when you initiate the CombinedDataSet
        # with the default values, but try to add a DataSet with non-default values.
        assert (self._data_brs == ds._data_brs).all()
        assert (self.fit_start_brs == ds.fit_start_brs).all()
        assert self.signal_scaler == ds.signal_scaler

    @property
    def _channels(self):
        channels = {}
        for prefix in self._data_sets:
            for name, channel in self._data_sets[prefix].get_channels().items():
                n = f"{prefix}:{name}"
                if n in channels:
                    raise DataSetError(f"Multiple channels with same name: {n}.")
                channels[n] = channel
        return channels

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
        for ds in self._data_sets.values():
            ds.decay_names = new_names
        self._decay_names = new_names

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
        for ds in self._data_sets.values():
            ds.data_brs = new_brs
        self._data_brs = new_brs

    @property
    def fit_start_brs(self):
        return self._fit_start_brs

    @fit_start_brs.setter
    def fit_start_brs(self, new_brs):
        for ds in self._data_sets.values():
            ds.fit_start_brs = new_brs
        self._fit_start_brs = new_brs

    @property
    def signal_scaler(self):
        return self._signal_scaler

    @signal_scaler.setter
    def signal_scaler(self, new_value):
        for ds in self._data_sets.values():
            ds.signal_scaler = new_value
        self._signal_scaler = new_value

    def add_data_sets(self, data_set_dict):
        """Combine the specified DataSet(s) into this CombinedDataSet."""
        for prefix, ds in data_set_dict.items():
            self._validate_data_set(ds)
            if prefix in self._data_sets:
                raise DataSetError(f"A DataSet with {prefix=} already exists.")
            self._data_sets[prefix] = ds

    def __repr__(self):
        n_channels = len(self.get_channels())
        n_data_sets = len(self._data_sets)
        text = f"{self.__class__.__name__} with {n_channels} channels.\n"
        text += f"  {n_data_sets} DataSet objects: {list(self._data_sets)}.\n"
        if self.signal_scaler != 1:
            text += f"  The signal strength is rescaled by {self.signal_scaler}.\n"
        text += f"  Considered signal decays: {self.decay_names}.\n"
        return text
