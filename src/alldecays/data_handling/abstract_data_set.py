"""Common/required functionality for DataSet-like classes."""
from abc import ABC, abstractmethod


class AbstractDataSet(ABC):
    """Defines the functionality that code outside of data_handling can depend on.

    If a class abides to this protocol, it can be used in place of DataSet.
    E.g. CombinedDataSet.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def get_channels(self):
        pass

    @property
    @abstractmethod
    def decay_names(self, process_name):
        pass

    @decay_names.setter
    @abstractmethod
    def decay_names(self, new_names):
        pass

    @property
    @abstractmethod
    def data_brs(self, process_name):
        pass

    @data_brs.setter
    @abstractmethod
    def data_brs(self, new_names):
        pass

    @property
    @abstractmethod
    def fit_start_brs(self, process_name):
        pass

    @fit_start_brs.setter
    @abstractmethod
    def fit_start_brs(self, new_names):
        pass

    @property
    @abstractmethod
    def signal_scaler(self, process_name):
        pass

    @signal_scaler.setter
    @abstractmethod
    def signal_scaler(self, new_names):
        pass
