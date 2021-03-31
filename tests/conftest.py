import datetime as dt
from pathlib import Path
import pytest

import alldecays
from alldecays.data_handling.data_channel import _DataChannel


channel1_path = Path(__file__).parent / "data/unpolarized/channel1.csv"
channel2_path = Path(__file__).parent / "data/unpolarized/channel2.csv"
channel_polarized_path = Path(__file__).parent / "data/polarized/channel1"
decay_names = [f"dec{i}" for i in "ABC"]


@pytest.fixture(scope="module")
def test_plot_dir():
    timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    _test_plot_dir = Path(__file__).parent / "tmp" / timestamp
    _test_plot_dir.mkdir(exist_ok=True, parents=True)
    return _test_plot_dir


@pytest.fixture(scope="module")
def channel1():
    return _DataChannel(channel1_path, decay_names=decay_names)


@pytest.fixture(scope="module")
def channel2():
    return _DataChannel(channel2_path, decay_names=decay_names)


@pytest.fixture(scope="module")
def channel_polarized():
    polarization = (-0.8, 0.3)
    return _DataChannel(channel_polarized_path, decay_names, polarization)


@pytest.fixture(scope="module")
def data_set1(channel1):
    data_set = alldecays.DataSet(decay_names=decay_names)
    data_set.add_channel("no_pol", channel1_path)
    return data_set
