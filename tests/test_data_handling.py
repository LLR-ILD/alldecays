import numpy as np
from pathlib import Path
import pytest

import alldecays
from alldecays.data_handling.data_channel import _DataChannel


test_data_path = Path(__file__).parent / "data"
channel_paths = {
    "unpolarized": test_data_path / "unpolarized/channel1.csv",
    "polarized": test_data_path / "polarized/channel1",
}
polarizations = {
    "unpolarized": None,
    "polarized": (-0.8, 0.3),
}
decay_names = [f"dec{i}" for i in "ABC"]


def get_test_data_channel(data_type="polarized"):
    return _DataChannel(channel_paths[data_type], decay_names, polarizations[data_type])


@pytest.mark.parametrize("data_type", ["polarized", "unpolarized"])
def test_name_changes_work(data_type):
    dc = get_test_data_channel(data_type)
    pc0 = next(iter(dc._pure_channels.values()))

    new_decay_names = ["new" + n for n in decay_names]
    dc.decay_names = new_decay_names
    assert pc0.decay_names == new_decay_names

    new_bkg_names = ["new" + n for n in dc.bkg_names]
    dc.bkg_names = new_bkg_names
    assert set(pc0.bkg_names).issubset(new_bkg_names)

    new_box_names = ["new" + n for n in dc.box_names]
    dc.box_names = new_box_names
    assert list(pc0.box_names) == new_box_names


def test_purity_change():
    dc = get_test_data_channel("polarized")
    dc.polarization = (1.0, -1.0)


def test_expected_counts():
    dc = get_test_data_channel()
    box_exp = dc.get_expected_counts().values
    expected_should_be = np.array([6086.7, 5249.6, 2425.8, 4770.2])
    assert box_exp == pytest.approx(expected_should_be, abs=1e-1)

    changed_brs = np.zeros_like(dc.data_brs)
    changed_brs[0] = 1
    box_changed_br = dc.get_expected_counts(data_brs=changed_brs).values
    changed_should_be = np.array([5300.5, 5979.5, 2799.7, 4506.6])
    assert box_changed_br == pytest.approx(changed_should_be, abs=1e-1)


def test_toys():
    rng = np.random.default_rng(1)
    dc = get_test_data_channel()
    one_toy = dc.get_toys(rng=rng)
    toy_should_be = np.array([6049, 5175, 2463, 4845])
    assert (one_toy == toy_should_be).all()

    size = (2, 3)
    toy_sum_expected = np.ones(size) * sum(toy_should_be)
    assert (toy_sum_expected == dc.get_toys(size, rng=rng).sum(axis=-1)).all()


@pytest.mark.parametrize("data_type", ["polarized", "unpolarized"])
def test_data_set_add_channel(data_type):
    ds = alldecays.DataSet(decay_names, polarization=polarizations[data_type])
    ds.add_channel("by_name", channel_paths[data_type])
    ds.add_channel(channel_paths[data_type])
    ds.add_channels(
        {"nameA": channel_paths[data_type], "nameB": channel_paths[data_type]}
    )


def go_through_setters(ds, channel, is_combination=False):
    new_decay_names = ["new" + n for n in decay_names]
    ds.decay_names = new_decay_names
    assert channel.decay_names == new_decay_names

    changed_brs = np.zeros_like(ds.data_brs)
    changed_brs[0] = 1
    ds.data_brs = changed_brs
    assert (changed_brs == channel.data_brs).all()

    new_signal_scaler = 2.5
    ds.signal_scaler = new_signal_scaler
    assert channel.signal_scaler == new_signal_scaler

    if is_combination:
        return

    new_polarization = (1.0, -0.1)
    ds.polarization = new_polarization
    assert channel.polarization == new_polarization

    new_luminosity_ifb = 1.1
    ds.luminosity_ifb = new_luminosity_ifb
    assert channel.luminosity_ifb == new_luminosity_ifb


def test_data_set_setters():
    ds = alldecays.DataSet(decay_names, polarizations["polarized"])
    ds.add_channel("my_channel", channel_paths["polarized"])
    channel = ds._channels["my_channel"]
    go_through_setters(ds, channel)


def test_data_set_combined():
    ds1 = alldecays.DataSet(decay_names, polarizations["polarized"])
    ds2 = alldecays.DataSet(decay_names, (0, 0))
    ds1.add_channel("my_channel", channel_paths["polarized"])
    ds2.add_channel("my_channel", channel_paths["polarized"])
    ds2.add_channel("one_more_channel", channel_paths["polarized"])

    combined = alldecays.CombinedDataSet(decay_names, {"ds1": ds1, "ds2": ds2})
    combined.add_data_sets({"ds1_copy": ds1})

    channel = combined._channels["ds1:my_channel"]
    go_through_setters(combined, channel, is_combination=True)
