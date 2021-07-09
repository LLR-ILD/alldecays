import numpy as np
import pytest
from conftest import channel1_path, channel_polarized_path, decay_names

import alldecays


@pytest.mark.parametrize("data_type", ["polarized", "unpolarized"])
def test_name_changes_work(data_type, channel1, channel_polarized):
    channel = dict(polarized=channel_polarized, unpolarized=channel1)[data_type]
    pc0 = next(iter(channel._pure_channels.values()))

    old_decay_names = channel.decay_names
    new_decay_names = ["new" + n for n in channel.decay_names]
    channel.decay_names = new_decay_names
    assert pc0.decay_names == new_decay_names
    channel.decay_names = old_decay_names

    old_bkg_names = channel.bkg_names
    new_bkg_names = ["new" + n for n in channel.bkg_names]
    channel.bkg_names = new_bkg_names
    assert set(pc0.bkg_names).issubset(new_bkg_names)
    channel.bkg_names = old_bkg_names

    old_box_names = channel.box_names
    new_box_names = ["new" + n for n in channel.box_names]
    channel.box_names = new_box_names
    assert list(pc0.box_names) == new_box_names
    channel.box_names = old_box_names


def test_purity_change(channel_polarized):
    old_polarization = channel_polarized.polarization
    channel_polarized.polarization = (1.0, -1.0)
    channel_polarized.polarization = old_polarization


def test_expected_counts(channel_polarized):
    channel = channel_polarized

    box_exp = channel.get_expected_counts().values
    expected_should_be = np.array([6086.7, 5249.6, 2425.8, 4770.2])
    assert box_exp == pytest.approx(expected_should_be, abs=1e-1)

    changed_brs = np.zeros_like(channel.data_brs)
    changed_brs[0] = 1
    box_changed_br = channel.get_expected_counts(data_brs=changed_brs).values
    changed_should_be = np.array([5300.5, 5979.5, 2799.7, 4506.6])
    assert box_changed_br == pytest.approx(changed_should_be, abs=1e-1)


def test_toys(channel_polarized):
    rng = np.random.default_rng(1)
    one_toy = channel_polarized.get_toys(rng=rng)
    toy_should_be = np.array([6049, 5175, 2463, 4845])
    assert (one_toy == toy_should_be).all()

    size = (2, 3)
    toy_sum_expected = np.ones(size) * sum(toy_should_be)
    toy_sum_obtained = channel_polarized.get_toys(size, rng=rng).sum(axis=-1)
    assert (toy_sum_expected == toy_sum_obtained).all()


@pytest.mark.parametrize("data_type", ["polarized", "unpolarized"])
def test_data_set_add_channel(data_type):
    channel_paths = {
        "unpolarized": channel1_path,
        "polarized": channel_polarized_path,
    }
    polarizations = {
        "unpolarized": None,
        "polarized": (-0.8, 0.3),
    }
    ds = alldecays.DataSet(decay_names, polarization=polarizations[data_type])
    ds.add_channel("by_name", channel_paths[data_type])
    ds.add_channel(channel_paths[data_type])
    ds.add_channels(
        {"nameA": channel_paths[data_type], "nameB": channel_paths[data_type]}
    )


def go_through_setters(ds, channel, is_combination=False):
    old_decay_names = ds.decay_names
    new_decay_names = ["new" + n for n in decay_names]
    ds.decay_names = new_decay_names
    assert channel.decay_names == new_decay_names
    ds.decay_names = old_decay_names

    old_data_brs = ds.data_brs
    changed_brs = np.zeros_like(ds.data_brs)
    changed_brs[0] = 1
    ds.data_brs = changed_brs
    assert (changed_brs == channel.data_brs).all()
    ds.data_brs = old_data_brs

    old_signal_scaler = ds.signal_scaler
    new_signal_scaler = 2.5
    ds.signal_scaler = new_signal_scaler
    assert channel.signal_scaler == new_signal_scaler
    ds.signal_scaler = old_signal_scaler

    if is_combination:
        return

    old_polarization = ds.polarization
    new_polarization = (1.0, -0.1)
    ds.polarization = new_polarization
    assert channel.polarization == new_polarization
    ds.polarization = old_polarization

    old_luminosity_ifb = ds.luminosity_ifb
    new_luminosity_ifb = 1.1
    ds.luminosity_ifb = new_luminosity_ifb
    assert channel.luminosity_ifb == new_luminosity_ifb
    ds.luminosity_ifb = old_luminosity_ifb


def test_data_set_setters():
    ds = alldecays.DataSet(decay_names, polarization=(-0.8, 0.3))
    ds.add_channel("my_channel", channel_polarized_path)
    channel = ds._channels["my_channel"]
    go_through_setters(ds, channel)


def test_data_set_combined():
    ds1 = alldecays.DataSet(decay_names, polarization=(-0.8, 0.3))
    ds2 = alldecays.DataSet(decay_names, polarization=(0, 0))
    ds1.add_channel("my_channel", channel_polarized_path)
    ds2.add_channel("my_channel", channel_polarized_path)
    ds2.add_channel("one_more_channel", channel_polarized_path)

    combined = alldecays.CombinedDataSet(decay_names, {"ds1": ds1, "ds2": ds2})
    combined.add_data_sets({"ds1_copy": ds1})

    channel = combined._channels["ds1:my_channel"]
    go_through_setters(combined, channel, is_combination=True)


def test_data_set_subclassing():
    from alldecays.data_handling.abstract_data_set import AbstractDataSet

    assert isinstance(alldecays.DataSet(decay_names), AbstractDataSet)
    assert isinstance(alldecays.CombinedDataSet(decay_names), AbstractDataSet)
