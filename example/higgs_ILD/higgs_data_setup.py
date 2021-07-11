"""This file  shows a typical setup procedure.
"""
import numpy as np

import alldecays

_brs = {
    # "H→ss":     0.00034,
    "H→cc": 0.02718,
    "H→bb": 0.57720,
    "H→μμ": 0.00030,
    "H→ττ": 0.06198,
    "H→Zγ": 0.00170,
    "H→gg": 0.08516 + 0.00034,
    "H→γγ": 0.00242,
    "H→ZZ": 0.02616,
    "H→WW": 0.21756,
}
_brs = dict(
    sorted(_brs.items(), key=lambda item: item[1])[::-1]
)  # Apply BR size sorting.
decay_names = list(_brs.keys())
brs = np.array(list(_brs.values()))

data_set = alldecays.DataSet(decay_names, polarization=None)
data_set.add_channel("higgs", "data/v06_no_overlay.csv")
# data_set.add_channel("higgs2", "data/v06_fake_bkg.csv")
data_set.signal_scaler = 0.1  # Assumed efficiency for the analysis.
data_set.luminosity_ifb = 2000
data_set.data_brs = brs
data_set.fit_start_brs = data_set.data_brs

combi_data_set = alldecays.CombinedDataSet(
    decay_names,
    {"ds": data_set},
    data_brs=data_set.data_brs,
    fit_start_brs=data_set.fit_start_brs,
    signal_scaler=data_set.signal_scaler,
)
