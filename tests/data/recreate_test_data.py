#!/usr/bin/env python

"""Script that produced the test data.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def create_test_data():
    rng = np.random.default_rng(seed=1)
    test_data = Path(__file__).parent

    boxes = ["unselected"] + [f"box{i}" for i in range(1, 5)]
    polarizations = ["eLpL", "eLpR", "eRpL", "eRpR"]
    decays = [f"dec{i}" for i in "ABC"]
    processes = decays + [f"bkg{i}" for i in range(1, 4)]
    for i in range(1, 3):  # 2 channels.
        pure_polarization = {}
        for p in polarizations:
            high = 100 if p in ["eLpL", "eRpR"] else 1000
            pure_polarization[f"{p}"] = pd.DataFrame(
                rng.integers(high, size=(len(processes), len(boxes))),
                columns=boxes,
                index=processes,
            )
            pure_df = pure_polarization[f"{p}"]
            pure_df["unselected"] += [
                0 if process.startswith("dec") else rng.integers(2 * high)
                for process in processes
            ]
            # Add cross section information.
            cs_col = "cross section [fb]"
            pure_df.insert(0, cs_col, 5 * rng.random(size=len(processes)))
            # Enforce consistent signal cross sections
            # for the polarizations and channel.
            signal_cs = pure_df.loc[decays, cs_col].sum()
            br_example = np.arange(1, len(decays) + 1) / sum(
                np.arange(1, len(decays) + 1)
            )
            pure_df.loc[decays, cs_col] = br_example * signal_cs

            # Alter the second channel a bit to show what is possible.
            if i == 2:
                pure_df.drop(columns=[boxes[-1]], inplace=True)
                pure_df.rename(
                    index={"bkg1": "bkg4"},
                    columns={
                        old_name: f"bix{old_name[-1]}"
                        for old_name in boxes
                        if old_name.startswith("box")
                    },
                    inplace=True,
                )

        pure_polarization["eRpR"].drop(decays + ["bkg3"], inplace=True)
        pure_polarization["eRpL"].drop(["bkg2"], inplace=True)
        pure_polarization["eLpL"].drop(["bkg2"], inplace=True)

        polarized_dir = test_data / "polarized" / f"channel{i}"
        polarized_dir.mkdir(exist_ok=True, parents=True)
        unpolarized_dir = test_data / "unpolarized"
        unpolarized_dir.mkdir(exist_ok=True, parents=True)

        def ensure_rows_for_decays(df):
            if set(decays).issubset(df.index):
                return df
            if len(set(decays).intersection(df.index)) > 0:
                raise Exception(f"{decays=}, {df.index=}.")
            return df.append(pd.DataFrame(0, columns=df.columns, index=decays))

        unpolarized = pd.DataFrame()
        for p in polarizations:
            df = ensure_rows_for_decays(pure_polarization[p])
            df.to_csv(polarized_dir / f"{p}.csv")
            unpolarized = unpolarized.add(df, fill_value=0)
        unpolarized.to_csv(unpolarized_dir / f"channel{i}.csv")


if __name__ == "__main__":
    create_test_data()
