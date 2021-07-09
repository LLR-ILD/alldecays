"""Pure data channel class.

Has a fixed polarization at creation.
A DataChannel uses one or more of these under the hood.
"""
from pathlib import Path

import numpy as np
import pandas as pd

cross_section_column = "cross section [fb]"
unselected_column = "unselected"
bookkeeping_columns = (cross_section_column, unselected_column)


class _PureDataChannel:
    """Wrapper for a single .csv data file.

    Contains the data from one initial polarization for one channel.
    This usually is stored in a single file on disk.
    For internal usage.
    """

    def __init__(self, channel_path, decay_names):
        self._channel_path = channel_path
        self._decay_names = decay_names
        df = self._get_dataframe()

        cs_default = self._get_default_cross_sections(df)
        self.signal_cs_default = cs_default[0]
        self.bkg_cs_default = cs_default[1:]

        probability_matrices = self._get_probabilities(df)
        self.mc_matrix = probability_matrices[0]
        self._data_faker = probability_matrices[1]

    @property
    def decay_names(self):
        return self._decay_names

    @decay_names.setter
    def decay_names(self, new_names):
        if len(self.decay_names) != len(new_names):
            raise Exception(f"{self.decay_names=}, {new_names=}.")
        for df in [self._data_faker, self.mc_matrix]:  # Order matters!
            df.rename(
                columns={old: new for (old, new) in zip(self.decay_names, new_names)},
                inplace=True,
            )
        self._decay_names = new_names

    @property
    def bkg_names(self):
        return [n for n in self.mc_matrix.columns if n not in self.decay_names]

    @bkg_names.setter
    def bkg_names(self, new_names):
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
        if len(self.box_names) != len(new_names):
            raise Exception(f"{self.box_names=}, {new_names=}.")
        for df in [self._data_faker, self.mc_matrix]:  # Order matters!
            df.rename(
                index={old: new for (old, new) in zip(self.box_names, new_names)},
                inplace=True,
            )

    def _get_dataframe(self):
        csv_path = Path(self._channel_path)
        if csv_path.is_dir():
            dir_files = list(csv_path.glob("*.csv"))
            if len(dir_files) == 1:
                csv_path = dir_files[0]
            else:
                if len(dir_files) == 0:
                    file_str = f" in {str(csv_path)}."
                else:
                    file_str = "\n    ".join(map(str, [":"] + dir_files))
                txt = f"{len(dir_files)} candidate files found" + file_str
                raise Exception(txt)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        if not csv_path.suffix == ".csv":
            raise NotImplementedError(csv_path)

        df = pd.read_csv(csv_path, index_col=0)
        df = self._order_processes(df)
        return df

    def _order_processes(self, df):
        if not set(self.decay_names).issubset(df.index):
            txt = "Not all decay names were found in the channel:"
            txt += "\n" + f"{sorted(self.decay_names)=}"
            txt += "\n" + f"{sorted(df.index.values)=}"
            txt += "\nIf this is expected behavior (e.g. for a specific pure polarization),"
            txt += "\nsimply add zero-count rows for each channel."
            raise Exception(txt)
        df = df.reindex(
            self.decay_names + sorted(i for i in df.index if i not in self.decay_names)
        )
        return df

    def _get_default_cross_sections(self, df):
        """
        TODO: Rethink this implementation.
        In the current implementation,
        it is irrelevant how the cross section is distributed
        amongst the decay modes:
        Only the sum of the signal cross sections in the channel matters.
        This might not be the ideal construction.
        Inconsistencies in Monte Carlo branching ratios
        between polarizations or channels
        can currently not be checked.
        """
        signal_strength = sum(
            [df.loc[bkg, cross_section_column] for bkg in self.decay_names]
        )
        bkg_strengths = [
            df.loc[bkg, cross_section_column]
            for bkg in df.index
            if bkg not in self.decay_names
        ]
        return np.array([signal_strength] + bkg_strengths)

    def _get_probabilities_from_counts(self, counts):
        """Translate the counts into a probability matrix.

        One row is allocated per category in the sample.
        The row for the unselected events is omitted.
        Thus, the columns will not necessarily sum to 1 (but a smaller value).
        Each column features one decay mode or bkg process.

        Layout example:

                  decA      decB      decC      bkg1  ...
        box1  0.104579  0.167413  0.246389  0.168419
        box2  0.267327                      0.177159
         ...                                          ...
        """
        box_columns = [c for c in counts.columns if c not in bookkeeping_columns]
        process_counts = counts[[unselected_column] + box_columns].sum(axis=1)
        # This division create NaN values for processes with sum zero.
        # We _want_ this: they are only useful when combining pure polarizations
        # and indicate that this column was not present.
        proba = counts[box_columns].T / process_counts
        return proba

    def _get_probabilities(self, df):
        for required_column in [cross_section_column, unselected_column]:
            if required_column not in df.columns:
                txt = f"Required data column is missing: {required_column}. "
                txt += f"File: {self._channel_path}."
                raise Exception(txt)
        train_counts, test_counts = self._split_monte_carlo_into_two(
            df[[c for c in df.columns if c != cross_section_column]]
        )
        train_proba = self._get_probabilities_from_counts(train_counts)
        test_proba = self._get_probabilities_from_counts(test_counts)
        return train_proba, test_proba

    def _split_monte_carlo_into_two(self, df):
        """Share the available MC events in a fair way between two uses.

        1. MC data for populating the probability matrix needed in the fit.
        2. MC data that is used to generate the detector/expected data.

        Using statistically independent events for these two cases allows us
        to evaluate the contribution to the uncertainty of the analysis
        from limited/finite simulated MC statistics.
        """
        test_fraction = 0.5
        rng = np.random.default_rng(seed=1)  # Fixed for reproducibility.
        test_rows = {}
        for process, row in df.iterrows():
            mc_counts = row.astype(int)
            try:
                test_rows[process] = rng.multivariate_hypergeometric(
                    mc_counts, int(test_fraction * sum(mc_counts))
                )
            except ValueError as ve:
                print(
                    "\n".join(
                        [
                            "Error is expected if MC counts are extremely high (in the billions).",
                            f"{self._channel_path=}, {mc_counts=}.",
                            "This could be handled by adapting the random drawer,",
                            "but was deemed unlikely to happen and thus not implemented.",
                        ]
                    )
                )
                raise ve
        test_counts = pd.DataFrame.from_dict(test_rows, orient="index")
        test_counts.columns = df.columns
        train_counts = df.astype(int) - test_counts
        return train_counts, test_counts

    def __repr__(self):
        txt = f"{self.__class__.__name__} "
        txt += f"with data from {str(self._channel_path)}."
        return txt
