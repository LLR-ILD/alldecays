"""The toy values class."""


class ToyValues:
    """Storage class from results of a toy fit run."""

    def __init__(
        self,
        internal,
        physics,
        valid,
        accurate,
        nfcn,
        fval,
        channel_counts=None,
    ):
        self.internal = internal
        self.physics = physics
        self.valid = valid
        self.accurate = accurate
        self.nfcn = nfcn
        self.fval = fval
        self._channel_counts = channel_counts
        self._validate_lengths()

    def _validate_lengths(self):
        n_toys = len(self)
        assert n_toys == self.internal.shape[0]
        assert n_toys == self.physics.shape[0]
        assert n_toys == self.valid.shape[0]
        assert n_toys == self.accurate.shape[0]
        assert n_toys == self.nfcn.shape[0]
        assert n_toys == self.fval.shape[0]
        if self._channel_counts is not None:
            assert n_toys == len(self._channel_counts)

    def __len__(self):
        return self.physics.shape[0]

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)} draws)"

    def get_copy_after_mask(self, mask):
        """Get a ToyValues object from only the toys passing a mask.

        This can especially be useful for (temporarily) restricting the used
        toys for some plots or calculations.

        Example:
            all_toys = fit.toys
            mask = all_toys.accurate
            accurate_toys = all_toys.get_copy_after_mask(mask)
            fit.toys = accurate_toys
        """
        if self._channel_counts is not None:
            ccc = [cc for i, cc in enumerate(self._channel_counts) if mask[i]]
        else:
            ccc = None
        return ToyValues(
            internal=self.internal[mask],
            physics=self.physics[mask],
            valid=self.valid[mask],
            accurate=self.accurate[mask],
            nfcn=self.nfcn[mask],
            fval=self.fval[mask],
            channel_counts=ccc,
        )
