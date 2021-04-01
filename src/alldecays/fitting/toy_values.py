class ToyValues:
    def __init__(
        self,
        internal,
        physics,
        valid,
        accurate,
        nfcn,
        channel_counts=None,
    ):
        self.internal = internal
        self.physics = physics
        self.valid = valid
        self.accurate = accurate
        self.nfcn = nfcn
        self._channel_counts = channel_counts
        self._validate_lengths()

    def _validate_lengths(self):
        n_toys = len(self)
        assert n_toys == self.internal.shape[0]
        assert n_toys == self.physics.shape[0]
        assert n_toys == self.valid.shape[0]
        assert n_toys == self.accurate.shape[0]
        assert n_toys == self.nfcn.shape[0]
        if self._channel_counts is not None:
            assert n_toys == len(self._channel_counts)

    def __len__(self):
        return self.physics.shape[0]

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)} draws)"
