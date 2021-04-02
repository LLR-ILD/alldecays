"""All Exception types that can be thrown by alldecays."""


class DataChannelError(Exception):
    """Something did not go as planned when creating/loading a _DataChannel."""

    pass


class DataSetError(Exception):
    """The DataSet cannot be created."""

    pass


class FitException(Exception):
    """The Fit class does not likethe provided arguments."""

    pass


class InvalidFitException(FitException):
    """Raised if Minuit.valid is False."""

    pass
