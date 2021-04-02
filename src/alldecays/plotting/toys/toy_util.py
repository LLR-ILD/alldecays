from alldecays.fitting.toy_values import ToyValues


def get_valid_toy_values(fit, channel_counts_needed=False):
    """Wrapper for consistent error handling for getting toys from a fit."""
    if hasattr(fit, "toys"):
        toy_values = fit.toys
        assert isinstance(toy_values, ToyValues)
    else:
        raise AttributeError(
            "Plots skipped: Fit passed without throwing toys first, or not a fit."
        )
    if channel_counts_needed and (
        not hasattr(toy_values, "_channel_counts") or toy_values._channel_counts is None
    ):
        raise AttributeError(
            "Plots skipped: _channel_counts not filled for the toys. \n"
            "Set `store_channel_counts=True` in fit.fill_toys (Usually a few \n"
            "(<< 100) toys are enough for diagnostics)."
        )
    return toy_values
