plotting_keywords = {
    "combine_bkg",
    "combine_boxes",
    "experiment_tag",
    "is_normalized",
    "no_bkg",
    "omit_zero",
}


def basic_kwargs_check(**kwargs):
    """Check that the provided kwargs are keywords used in alldecays.plotting.

    Note that this only warns for keyword arguments that are
    not used in any of the plotting functions.
    Thus it is still possible to pass keyword arguments to the function without
    them having an effect
    (as long as they are valid keywords for another function).

    The idea is that this should prevent misspellings from going unnoticed.
    """
    invalid_kwargs = set(kwargs.keys()) - plotting_keywords

    n_invalid_kwargs = len(invalid_kwargs)
    if n_invalid_kwargs != 0:
        str_inv = "" if n_invalid_kwargs == 1 else f"s({n_invalid_kwargs})"
        print(
            f"Invalid keyword argument{str_inv}: {', '.join(invalid_kwargs)}. "
            "Maybe a misspelling."
        )
