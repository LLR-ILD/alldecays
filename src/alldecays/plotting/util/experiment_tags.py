"""Provide some tag/watermarking choices for plots."""


def ild_preliminary(ax):
    """ILD preliminary watermark."""
    ax.text(
        0.99,
        0.99,
        "ILD preliminary",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        color="gray",
        weight="bold",
        alpha=0.6,
        fontsize=16,
    )


provided_experiment_tags = {
    "ILD_preliminary": ild_preliminary,
}


def do_nothing(ax):
    """Do not add any watermark."""
    return ax


def get_experiment_tag(name):
    """Interfaces to callables that add a tag to the matplotlib axis.

    This is a light-weight approach to a watermarking of a plot in a way
    that is common in particle physics experiments and groups.
    `name` can be an identifier for one of the styles provided here.
    Alternatively, a custom callable can be defined.
    By using this function we have a common interface for both cases.
    """
    if name in provided_experiment_tags:
        return provided_experiment_tags[name]
    elif callable(name):
        # This option allows providing your own tags.
        return name
    else:
        valid_keys = ", ".join(provided_experiment_tags)
        print(
            f"Ignored invalid experiment tag: {name}. " f"Choose one of: {valid_keys}."
        )
        return do_nothing
