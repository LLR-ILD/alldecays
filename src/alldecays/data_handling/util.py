"""Utility code for polarization handling"""
_polarization_cases = sorted(["eLpL", "eLpR", "eRpL", "eRpR"])


def get_polarization_weights(pol=(0.8, -0.3)):
    """Calculate the polarization weights."""

    def valid_pol(p):
        return p >= -1 and p <= 1

    if len(pol) != 2 or not valid_pol(pol[0]) or not valid_pol(pol[1]):
        raise Exception(
            f"The polarisation {pol} is not understood. Should be "
            "of the same form as (.8,.3)."
        )
    e, p = pol
    pol_weight = {
        "eLpR": (1 - (1 + e) / 2.0) * (1 + p) / 2.0,
        "eRpL": (1 + e) / 2.0 * (1 - (1 + p) / 2.0),
        "eRpR": (1 + e) / 2.0 * (1 + p) / 2.0,
        "eLpL": (1 - (1 + e) / 2.0) * (1 - (1 + p) / 2.0),
    }
    return pol_weight
