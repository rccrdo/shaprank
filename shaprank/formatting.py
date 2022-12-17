"""Text formatting and presentation utilities."""

from typing import Union


def number(x: Union[int, float], decimals: int = 6) -> str:
    """
    Formats a numeric type to a string.

    Parameters:
        x       : The numeric-like obj
        decimals: The number of decimal points to use for presentation

    Returns:
        The sought string representation
    """
    fmt = f"%.{decimals}f"
    result = (fmt % x).rstrip("0").rstrip(".")
    return result
