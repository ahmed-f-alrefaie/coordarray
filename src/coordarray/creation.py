import typing as t

import numpy as np
import numpy.typing as npt

from .core import coordarray

T = t.TypeVar("T")


def coordlike(x: "coordarray", data: npt.ArrayLike) -> "coordarray":
    """Create coordinate array with same coordinates as another array.

    Args:
        x: Array to copy coordinates from
        data: Data to use

    Returns:
        coordarray
    """
    return coordarray(np.asanyarray(data), x.coords)


def zeros_like(x: "coordarray") -> "coordarray":
    """Create coordinate array with same coordinates as another array.

    Args:
        x: Array to copy coordinates from

    Returns:
        coordarray with zeros

    """
    return coordlike(x, np.zeros_like(x.data))


def full_like(x: "coordarray", fill_value: float) -> "coordarray":
    """Create coordinate array with same coordinates as another array.

    Args:
        x: Array to copy coordinates from
        fill_value: Value to fill array with

    Returns:
        coordarray with fill value

    """
    return coordlike(x, np.full_like(x, fill_value))


def ones_like(x: "coordarray") -> "coordarray":
    """Create coordinate array with same coordinates as another array.

    Args:
        x: Array to copy coordinates from

    Returns:
        coordarray with ones

    """
    return coordlike(x, np.ones_like(x))
