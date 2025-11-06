# Interpolation routines
import typing as t
from typing import Union

import numpy as np
import numpy.typing as npt

from .core import coordarray


def find_nearest_points_pair(
    array: npt.NDArray[np.number], values: Union[float, int, npt.NDArray[np.number]]
) -> npt.NDArray[np.int64]:
    """Find the indices of the nearest points surrounding a given value in a sorted array.

    Args:
        array: A 1D sorted array of numeric values.
        values: A single numeric value or an array of numeric values to find nearest points for.

    Returns:
        An array of shape (2,) where N is the number of input values. Each row contains the indices
        of the two nearest points in the input array that surround the corresponding value.
    """
    values = np.atleast_1d(values)

    below_bounds = values < array.min()
    above_bounds = values > array.max()

    da = np.abs(array[:, None] - values[None, :])
    res = np.argsort(da, axis=0)[:2]

    res[0, below_bounds] = 0
    res[1, below_bounds] = 0
    res[0, above_bounds] = len(array) - 1
    res[1, above_bounds] = len(array) - 1

    return res.T


def coordarray_subset(array: coordarray, new_coordinates: dict[str, npt.NDArray[np.number]]) -> coordarray:
    """Find the subset of a coordarray that contains the nearest points for specified new coordinates.

    Args:
        array: The original coordarray.
        new_coordinates: A dictionary where keys are dimension names and values are arrays of new coordinates.
    Returns:
        A coordarray containing the nearest points for the specified new coordinates.
    """
    # Ensure that new_coordinates keys are valid dimensions
    # Chaeck that the new coordinate keys exist in the original array
    array_keys = set(array.coords.keys())
    new_keys = set(new_coordinates.keys())
    if not new_keys.issubset(array_keys):
        raise ValueError("New coordinate keys must exist in the original coordarray.")

    smallest_array = array

    for dim, new_coords in new_coordinates.items():
        array_coords = smallest_array.coords[dim]
        pairs = find_nearest_points_pair(array_coords, new_coords)
        pairs_unique = np.unique(pairs.ravel())
        smallest_array = smallest_array[dim, pairs_unique]
    return smallest_array


def bilinear_interpolate(
    data: npt.NDArray[np.float64],
    x: t.Union[npt.NDArray[np.float64], float],
    y: t.Union[npt.NDArray[np.float64], float],
    x_coord: npt.NDArray[np.float64],
    y_coord: npt.NDArray[np.float64],
    axes: tuple[int, int] = (0, 1),
    mode: t.Literal["zero", "hold"] = "hold",
) -> npt.NDArray[np.float64]:
    """Bilinear interpolation.

    Compatible with any numpy-like array

    Args:
        x: x values to interpolate
        y: y values to interpolate
        x_coord: x coordinates of data
        y_coord: y coordinates of data
        data: data to interpolate
        axes: axes to interpolate over
        mode: mode to use for extrapolation

    Returns:
        npt.NDArray[np.float64]: interpolated data

    Raises:
        ValueError: If data is not at least 2D

    """
    if data.ndim < 2:
        raise ValueError("Data must be at least 2D")

    min_x, max_x = x_coord.min(), x_coord.max()
    min_y, max_y = y_coord.min(), y_coord.max()

    x_ravel = x.ravel()
    y_ravel = y.ravel()

    idx_x1 = x_coord.searchsorted(x_ravel, side="right")
    idx_y1 = y_coord.searchsorted(y_ravel, side="right")
    idx_x1 = idx_x1.clip(1, len(x_coord) - 1)
    idx_y1 = idx_y1.clip(1, len(y_coord) - 1)
    idx_x0 = idx_x1 - 1
    idx_y0 = idx_y1 - 1

    x_ravel = x_ravel.clip(min_x, max_x)
    y_ravel = y_ravel.clip(min_y, max_y)

    # ia = data.take(idx_x0, axis=axes[0]).take(idx_y0, axis=axes[1])
    # ib = data.take(idx_x1, axis=axes[0]).take(idx_y0, axis=axes[1])
    # ic = data.take(idx_x0, axis=axes[0]).take(idx_y1, axis=axes[1])
    # id = data.take(idx_x1, axis=axes[0]).take(idx_y1, axis=axes[1])

    if axes[0] != 0:
        data = data.swapaxes(axes[0], 0)
    if axes[1] != 1:
        data = data.swapaxes(axes[1], 1)

    ia = data[idx_x0, idx_y0]
    ib = data[idx_x1, idx_y0]
    ic = data[idx_x0, idx_y1]
    id = data[idx_x1, idx_y1]  # noqa: A001

    x1 = x_coord[idx_x1]
    x0 = x_coord[idx_x0]
    y1 = y_coord[idx_y1]
    y0 = y_coord[idx_y0]

    factor = (x1 - x0) * (y1 - y0)
    wa = (x1 - x_ravel) * (y1 - y_ravel)
    wb = (x_ravel - x0) * (y1 - y_ravel)
    wc = (x1 - x_ravel) * (y_ravel - y0)
    wd = (x_ravel - x0) * (y_ravel - y0)

    diff = 0

    if wa.ndim != ia.ndim:
        # Add appropriate dimensions to end
        diff = ia.ndim - wa.ndim
        wa = wa.reshape(*wa.shape, *[1] * diff)
        wb = wb.reshape(*wb.shape, *[1] * diff)
        wc = wc.reshape(*wc.shape, *[1] * diff)
        wd = wd.reshape(*wd.shape, *[1] * diff)

        factor = factor.reshape(*factor.shape, *[1] * diff)

    result = (wa * ia + wb * ib + wc * ic + wd * id) / factor

    return result.reshape(*x.shape, *data.shape[2:])


def linear_interpolation(
    data: npt.NDArray[np.float64],
    x: t.Union[npt.NDArray[np.float64], float],
    x_coord: npt.NDArray[np.float64],
    axis: int = 0,
    mode: t.Literal["zero", "hold"] = "hold",
) -> npt.NDArray[np.float64]:
    """Linear interpolation.

    Compatible with any numpy-like array

    Args:
        data: data to interpolate
        x: x values to interpolate
        x_coord: x coordinates of data
        axis: axis to interpolate over
        mode: mode to use for extrapolation

    Returns:
        npt.NDArray[np.float64]: interpolated data

    """
    x_ravel = x.ravel()

    x_min, x_max = x_coord.min(), x_coord.max()

    idx_x1 = x_coord.searchsorted(x_ravel, side="right")
    idx_x1 = idx_x1.clip(1, len(x_coord) - 1)
    idx_x0 = idx_x1 - 1

    x1 = x_coord[idx_x1]
    x0 = x_coord[idx_x0]
    if axis != 0:
        data = data.swapaxes(axis, 0)

    factor = x1 - x0

    x_ravel = x_ravel.clip(x_min, x_max)

    wa = np.zeros(x_ravel.shape)
    wb = np.zeros(x_ravel.shape)

    ia = np.zeros(x_ravel.shape + data.shape[1:])
    ib = np.zeros(x_ravel.shape + data.shape[1:])

    ia = data.take(idx_x0, axis=axis)
    ib = data.take(idx_x1, axis=axis)

    wa = (x1 - x_ravel) / factor
    wb = (x_ravel - x0) / factor

    if wa.ndim != ia.ndim:
        # Add appropriate dimensions to end
        diff = ia.ndim - wa.ndim
        wa = wa.reshape(*wa.shape, *[1] * diff)
        wb = wb.reshape(*wb.shape, *[1] * diff)

        factor = factor.reshape(*factor.shape, *[1] * diff)

    result = wa * ia + wb * ib
    return result.reshape(*x.shape, *data.shape[1:])


def trilinear_interpolation(
    data: npt.NDArray[np.float64],
    x: t.Union[npt.NDArray[np.float64], float],
    y: t.Union[npt.NDArray[np.float64], float],
    z: t.Union[npt.NDArray[np.float64], float],
    x_coord: npt.NDArray[np.float64],
    y_coord: npt.NDArray[np.float64],
    z_coord: npt.NDArray[np.float64],
    axes: tuple[int, int, int] = (0, 1, 2),
    mode: t.Literal["zero", "hold"] = "hold",
) -> npt.NDArray[np.float64]:
    """Trilinear interpolation."""
    x_ravel = x.ravel()
    y_ravel = y.ravel()
    z_ravel = z.ravel()

    idx_x1 = x_coord.searchsorted(x_ravel, side="right")
    idx_y1 = y_coord.searchsorted(y_ravel, side="right")
    idx_z1 = z_coord.searchsorted(z_ravel, side="right")

    idx_x1 = idx_x1.clip(1, len(x_coord) - 1)
    idx_y1 = idx_y1.clip(1, len(y_coord) - 1)
    idx_z1 = idx_z1.clip(1, len(z_coord) - 1)

    idx_x0 = idx_x1 - 1
    idx_y0 = idx_y1 - 1
    idx_z0 = idx_z1 - 1

    x1 = x_coord[idx_x1]
    x0 = x_coord[idx_x0]
    y1 = y_coord[idx_y1]
    y0 = y_coord[idx_y0]
    z1 = z_coord[idx_z1]
    z0 = z_coord[idx_z0]

    factor = (x1 - x0) * (y1 - y0) * (z1 - z0)

    x_ravel = x_ravel.clip(x0, x1)
    y_ravel = y_ravel.clip(y0, y1)
    z_ravel = z_ravel.clip(z0, z1)

    wa = (x1 - x_ravel) * (y1 - y_ravel) * (z1 - z_ravel)
    wb = (x_ravel - x0) * (y1 - y_ravel) * (z1 - z_ravel)
    wc = (x1 - x_ravel) * (y_ravel - y0) * (z1 - z_ravel)
    wd = (x_ravel - x0) * (y_ravel - y0) * (z1 - z_ravel)
    we = (x1 - x_ravel) * (y1 - y_ravel) * (z_ravel - z0)
    wf = (x_ravel - x0) * (y1 - y_ravel) * (z_ravel - z0)
    wg = (x1 - x_ravel) * (y_ravel - y0) * (z_ravel - z0)
    wh = (x_ravel - x0) * (y_ravel - y0) * (z_ravel - z0)

    ia = data.take(idx_x0, axis=axes[0]).take(idx_y0, axis=axes[1]).take(idx_z0, axis=axes[2])
    ib = data.take(idx_x1, axis=axes[0]).take(idx_y0, axis=axes[1]).take(idx_z0, axis=axes[2])
    ic = data.take(idx_x0, axis=axes[0]).take(idx_y1, axis=axes[1]).take(idx_z0, axis=axes[2])
    id = data.take(idx_x1, axis=axes[0]).take(idx_y1, axis=axes[1]).take(idx_z0, axis=axes[2])  # noqa: A001
    ie = data.take(idx_x0, axis=axes[0]).take(idx_y0, axis=axes[1]).take(idx_z1, axis=axes[2])
    if_ = data.take(idx_x1, axis=axes[0]).take(idx_y0, axis=axes[1]).take(idx_z1, axis=axes[2])
    ig = data.take(idx_x0, axis=axes[0]).take(idx_y1, axis=axes[1]).take(idx_z1, axis=axes[2])
    ih = data.take(idx_x1, axis=axes[0]).take(idx_y1, axis=axes[1]).take(idx_z1, axis=axes[2])

    if wa.ndim != ia.ndim:
        # Add appropriate dimensions to end
        diff = ia.ndim - wa.ndim
        wa = wa.reshape(*wa.shape, *[1] * diff)
        wb = wb.reshape(*wb.shape, *[1] * diff)
        wc = wc.reshape(*wc.shape, *[1] * diff)
        wd = wd.reshape(*wd.shape, *[1] * diff)
        we = we.reshape(*we.shape, *[1] * diff)
        wf = wf.reshape(*wf.shape, *[1] * diff)
        wg = wg.reshape(*wg.shape, *[1] * diff)
        wh = wh.reshape(*wh.shape, *[1] * diff)

        factor = factor.reshape(*factor.shape, *[1] * diff)

    result = (wa * ia + wb * ib + wc * ic + wd * id + we * ie + wf * if_ + wg * ig + wh * ih) / factor

    return result.reshape(*x.shape, *data.shape[3:])


def nlinear_interpolation(
    data: list[npt.NDArray[np.float64]],
    x: list[t.Union[npt.NDArray[np.float64], float]],
    coords: list[npt.NDArray[np.float64]],
    axes: t.Optional[list[int]] = None,
    mode: t.Literal["zero", "hold"] = "hold",
) -> npt.NDArray[np.float64]:
    """Nlinear interpolation."""
    if len(x) != len(coords):
        raise ValueError("x and coords must have same length")

    if axes is None:
        axes = list(range(len(coords)))

    if len(axes) != len(coords):
        raise ValueError("axes and coords must have same length")

    idx0 = [coord.searchsorted(val, side="right").clip(1, len(coord) - 1) for val, coord in zip(x, coords, strict=True)]

    min_x = [coord.min() for coord in coords]
    max_x = [coord.max() for coord in coords]

    x = [val.clip(min_x, max_x) for val in x]

    idx1 = [idx - 1 for idx in idx0]

    x0 = [coord[idx] for idx, coord in zip(idx0, coords, strict=True)]
    x1 = [coord[idx] for idx, coord in zip(idx1, coords, strict=True)]
    factor = [x1 - x0 for x0, x1 in zip(x0, x1, strict=True)]

    return factor
