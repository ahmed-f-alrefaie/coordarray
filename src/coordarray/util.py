"""Array manipulation utilities."""

import typing as t

import numpy as np
import numpy.typing as npt

T = t.TypeVar("T", bound=np.ndarray)


def broadcast_array_to_shape(
    array: npt.NDArray[t.Any],
    target_shape: tuple[int, ...],
    axis: t.Optional[t.Union[t.Sequence[int], int]] = None,
    broadcast_to: bool = True,
) -> npt.NDArray[t.Any]:
    if axis is None:
        axis_seq: tuple[int, ...] = tuple([1] * array.ndim)
    elif isinstance(axis, int):
        axis_seq = (axis,)
    else:
        axis_seq = tuple(axis)

    ndim = len(target_shape)

    shape = [1] * ndim
    for ax, ar_shape in zip(axis_seq, array.shape, strict=True):
        shape[ax] = ar_shape

    array = array.reshape(*shape)
    if broadcast_to:
        return t.cast(npt.NDArray[t.Any], np.broadcast_to(array, target_shape, subok=True))

    return t.cast(npt.NDArray[t.Any], array)
