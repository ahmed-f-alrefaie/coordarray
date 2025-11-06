"""An class that holds both data and coordinates."""

import typing as t

import numpy as np
import numpy.typing as npt

T = t.TypeVar("T", bound=np.ndarray)
S = t.TypeVar("S", bound=np.ndarray)
_binary_ops = [
    "add",
    "sub",
    "mul",
    "truediv",
    "pow",
    "floordiv",
    "mod",
    "and",
    "or",
    "xor",
    "lshift",
    "rshift",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
]

_unary_ops = [
    "neg",
    "pos",
    "abs",
    "invert",
]


class TransformProtocol(t.Protocol):
    def __call__(self, coordinates: dict[str, S], data: T, *args: t.Any, **kwds: t.Any) -> tuple[dict[str, S], T]: ...


def _binary_op_factory(
    op_name: str,
) -> tuple[
    t.Callable[..., "coordarray"],
    t.Callable[..., "coordarray"],
    t.Callable[..., "coordarray"],
]:
    def _binary_op(self: "coordarray[T,S]", other: t.Union[float, int, npt.ArrayLike, "coordarray"]) -> "coordarray":
        """Binary operation."""
        data = self.data
        new_coords = self.coords
        if isinstance(other, coordarray):
            new_coords, data, other = make_compatible_coords_shape(self, other)

        result = getattr(data, f"__{op_name}__")(other)
        if result is NotImplemented:
            result = getattr(other, f"__r{op_name}__")(data)

        return coordarray(result, new_coords)

    def _inplace_binary_op(
        self: "coordarray[T,S]", other: t.Union[float, int, npt.ArrayLike, "coordarray"]
    ) -> "coordarray":
        """Inplace binary operation."""
        data = self.data
        if isinstance(other, coordarray):
            _, data, other = make_compatible_coords_shape(self, other)

        result = getattr(data, f"__{op_name}__")(other)
        if result is NotImplemented:
            result = getattr(other, f"__r{op_name}__")(data)
        self.data = result
        return self

    def _reflected_binary_op(
        self: "coordarray[T,S]", other: t.Union[float, int, npt.ArrayLike, "coordarray"]
    ) -> "coordarray":
        """Reflected binary operation."""
        data = self.data
        new_coords = self.coords
        if isinstance(other, coordarray):
            new_coords, other, data = make_compatible_coords_shape(other, self)

        result = getattr(data, f"__r{op_name}__")(other)

        return coordarray(result, new_coords)

    return _binary_op, _inplace_binary_op, _reflected_binary_op


def _unary_op_factory(op_name: str) -> t.Callable[..., "coordarray"]:
    def _unary_op(self: "coordarray") -> "coordarray":
        """Unary operation."""
        data = self.data
        result = getattr(data, f"__{op_name}__")()
        return coordarray(result, self.coords)

    return _unary_op


def _matching_coords(coords1: dict[str, S], coords2: dict[str, S]) -> set[str]:
    """Check if any coordinates match.

    Args:
        coords1: First set of coordinates
        coords2: Second set of coordinates

    Returns:
        coordinates that match
    """
    return set(coords1.keys()).intersection(set(coords2.keys()))


def _all_matching_coords(coords1: t.Sequence[str], coords2: t.Sequence[str]) -> bool:
    """Check if all coordinates match.

    Args:
        coords1: First set of coordinates
        coords2: Second set of coordinates

    Returns:
        True if all coordinates match
    """
    return set(coords1) == set(coords2)


def same_coords(array1: "coordarray[T,S]", array2: "coordarray[T,S]") -> bool:
    """Check if two coordarrays have the same coordinates.

    Args:
        array1: First array
        array2: Second array

    Returns:
        True if they have the same coordinates
    """
    same_keys = array1.coords.keys() == array2.coords.keys()
    try:
        same_values = all(np.array_equal(array1.coords[k], array2.coords[k]) for k in array1.coords)
    except KeyError:
        return False

    return same_keys and same_values


def is_increasing(array: T) -> bool:
    """Check if array is increasing.

    Args:
        array: Array to check

    Returns:
        True if array is increasing
    """
    return bool(
        np.all(np.diff(array) > 0 * array[0])
    )  # If array contains units, multiply by first element to keep units consistent


def ensure_same_coords_ordered(
    array1: "coordarray[T,S]",
    array2: "coordarray[T,S]",
) -> "coordarray[T,S]":
    """Ensure two coordarrays have the same ordering."""
    for k, v in array1.array_coords.items():
        if k not in array2.array_coords:
            continue
        array1_order = is_increasing(v)
        array2_order = is_increasing(array2.array_coords[k])
        if array1_order != array2_order:
            array2 = array2.sort_coord(k, reverse=not array1_order)
    return array2


def _new_axes_layout(axes1: t.Sequence[str], axes2: t.Sequence[str]) -> t.Sequence[str]:
    """New axes layout.

    Prioritize the first array axes then add second axes that are not in first array

    Args:
        axes1: First axes
        axes2: Second axes

    Returns:
        New axes layout

    """
    new_axes = list(axes1)
    for ax in axes2:
        if ax not in new_axes:
            new_axes.append(ax)
    return new_axes


def make_compatible_coords_shape(
    array1: "coordarray[T,S]",
    array2: "coordarray[T,S]",
) -> tuple[dict[str, S], T, T]:
    """Make two coordarrays compatible.

    First we determine which coordinates match. Then we reorder the second
    array to match the first. Finally we broadcast the coordinates to the
    any missing coords for both arrays. Prioritizing the first array.



    Args:
        array1: First array
        array2: Second array

    Returns:
        New coordinates, broadcasted array1, broadcasted array2
    """
    from .util import broadcast_array_to_shape

    array1_axes = array1.array_axes

    array2 = array2.transpose(*array1_axes, mode="ignore")
    array2 = ensure_same_coords_ordered(array1, array2)
    if same_coords(array1, array2):
        return array1.coords, array1.data, array2.data
    array1_coords = array1.array_coords
    array2_coords = array2.array_coords

    array1_axes = array1.array_axes
    array2_axes = array2.array_axes

    unique_array2_axes = tuple(x for x in array2_axes if x not in array1_axes)

    unique_array2_shape = tuple(array2.shape[array2._coord_to_axis(x)] for x in unique_array2_axes)
    target_axes = tuple(array1_axes) + unique_array2_axes
    target_shape = array1.shape + unique_array2_shape
    target_coords = {k: array1_coords[k] if k in array1_coords else array2_coords[k] for k in target_axes}

    remap_arr1 = tuple(target_axes.index(x) for x in array1_axes)
    remap_arr2 = tuple(target_axes.index(x) for x in array2_axes)

    if array1.ndim == 0:
        return target_coords, array1.data, array2.data

    new_arr1 = broadcast_array_to_shape(array1.data, target_shape, axis=remap_arr1, broadcast_to=False)
    new_arr2 = broadcast_array_to_shape(array2.data, target_shape, axis=remap_arr2, broadcast_to=False)

    return target_coords, t.cast(T, new_arr1), t.cast(T, new_arr2)


class coordarray(t.Generic[T, S]):
    """An array that holds both data and coordinates.

    The concept is to abstract out the need to broadcast coordinates
    to the data. This is useful for when you have a 1D array of data
    and a 2D array of coordinates. This is a common case in radiative
    transfer calculations.

    For example we want to acheive the following:

    >>> data = np.random.rand(10)
    >>> coord = np.linspace(0, 1, 10)
    >>> coord_array = CoordArray(data, coord)

    >>> coord_array.shape

    """

    data: T
    coords: dict[str, S]

    # @classmethod
    # def generate_from_func(
    #     cls,
    #     coords: dict[str, S],
    #     func: t.Callable[..., npt.ArrayLike],
    #     function_axes: t.Sequence[str],
    #     arrays: t.Sequence["coordarray"],
    #     ensure_order: t.Optional[bool] = True,
    #     args: t.Optional[t.Sequence[t.Any]] = (),
    #     **kwargs: dict[str, t.Any],
    # ) -> "coordarray":
    #     """Generate a coordarray from a function.

    #     Args:
    #         coords: Coordinates
    #         func: Function to use
    #         function_axes: Axes to apply function over
    #         arrays: Arrays to map
    #         ensure_order: Ensure order of coordinates
    #         args: Arguments for function
    #         kwargs: Keyword arguments for function

    #     Returns:
    #         coordarray

    #     Raises:
    #         ValueError: If the function axes are not in the arrays
    #         NotImplementedError: If you try to generate a coordarray from a function
    #     """
    #     import itertools

    #     axes_to_map = []
    #     for arr in arrays:
    #         axes_to_map = _new_axes_layout(axes_to_map, arr.axes)

    #     if not axes_to_map:
    #         raise ValueError("No axes to map")

    #     if not set(function_axes).issubset(set(axes_to_map)):
    #         raise ValueError("Combination of arrays do not contain function axes")

    #     axes_to_map = tuple(x for x in axes_to_map if x not in function_axes)
    #     axes_lengths = {}
    #     axes_set = set(axes_to_map)
    #     for arr in arrays:
    #         axes_lengths = {
    #             **axes_lengths,
    #             **{k: v.size for k, v in arr.array_coords.items() if k in axes_set},
    #         }
    #         if set(axes_lengths.keys()) == axes_set:
    #             break

    #     axes_lengths = [axes_lengths[x] for x in axes_to_map]

    #     for indices in itertools(*(range(x) for x in axes_lengths)):
    #         array_slices = list(zip(axes_to_map, indices, strict=True))
    #         array_args = []
    #         for arr in arrays:
    #             arr_val = arr
    #             for ax, idx in array_slices:
    #                 if ax in arr_val.axes:
    #                     arr_val = arr_val.take(ax, idx)
    #             array_args.append(arr_val)
    #         array_args = tuple(array_args)

    #     raise NotImplementedError

    #     return None

    def __init__(
        self,
        data: t.Union["coordarray", T],
        coords: t.Optional[dict[str, S]] = None,
        **kwargs: dict[str, S],
    ) -> None:
        """Create a coordarray."""
        data_value = data.data if isinstance(data, coordarray) else data

        self.data = np.asanyarray(data_value)
        coords = coords or {}
        self.coords: dict[str, S] = {
            **{k: t.cast(S, np.asanyarray(v)) for k, v in kwargs.items()},
            **{k: t.cast(S, np.asanyarray(v)) for k, v in coords.items()},
        }

        for k, coord in coords.items():
            if coord.ndim > 1:
                raise ValueError("Coordinate must be 1D")
            if isinstance(coord, coordarray):
                self.coords[k] = coord.data
        if len(self.array_coords) != data.ndim:
            raise ValueError(
                f"Number of coordinates ({len(coords)}) must match number of dimensions of data ({data.ndim})"
            )

        for shape, coord in zip(data.shape, self.array_coords.values(), strict=True):
            if shape != coord.shape[0]:
                raise ValueError(f"Coordinate shape {coord.shape} does not match data shape {shape}")
            if np.sort(np.unique(coord)).size != coord.size:
                raise ValueError("Coordinate must be unique")

    @property
    def value(self) -> T:
        """Get the data value."""
        return self.data

    @property
    def axes(self) -> tuple[str, ...]:
        """Dimensions of data."""
        return tuple(self.coords.keys())

    @property
    def scalar_axes(self) -> tuple[str, ...]:
        """Dimensions of data."""
        array_axes = self.array_axes
        return tuple(x for x in self.axes if x not in array_axes)

    @property
    def scalar_coords(self) -> dict[str, S]:
        """Dimensions of data."""
        return {k: v for k, v in self.coords.items() if v.ndim == 0}

    @property
    def array_axes(self) -> tuple[str, ...]:
        """Dimensions of data."""
        return tuple(self.array_coords.keys())

    @property
    def array_coords(self) -> dict[str, S]:
        """Dimensions of data."""
        return {k: v for k, v in self.coords.items() if v.ndim > 0}

    @property
    def _slicable_coords(self) -> dict[str, S]:
        """Get coordinates that can be sliced."""
        return self.array_coords

    # This method is for when numpy function needs to convert your object to ndarray
    def __array__(self, dtype: t.Optional[np.dtype] = None) -> npt.NDArray[t.Any]:
        """Convert to numpy array."""
        return np.asarray(self.data, dtype=dtype)

    # def to(
    #     self, unit: u.Unit, equivalencies: t.Optional[u.Unit] = None
    # ) -> "coordarray":
    #     """Convert to unit.

    #     Args:
    #         unit: Unit to convert to
    #         equivalencies: Equivalencies to use

    #     Returns:
    #         coordarray with new units

    #     """
    #     return coordarray(self.data.to(unit, equivalencies), self.coords)

    def transpose(self, *axes: str, mode: t.Optional[t.Literal["raise", "ignore"]] = "ignore") -> "coordarray[T,S]":
        """Transpose data and coordinates.

        Args:
            axes: Axes to transpose
            mode: Mode to use when transposing

        Returns:
            coordarray with transposed data and coordinates

        Raises:
            ValueError: If axes are not in coordinates and mode is ``raise``
        """
        if not axes:
            axes = self.array_axes[::-1]

        if mode == "raise":
            leftover = set(axes) - (set(axes) & set(self.array_axes))
            if leftover:
                raise ValueError(f"Cannot transpose axes {leftover}")

        axes_list = [x for x in axes if x in self.array_axes]

        if len(axes_list) < 2 or axes_list == self.array_axes:
            return coordarray(self.data.copy(), self.coords)

        new_axes = tuple(self._coord_to_axis(x) for x in axes_list)

        # Reorder coords
        coords_list = list(self.array_coords.items())
        new_coords_list = [coords_list[x] for x in new_axes]
        new_coords = dict(new_coords_list)

        new_coords = {
            **self.scalar_coords,
            **new_coords,
        }

        leftover_axes = tuple(sorted(set(range(self.ndim)) - (set(new_axes))))

        leftover_coords = {k: v for idx, (k, v) in enumerate(self.array_coords.items()) if idx in leftover_axes}

        return coordarray(
            self.data.transpose(new_axes + leftover_axes),
            {**new_coords, **leftover_coords},
        )

    def cumsum(self, axis: t.Optional[str] = None) -> t.Union["coordarray[T,S]", T]:
        """Cumulative sum."""
        if axis is None:
            return t.cast(T, self.data.cumsum())
        return coordarray(t.cast(T, self.data.cumsum(axis=self._coord_to_axis(axis))), self.coords)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of data."""
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions of data."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Size of data."""
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        """Data type of data."""
        return self.data.dtype

    def _coord_to_axis(self, coord: str) -> int:
        """Convert coord name to axis."""
        return list(self.array_coords.keys()).index(coord)

    def _broadcast_coord_to_target_shape(
        self,
        coord: str,
        target_shape: t.Optional[tuple[int, ...]] = None,
        axis: t.Optional[int] = None,
    ) -> T:
        """Broadcast coordinate to data shape."""
        from .util import broadcast_array_to_shape

        coord_axis = axis
        if axis is None:
            coord_axis = self._coord_to_axis(coord)

        if target_shape is None:
            target_shape = self.shape

        return t.cast(T, broadcast_array_to_shape(self.coords[coord], self.data.shape, axis=coord_axis))

    def __getitem__(  # noqa: C901
        self, key: t.Union[str, t.Sequence[t.Union[int, slice, None]]]
    ) -> t.Union["coordarray", T]:
        """Get item."""
        if isinstance(key, str):
            return self._broadcast_coord_to_target_shape(key)
        elif isinstance(key, (list, tuple, np.ndarray)):
            if any(x is None for x in key):
                raise ValueError("None is not allowed in slices with coord array")
            total_keys = len(key)

            if total_keys == 2 and isinstance(key[0], str):
                coord = key[0]
                slice_var = key[1]
                axis = self._coord_to_axis(coord)

                if isinstance(slice_var, int) and slice_var < 0:
                    slice_var = self.data.shape[axis] + slice_var

                new_coords = {
                    k: v if idx != axis else v[slice_var] for idx, (k, v) in enumerate(self.array_coords.items())
                }
                return self.take(coord, slice_var, as_array=False)
            else:
                slices = [x for x in key if x is not Ellipsis]

                if Ellipsis in key:
                    ellipsis_idx = key.index(Ellipsis)
                    leftover_dims = max(self.data.ndim - total_keys, 0)
                    if ellipsis_idx == 0:
                        slices = [slice(None) for n in range(leftover_dims)] + slices
                    elif ellipsis_idx == total_keys - 1:
                        slices = slices + [slice(None)] * leftover_dims
                    else:
                        raise ValueError("Ellipsis must be at the start or end of slice")

                if len(slices) > self.data.ndim:
                    raise ValueError("Too many slices")

                new_coords = {
                    k: v[slice_var] for slice_var, (k, v) in zip(enumerate(self.array_coords.items(), slices))
                }
                return coordarray(self.data[key], new_coords)
        elif isinstance(key, int):
            coords = {k: v[key] if idx == 0 else v for idx, (k, v) in enumerate(self.coords.items())}

            return coordarray(self.data[key], coords=coords)
        elif isinstance(key, np.ndarray):
            if key.ndim == 1:
                new_coords = {k: v[key] if idx == 0 else v for idx, (k, v) in enumerate(self.coords.items())}
                return coordarray(self.data[key], coords=new_coords)

            return self.data[key]
        else:
            return self.data[key]

    def __setitem__(self, key: str, value: t.Any) -> None:  # noqa: C901
        """Set an item in the array."""
        if isinstance(key, str):
            self.coords[key] = value

        elif isinstance(key, (list, tuple, np.ndarray)):
            if None in key:
                raise ValueError("None is not allowed in slices with coord array")
            total_keys = len(key)
            has_str = any(isinstance(x, str) for x in key)
            if has_str:
                # Check if in pairs
                if total_keys & 1:
                    raise ValueError("Setting coordinates must be in pairs")
                accessor = [slice(None)] * self.data.ndim

                for coord, slice_var in zip(key[::2], key[1::2], strict=True):
                    if isinstance(slice_var, int):
                        slice_var = slice(slice_var, slice_var + 1)
                    axis = self._coord_to_axis(coord)
                    accessor[axis] = slice_var

                if isinstance(value, coordarray):
                    self.data.__setitem__(*accessor, value.data)
                else:
                    self.data.__setitem__(*accessor, value)
                return
            slices = [x for x in key if x is not Ellipsis]
            if Ellipsis in key:
                ellipsis_idx = key.index(Ellipsis)
                leftover_dims = max(self.data.ndim - total_keys, 0)
                if ellipsis_idx == 0:
                    slices = [slice(None)] * leftover_dims + slices
                elif ellipsis_idx == total_keys - 1:
                    slices = slices + [slice(None)] * leftover_dims
                else:
                    raise ValueError("Ellipsis must be at the start or end of slice")

            if len(slices) > self.data.ndim:
                raise ValueError("Too many slices")

        elif isinstance(key, int):
            self.data[key] = value
        else:
            return self.data.__setitem__(key, value)

    def match_coordinates(self, coords: t.Union["coordarray", dict[str, S]]) -> "coordarray":
        """Match same axes and coordinates."""
        array = None
        for name, coord in coords.items():
            if name in self.array_coords:
                coordinate = self.coords[name]
                where = np.where(np.isin(coordinate, coord, assume_unique=True))[0]
                if array is None:
                    array = self.take(name, where, as_array=False)
                else:
                    array = array.take(name, where, as_array=False)

        return array

    def sum(self, axis: t.Union[str, int, t.Optional[t.Sequence[t.Union[int, str]]]] = None) -> "coordarray":
        """Sum over axis.

        Args:
            axis: Axis to sum over

        Returns:
            coordarray
        """
        # Construct axis
        if axis is None:
            return self.data.sum()

        if isinstance(axis, str):
            axis = (self._coord_to_axis(axis),)
        elif isinstance(axis, t.Sequence):
            axis = tuple(self._coord_to_axis(ax) if isinstance(ax, str) else ax for ax in axis)

        coords = {k: v for idx, (k, v) in enumerate(self.array_coords.items()) if idx not in axis}
        return coordarray(self.data.sum(axis=axis), coords)

    def squeeze(self, axis: t.Optional[t.Union[str, int]] = None) -> "coordarray":
        """Squeeze array.

        Args:
            axis: Axis to squeeze

        Returns:
            coordarray with squeezed data

        """
        if axis is None:
            return coordarray(self.data.squeeze(), self.coords)
        if isinstance(axis, str):
            axis = self._coord_to_axis(axis)
        coords = {k: v if idx != axis else v.squeeze() for idx, (k, v) in enumerate(self.array_coords.items())}
        return coordarray(self.data.squeeze(axis=axis), coords)

    def take(
        self,
        axis: str,
        indices: t.Optional[t.Union[t.Sequence, npt.ArrayLike, slice]] = None,
        at: t.Optional[t.Literal["min", "max"]] = None,
        as_array: t.Optional[bool] = True,
    ) -> t.Union[T, "coordarray"]:
        """Take along axis.

        Args:
            axis: Axis to take along
            indices: Indices to take
            at: Take at min or max
            as_array: Return as array

        Returns:
            coordarray

        Raises:
            ValueError: If neither ``indices`` or ``at`` is specified
        """
        axis = self._coord_to_axis(axis)
        if indices is None:
            if at is None:
                raise ValueError("Either indices or 'at' must be specified")
            if at == "min":
                indices = (self.coords[axis].argmin(),)
            elif at == "max":
                indices = (self.coords[axis].argmax(),)

        coords = {k: v[indices] if idx == axis else v for idx, (k, v) in enumerate(self.array_coords.items())}
        if isinstance(indices, slice):
            if indices == slice(None):
                return self
            indices = np.arange(0, self.shape[axis])[indices]
        result = self.data.take(indices, axis=axis)
        if as_array:
            return result
        else:
            return coordarray(result, coords)

    def sort_coord(self, axis: str, reverse: t.Optional[bool] = False) -> "coordarray[T,S]":
        """Sort along axis.

        Args:
            axis: Axis to sort along
            reverse: Reverse sort

        Returns:
            coordarray
        """
        coord_data = self.coords[axis]
        axis = self._coord_to_axis(axis)
        indices = np.argsort(coord_data)[::-1] if reverse else np.argsort(coord_data)
        coords = {k: v if idx != axis else v[indices] for idx, (k, v) in enumerate(self.array_coords.items())}
        return coordarray(self.data.take(indices, axis), coords)

    def as_array(self, copy=False) -> T:
        """Return data as array."""
        if copy:
            return self.data.copy()
        return self.data

    def interp_match(
        self,
        coords: t.Union["coordarray[T,S]", dict[str, S]],
    ) -> "coordarray":
        """Interpolate to match another coordarray's coordinates.

        Args:
            coords: Coordinates to match
        Returns:
            coordarray interpolated to match coordinates
        Raises:
            ValueError: If no matching coordinates to interpolate


        """
        from scipy.interpolate import make_interp_spline

        if isinstance(coords, coordarray):
            coords = coords.array_coords

        same_coords = _matching_coords(self.array_coords, coords)

        if not same_coords:
            raise ValueError("No matching coordinates to interpolate")

        interp_coords = {k: coords[k] for k in same_coords}
        new_data = self.data

        current_coords = self.coords.copy()

        for k, coord in interp_coords.items():
            spline = make_interp_spline(self.coords[k], new_data, k=1, axis=self._coord_to_axis(k))
            new_data = spline(coord)
            current_coords[k] = coord

        return coordarray(new_data, current_coords)

    def interp_to(
        self,
        coords: t.Optional[dict[str, "coordarray"]] = None,
        **kwargs: "coordarray",
    ) -> "coordarray":
        """Interpolate to new coordinate system."""
        from .interpolation import bilinear_interpolate, linear_interpolation

        if coords is None:
            coords = kwargs

        coord_values = list(coords.values())
        first = coord_values[0].coords

        if not all(_all_matching_coords(c.coords, first) for c in coord_values):
            raise ValueError("All coordinates must match in order to interpolate to a new coordinate system")

        new_coords = {**self.coords}
        for k in coords:
            new_coords.pop(k, None)

        new_coords = {**first, **new_coords}

        interpolation_axes = tuple(self._coord_to_axis(x) for x in coords)

        interpolation_function = bilinear_interpolate
        if len(coords) == 1:
            interpolation_function = linear_interpolation
        elif len(coords) > 2:
            raise NotImplementedError("Only linear and bilinear interpolation is supported")

        interpolation_coords = tuple(self.coords[x] for x in coords)

        new_data = interpolation_function(
            self.data,
            *tuple(c.data for c in coords.values()),
            *interpolation_coords,
            axes=interpolation_axes,
        )

        return coordarray(new_data, new_coords)

    # This is for operations like np.square, np.add, etc.
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: t.Any, **kwargs: t.Any):
        """Array ufunc."""
        new_inputs: list[t.Any] = [(inp.data if isinstance(inp, coordarray) else inp) for inp in inputs]

        if method == "__call__":
            outputs = kwargs.pop("out", None)
            if outputs:
                kwargs["out"] = tuple((x.data if isinstance(x, coordarray) else x) for x in outputs)
            else:
                outputs = (None,) * ufunc.nout

            results = ufunc(*new_inputs, **kwargs)
            if ufunc.nout == 1:
                results = (results,)

            results = tuple(
                (coordarray(result, self.coords) if output is None else output)
                for result, output in zip(results, outputs, strict=True)
            )
            return results[0] if len(results) == 1 else results
        elif method == "reduce":
            axis = kwargs.pop("axis", None)

            outputs = kwargs.pop("out", None)
            if outputs:
                kwargs["out"] = tuple((x.data if isinstance(x, coordarray) else x) for x in outputs)
            else:
                outputs = (None,) * ufunc.nout

            if axis is None:
                return ufunc.reduce(self.data, **kwargs)
            else:
                coord = self.array_axes[axis]
                coord_data = self.coords.copy()
                coord_data.pop(coord, None)

                results = ufunc(*new_inputs, **kwargs)
                if ufunc.nout == 1:
                    results = (results,)

                results = tuple(
                    (coordarray(result, self.coords) if output is None else output)
                    for result, output in zip(results, outputs, strict=True)
                )
                return results[0] if len(results) == 1 else results

                return coordarray(
                    ufunc.reduce(self.data, **kwargs),
                    coord_data,
                )

        else:
            raise NotImplementedError(f"Not implemented {method}")

    def __repr__(self) -> str:
        """Representation."""
        return f"Coordinate Array\n data: {self.data.__repr__()}\n coords: {self.coords}"

    def apply_coordinate_transform(
        self,
        coords: t.Sequence[str],
        func: TransformProtocol,
        ravel_other_dimensions: t.Optional[bool] = True,
        args: tuple[t.Any, ...] = (),
        kwargs: t.Optional[dict[str, t.Any]] = None,
    ) -> "coordarray[T,S]":
        """Apply a transformation to the coordinates.

        Args:
            coords: Coordinates to transform
            func: Transformation function
            ravel_other_dimensions: Ravel other dimensions

        Returns:
            Transformed coordarray
        """
        kwargs = kwargs or {}
        check_if_coords_exist = all(k in self.coords for k in coords)
        if not check_if_coords_exist:
            raise ValueError("Coordinates to transform must exist in coordarray")

        # Transpose so that coords to transform are first
        new_axes = list(coords) + [k for k in self.array_axes if k not in coords]
        transposed = self.transpose(*new_axes)
        data = transposed.data
        coord_data = transposed.array_coords
        if ravel_other_dimensions:
            # Ravel other dimensions
            shape_to_ravel = (np.prod(data.shape[len(coords) :]),)
            new_shape = data.shape[: len(coords)] + shape_to_ravel
            data = data.reshape(new_shape)

        new_coord_data, new_data = func({k: coord_data[k] for k in coords}, data, *args, **kwargs)

        # Reshape back to original shape
        if ravel_other_dimensions:
            num_dims = new_data.ndim
            preserved_shape = data.shape[len(coords) :]
            num_dims_to_reshape = num_dims - len(coords)
            new_shape = new_data.shape[:num_dims_to_reshape] + preserved_shape
            new_data = new_data.reshape(new_shape)

        new_coords = {**new_coord_data, **{k: v for k, v in coord_data.items() if k not in coords}}
        return coordarray(new_data, new_coords).transpose(*self.array_axes)


for op_name in _binary_ops:
    binary_op, inplace_binary_op, reflected_binary_op = _binary_op_factory(op_name)
    setattr(coordarray, f"__{op_name}__", binary_op)
    setattr(coordarray, f"__i{op_name}__", inplace_binary_op)
    setattr(coordarray, f"__r{op_name}__", reflected_binary_op)

for op_name in _unary_ops:
    unary_op = _unary_op_factory(op_name)
    setattr(coordarray, f"__{op_name}__", unary_op)
